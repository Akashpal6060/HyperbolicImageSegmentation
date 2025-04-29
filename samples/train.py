import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from hesp.config.config import Config
from hesp.config.dataset_config import DATASET_CFG_DICT
from hesp.models.model import ModelFactory
from datasets.iddaw_dataset import IDDAWDataset  # adjust import if needed

# --- setup logging ---
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for imgs, masks in loop:
        imgs = imgs.to(device)
        masks = masks.to(device)

        out = model(imgs)
        loss = model.compute_loss(out["probabilities"], masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def evaluate(model, loader, device, epoch):
    model.eval()
    correct = total = 0
    loop = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)
    with torch.no_grad():
        for imgs, masks in loop:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model.predict(imgs)
            correct += (preds == masks).sum().item()
            total += masks.numel()
            loop.set_postfix(acc=f"{correct/total:.4f}")
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="PyTorch Hyperbolic SegFormer Trainer")
    parser.add_argument("--mode",     choices=["segmenter"], required=True)
    parser.add_argument("--dataset",  choices=DATASET_CFG_DICT.keys(), required=True)
    parser.add_argument("--geometry", choices=["euclidean","hyperbolic"], default="hyperbolic")
    parser.add_argument("--dim",      type=int,   default=256)
    parser.add_argument("--c",        type=float, default=1.0)
    parser.add_argument("--flat",     action="store_true")
    parser.add_argument("--freeze_bb", action="store_true")
    parser.add_argument("--freeze_bn", action="store_true")
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--num_epochs", type=int,   default=50)
    parser.add_argument("--slr",        type=float, default=1e-3)
    parser.add_argument("--pretrained", type=str,   default="nvidia/segformer-b5-finetuned-ade-640-640")
    parser.add_argument("--output_stride", type=int, choices=[8,16], default=16)
    parser.add_argument("--gpu",         type=int,   default=0)
    parser.add_argument("--transform",   action="store_true")
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--test",        action="store_true")
    args = parser.parse_args()

    # --- build config ---
    config = Config(
        dataset=args.dataset,
        mode=args.mode,
        base_save_dir=os.getcwd(),
        gpu_idx=args.gpu
    )
    config.embedding_space._GEOMETRY       = args.geometry
    config.embedding_space._DIM            = args.dim
    config.embedding_space._INIT_CURVATURE = args.c
    config.embedding_space._HIERARCHICAL   = not args.flat

    config.segmenter._PRETRAINED_MODEL     = args.pretrained
    config.segmenter._FREEZE_BACKBONE      = args.freeze_bb
    config.segmenter._FREEZE_BN            = args.freeze_bn
    config.segmenter._OUTPUT_STRIDE        = args.output_stride
    config.segmenter._BATCH_SIZE           = args.batch_size
    config.segmenter._INITIAL_LEARNING_RATE= args.slr
    config.segmenter._NUM_EPOCHS           = args.num_epochs
    config.segmenter._NUM_TRAIN            = 0  # not used here

    logger.info("Configuration:")
    config.pretty_print()

    # --- device ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- data ---
    train_ds = IDDAWDataset(root_dir=os.path.join("datasets", args.dataset), mode="train", image_size=512)
    logger.info(f"🧾 Found {len(train_ds)} training samples.")
    if len(train_ds) == 0:
        raise ValueError("🚨 Your training dataset is empty! Check root_dir, mode, or folder structure.")
    val_ds = IDDAWDataset(root_dir=os.path.join("datasets", args.dataset), mode="val", image_size=512)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # --- model ---
    model = ModelFactory.create(model_name=args.mode, cfg=config)
    model.to(device)

    # --- optimizer & scheduler ---
    optimizer = SGD(
        [
            {'params': model.backbone.parameters(), 'lr': args.slr * 0.1},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone")], 'lr': args.slr},
        ],
        momentum=0.9,
        weight_decay=1e-4,
    )
    total_iters = len(train_loader) * args.num_epochs
    scheduler = PolynomialLR(optimizer, total_iters=total_iters, power=0.9, last_epoch=-1)

    # --- tensorboard writer ---
    tb_log_dir = os.path.join(os.getcwd(), "tb_logs")
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logs at: {tb_log_dir}")

    # --- training loop ---
    if args.train:
        best_acc = 0.0
        for epoch in trange(1, args.num_epochs + 1, desc="Epoch", leave=True):
            loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            acc  = evaluate(model, val_loader, device, epoch)

            # log scalars
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Acc/val",   acc,  epoch)

            logger.info(f"Epoch {epoch}/{args.num_epochs} — loss: {loss:.4f}, val acc: {acc:.4f}")

            # save checkpoints
            save_dir = model.config.segmenter_save_dir
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch{epoch:03}.pth"))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

        writer.close()

    # --- testing ---
    if args.test:
        ckpt = os.path.join(model.config.segmenter_save_dir, "best.pth")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        test_acc = evaluate(model, val_loader, device, epoch="Test")
        logger.info(f"Test accuracy: {test_acc:.4f}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
