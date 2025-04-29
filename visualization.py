# visualization.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from hesp.config.config import Config
from hesp.config.dataset_config import DATASET_CFG_DICT
from hesp.models.model import ModelFactory
from datasets.iddaw_dataset import IDDAWDataset


def save_visualization(img, pred, gt, out_dir, idx, num_classes):
    img = TF.to_pil_image(img.squeeze(0))
    pred = pred.squeeze(0).cpu().numpy()
    gt   = gt.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title("Input");       axes[0].axis("off")
    axes[1].imshow(pred, cmap="tab20", vmin=0, vmax=num_classes-1)
    axes[1].set_title("Prediction"); axes[1].axis("off")
    axes[2].imshow(gt,   cmap="tab20", vmin=0, vmax=num_classes-1)
    axes[2].set_title("Ground Truth"); axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"val_{idx:03}.png"), dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("GPU-accelerated visualization")
    parser.add_argument("--mode",       choices=["segmenter"],     required=True)
    parser.add_argument("--dataset",    choices=DATASET_CFG_DICT.keys(), required=True)
    parser.add_argument("--geometry",   choices=["euclidean","hyperbolic"], default="hyperbolic")
    parser.add_argument("--dim",        type=int,   default=256)
    parser.add_argument("--c",          type=float, default=1.0)
    parser.add_argument("--flat",       action="store_true")
    parser.add_argument("--freeze_bb",  action="store_true")
    parser.add_argument("--freeze_bn",  action="store_true")
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--pretrained", type=str,   default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--output_stride", type=int, choices=[8,16], default=16)
    parser.add_argument("--gpu",        type=int,   default=0)
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Optional override checkpoint path. Defaults to <save_dir>/best.pth")
    parser.add_argument("--out_dir",    type=str,   default="visualization_out")
    parser.add_argument("--num_samples",type=int,   default=20)
    args = parser.parse_args()

    # GPU setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    cudnn.benchmark = True
    print(f"Using device: {device}")

    # Build config (must match your training invocation)
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
    config.segmenter._NUM_TRAIN            = 0

    # Validation loader
    val_ds = IDDAWDataset(
        root_dir=os.path.join("datasets", args.dataset),
        mode="val",
        image_size=512
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"🧾 Loaded {len(val_ds)} image-mask pairs from val set.")

    # Instantiate model and derive checkpoint path
    model = ModelFactory.create(model_name=args.mode, cfg=config).to(device)
    default_ckpt = os.path.join(model.config.segmenter_save_dir, "best.pth")
    ckpt_path = args.checkpoint or default_ckpt

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found:\n  {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✅ Loaded checkpoint: {ckpt_path}")

    # Infer number of classes (fallback to 30 if not exposed)
    num_classes = getattr(val_ds, "num_classes", None) or 30

    # Run inference & save visualizations
    with torch.no_grad():
        for idx, (img, gt_mask) in enumerate(val_loader):
            if idx >= args.num_samples:
                break
            img     = img.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)
            pred    = model.predict(img)  # stays on GPU

            save_visualization(
                img.cpu(), pred, gt_mask,
                out_dir=args.out_dir,
                idx=idx,
                num_classes=num_classes
            )

    print(f"Saved {min(len(val_ds), args.num_samples)} visualizations → {args.out_dir}")


if __name__ == "__main__":
    main()
