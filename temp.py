# temp.py

import sys
import torch
from torch.utils.data import DataLoader

# adjust this import if your iddaw_dataset.py lives elsewhere
from datasets.iddaw_dataset import IDDAWDataset  

from hesp.config.config import Config
from hesp.hierarchy.tree import Tree
from hesp.models.embedding_functions.segformer import segformer_b0
from hesp.models.model import ModelFactory

# temp.py

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from hesp.config.config import Config
from hesp.hierarchy.tree import Tree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HuggingFace model ID for SegFormer-B0"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for a quick smoke test"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Resize images to this size"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # 1) DataLoader – exercise the *entire* train set to find true label range
    ds = IDDAWDataset(root_dir="datasets/IDDAW", mode="train", image_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    overall_min, overall_max = 999, -1
    for imgs, masks in loader:
        overall_min = min(overall_min, int(masks.min()))
        overall_max = max(overall_max, int(masks.max()))
    print(f">>> Dataset label range on TRAIN split: {overall_min} – {overall_max}\n")

    # Grab one batch for shape printing
    imgs, masks = next(iter(loader))
    print(">>> One‐batch shapes:")
    print("  images:", imgs.shape, imgs.dtype)
    print("  masks: ", masks.shape, masks.dtype, "\n")

    # 2) Build Config & flat Tree
    cfg = Config(dataset="iddaw", mode="segmenter", gpu_idx=0)
    cfg.embedding_space._HIERARCHICAL = False
    tree = Tree(i2c=cfg.dataset._I2C, json={})

    # 3) Instantiate SegFormer‑B0
    print(f">>> Loading SegFormer‑B0 backbone: {args.model_id}")
    backbone = SegformerForSemanticSegmentation.from_pretrained(
        args.model_id,
        ignore_mismatched_sizes=True,
        num_labels=tree.M
    ).to(device).eval()

    # 4) Forward pass
    imgs_dev = imgs.to(device)
    with torch.no_grad():
        out = backbone(pixel_values=imgs_dev)
    logits = out.logits  # (B, M, H', W')

    print("\n>>> Backbone forward shapes:")
    print("  logits:", logits.shape)
    if hasattr(out, "hidden_states"):
        print("  last hidden:", out.hidden_states[-1].shape)

    # 5) Parameter counts
    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\n>>> Params: total={total:,}, trainable={trainable:,}\n")

    # 6) Softmax sanity check
    B, C, Hc, Wc = logits.shape
    center_probs = torch.softmax(logits, dim=1)[0, :, Hc//2, Wc//2]
    print(f"Sum of softmax@center pixel: {center_probs.sum():.5f}\n")

    print("✅ All checks complete.")

if __name__ == "__main__":
    main()
