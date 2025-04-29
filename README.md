# Hyperbolic Image Segmentation with SegFormer-B0 on IDDAW Dataset

This repository contains a PyTorch-based implementation of **Hyperbolic Image Segmentation** adapted from [CVPR 2022 paper](https://arxiv.org/pdf/2203.05898.pdf), with the following custom modifications:

-  Replaced ResNet with **SegFormer-B0** backbone.
-  Adapted for the **IDDAW dataset** (Indian Driving Dataset for Adverse Weather).
-  Implemented training and visualization in **hyperbolic space** (Poincaré ball).
-  Uses hierarchical supervision for structured semantic segmentation.


---

##  Repository Structure

- **datasets/** – class mappings, label hierarchies, and utility JSONs.
- **hesp/** – model definitions, loss functions, embedding logic, and hierarchy utilities.
- **samples/** – training scripts, helper functions, and experiment configs.

---

## ⚙ Installation

```bash
# Clone the repository
git clone https://github.com/Akashpal6060/HyperbolicImageSegmentation.git
cd HyperbolicImageSegmentation

# Install package in editable mode
pip install -e .

# Install dependencies
sh requirements.sh




CUDA_VISIBLE_DEVICES=6 \
nohup python samples/train.py \
  --mode segmenter \
  --dataset IDDAW \
  --geometry hyperbolic \
  --dim 256 \
  --c 1.0 \
  --batch_size 64 \
  --slr 1e-3 \
  --num_epochs 500 \
  --output_stride 16 \
  --pretrained nvidia/segformer-b0-finetuned-ade-512-512 \
  --freeze_bn \
  --gpu 0 \
  > train_IDDAW.log 2>&1 &





Checkpoints will be saved to:
  poincare-hesp/save/segmenter/hierarchical_IDDAW_d256_hyperbolic_c1.0_os16_segformer_b0_bs64_lr0.001_fbnTrue_fbbFalse/


🖼️ Inference and Visualization
  To visualize predictions:

python visualization.py \
  --mode segmenter \
  --dataset IDDAW \
  --geometry hyperbolic \
  --dim 256 \
  --c 1.0 \
  --batch_size 64 \
  --pretrained nvidia/segformer-b0-finetuned-ade-512-512 \
  --output_stride 16 \
  --gpu 0 \
  --checkpoint /path/to/epoch500.pth \
  --out_dir visualization_out \
  --num_samples 20


📊 Plot Training Logs

python plot_log.py \
  --log_file train_IDDAW_gpu2_bs64.log \
  --output logs/my_custom_plot.png
