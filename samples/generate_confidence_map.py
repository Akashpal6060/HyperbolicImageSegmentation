import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from hesp.config.config import Config
from hesp.models.model import ModelFactory

# ----------------------
# 1. Configuration
# ----------------------
dataset = 'IDDAW'  # or 'coco', 'pascal', etc.
mode = 'segmenter'
# base_save_dir can be empty or your project root
gpu_idx = 0
config = Config(dataset=dataset, base_save_dir="", gpu_idx=gpu_idx, mode=mode)

# Hyperbolic embedding settings
config.embedding_space._GEOMETRY       = 'hyperbolic'
config.embedding_space._DIM            = 256
config.embedding_space._INIT_CURVATURE = 2.0
config.embedding_space._HIERARCHICAL   = True

# Segmenter settings (should match training)
config.segmenter._PRETRAINED_MODEL    = (
    "/users/student/pg/pg23/akash.pal/HyperbolicImageSegmentation/"
    "poincare-hesp/save/"
    "hierarchical_IDDAW_d256_hyperbolic_c1.0_os16_segformer_b0_bs2_lr0.001_fbnTrue_fbbFalse/"
    "epoch005.pth"
)
config.segmenter._OUTPUT_STRIDE       = 16
config.segmenter._BACKBONE            = 'segformer-b5'
config.segmenter._BATCH_SIZE          = 5
config.segmenter._FREEZE_BACKBONE     = False
config.segmenter._FREEZE_BN           = True
# NOTE: These must match training config
config.segmenter._NUM_EPOCHS          = config.dataset._NUM_EPOCHS
config.segmenter._NUM_TRAIN           = config.dataset._NUM_TRAIN
config.segmenter._INITIAL_LEARNING_RATE = config.dataset._INITIAL_LEARNING_RATE
config.segmenter._EFN_OUT_DIM         = config.embedding_space._DIM

# ----------------------
# 2. Model instantiation
# ----------------------
device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
model = ModelFactory.create(model_name=mode, cfg=config)
model.to(device)
model.eval()

# ----------------------
# 3. Load checkpoint
# ----------------------
# Assuming your checkpoints are saved as .pth in config._SEGMENTER_SAVE_DIR
ckpt_dir = config._SEGMENTER_SAVE_DIR
all_ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
if not all_ckpts:
    raise FileNotFoundError(f"No .pth checkpoints found in {ckpt_dir}")
# pick latest by lexicographic sort (or adjust to timestamp)
latest_ckpt = sorted(all_ckpts)[-1]
state = torch.load(os.path.join(ckpt_dir, latest_ckpt), map_location=device)
model.load_state_dict(state)

# If you saved normals, offsets, curvature separately as .npy:
# np.save(os.path.join(ckpt_dir, 'normals.npy'), model.embedding_space.normals_npy)
# Then load:
normals_npy = np.load(os.path.join(ckpt_dir, 'normals.npy'))
offsets_npy = np.load(os.path.join(ckpt_dir, 'offsets.npy'))
c_value    = np.load(os.path.join(ckpt_dir, 'curvature.npy'))
model.embedding_space.normals_npy = normals_npy
model.embedding_space.offsets_npy = offsets_npy
model.embedding_space.c_npy       = np.array(c_value, dtype=np.float32)

# ----------------------
# 4. Inference loop
# ----------------------
origin_path = 'path/to/images'
image_list_file = '../datasets/pascal/pascal_data/all.txt'

with open(image_list_file, 'r') as f:
    image_ids = [line.strip() for line in f if line.strip()]

for img_id in image_ids:
    img_path = os.path.join(origin_path, img_id + '.jpg')
    img_bgr  = cv2.imread(img_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Prepare tensor: HxWxC -> 1xCxHxW, float32
    img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # Dummy mask if required by signature:
    dummy_mask = torch.zeros_like(img_tensor)

    with torch.no_grad():
        results = model.predict(img_tensor, dummy_mask)

    # results['embeddings'] is 1xHxWxC
    embeddings = results['embeddings'][0].cpu().numpy()
    # Compute per-pixel confidence
    confidence_map = np.linalg.norm(embeddings, axis=-1)
    radius = 1.0 / math.sqrt(config.embedding_space._INIT_CURVATURE)
    normalized_confidence_map = confidence_map / radius

    # Display confidence map
    plt.figure()
    plt.imshow(confidence_map, cmap='gist_earth')
    plt.title(f"Confidence: {img_id}")
    plt.axis('off')
    plt.show()
