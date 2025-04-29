import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class IDDAWDataset(Dataset):
    def __init__(self, root_dir, mode='train', conditions=["FOG", "LOWLIGHT", "RAIN", "SNOW"], image_size=512):
        self.image_paths = []
        self.mask_paths = []

        for condition in conditions:
            rgb_root = os.path.join(root_dir, mode, condition, "rgb")
            mask_root = os.path.join(root_dir, mode, condition, "gtSeg_png")
            if not os.path.exists(rgb_root) or not os.path.exists(mask_root):
                continue

            # ✅ Walk recursively through rgb/ subfolders
            for dirpath, _, filenames in os.walk(rgb_root):
                for fname in sorted(filenames):
                    if fname.endswith("_rgb.png"):
                        image_path = os.path.join(dirpath, fname)

                        # Get relative path like "46/00000001_rgb.png"
                        rel_path = os.path.relpath(image_path, rgb_root)

                        # Construct mask path by replacing suffix
                        mask_path = os.path.join(mask_root, rel_path.replace("_rgb.png", "_mask.png"))

                        if os.path.exists(mask_path):
                            self.image_paths.append(image_path)
                            self.mask_paths.append(mask_path)

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.PILToTensor()
        ])

        print(f"🧾 Loaded {len(self.image_paths)} image-mask pairs from {mode} set.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        image = self.transform(image)
        mask = self.mask_transform(mask).long().squeeze(0)  # Convert [1, H, W] → [H, W]
        return image, mask
