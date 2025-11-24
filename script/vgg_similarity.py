"""
Extract facial-region VGG embeddings.

Pipeline
--------
1. Load 512x512 face images and their 256x256 binary masks.
2. Apply the same resize/center crop pipeline to both image and mask.
3. Encode the preprocessed image with VGG19 and pick a feature map.
4. Project the mask onto the feature map resolution (nearest-neighbor).
5. Perform masked global average pooling over the region of interest.
6. L2-normalize the pooled feature to obtain the final embedding.

Each processed sample produces a `.npy` file that stores the L2 normalized
embedding vector for the requested facial area.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model.vgg import VGG19

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LAYER_MAP = {
    "conv1_2": 0,
    "conv2_2": 1,
    "conv3_4": 2,
    "conv4_4": 3,
    "conv5_4": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract masked VGG embeddings for facial regions."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Folder that stores aligned 512x512 facial images.",
    )
    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Folder that stores binary masks for the same subjects.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory used to save `.npy` embedding vectors.",
    )
    parser.add_argument(
        "--layer",
        default="conv4_4",
        choices=list(LAYER_MAP.keys()),
        help="VGG19 feature layer used for region pooling.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=512,
        help="Target resolution for resize; set to 0 to skip resizing.",
    )
    parser.add_argument(
        "--center-crop",
        type=int,
        default=512,
        help="Center crop size applied after resizing; 0 disables cropping.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold (0-1) applied to mask values before binarization.",
    )
    parser.add_argument(
        "--skip-missing-mask",
        action="store_true",
        help="Skip an image when the paired mask cannot be located.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of image/mask pairs processed per GPU batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for the data loader.",
    )
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_mask_lookup(mask_root: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in mask_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            lookup.setdefault(path.stem, path)
    return lookup


def resolve_mask_path(
    image_path: Path,
    image_root: Path,
    mask_root: Path,
    mask_lookup: Dict[str, Path],
) -> Optional[Path]:
    try:
        rel = image_path.relative_to(image_root)
    except ValueError:
        rel = Path(image_path.name)

    candidate = mask_root / rel
    if candidate.exists():
        return candidate
    candidate = candidate.with_suffix(".png")
    if candidate.exists():
        return candidate

    return mask_lookup.get(image_path.stem)


def preprocess_pair(
    image_path: Path,
    mask_path: Path,
    resize: int,
    center_crop: int,
    mask_threshold: float,
    add_batch_dim: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if resize and resize > 0:
        image = image.resize((resize, resize), Image.BILINEAR)
        mask = mask.resize((resize, resize), Image.NEAREST)

    if center_crop and center_crop > 0:
        crop = transforms.CenterCrop(center_crop)
        image = crop(image)
        mask = crop(mask)

    img_tensor = transforms.ToTensor()(image)
    img_tensor = transforms.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
    )(img_tensor)
    if add_batch_dim:
        img_tensor = img_tensor.unsqueeze(0)

    mask_np = np.array(mask, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
    if add_batch_dim:
        mask_tensor = mask_tensor.unsqueeze(0)
    mask_tensor = (mask_tensor >= mask_threshold).float()

    return img_tensor, mask_tensor


def regional_grid_pool(
    feature_map: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Perform regional grid pooling (2x2) on the feature map within the mask's bounding box.
    
    Args:
        feature_map: (N, C, H, W) tensor.
        mask: (N, 1, H_orig, W_orig) tensor.
        
    Returns:
        (N, 4*C) L2-normalized embedding tensor.
    """
    # 1. Resize mask to match feature map spatial dims
    resized_mask = F.interpolate(mask, size=feature_map.shape[-2:], mode="nearest")
    # resized_mask: (N, 1, H, W)
    
    N, C, H, W = feature_map.shape
    output_embeddings = []
    
    for i in range(N):
        # Get mask for this sample
        m = resized_mask[i, 0] # (H, W)
        
        # Find bounding box
        nonzero_indices = torch.nonzero(m > 0.5)
        
        if nonzero_indices.size(0) == 0:
            # Empty mask case: return zero vector
            output_embeddings.append(torch.zeros(4 * C, device=feature_map.device))
            continue
            
        y_min = torch.min(nonzero_indices[:, 0]).item()
        y_max = torch.max(nonzero_indices[:, 0]).item()
        x_min = torch.min(nonzero_indices[:, 1]).item()
        x_max = torch.max(nonzero_indices[:, 1]).item()
        
        # Bounding box dimensions
        h_box = y_max - y_min + 1
        w_box = x_max - x_min + 1
        
        # Split into 2x2 grid
        y_mid = int(y_min + h_box / 2)
        x_mid = int(x_min + w_box / 2)
        
        # Define 4 regions: (y_start, y_end, x_start, x_end)
        # We use standard python slicing [start:end] where end is exclusive
        grids = [
            (y_min, y_mid, x_min, x_mid),          # Top-Left
            (y_min, y_mid, x_mid, x_max + 1),      # Top-Right
            (y_mid, y_max + 1, x_min, x_mid),      # Bottom-Left
            (y_mid, y_max + 1, x_mid, x_max + 1)   # Bottom-Right
        ]
        
        grid_embeddings = []
        for y1, y2, x1, x2 in grids:
            # Handle empty slices
            if y2 <= y1 or x2 <= x1:
                grid_embeddings.append(torch.zeros(C, device=feature_map.device))
                continue
                
            sub_feat = feature_map[i, :, y1:y2, x1:x2] # (C, h_sub, w_sub)
            sub_mask = m[y1:y2, x1:x2] # (h_sub, w_sub)
            
            # Masked Average within the grid
            # If sub_mask is all zeros (mask doesn't cover this part of bbox), result is 0
            mask_sum = sub_mask.sum()
            if mask_sum < 1e-6:
                grid_embeddings.append(torch.zeros(C, device=feature_map.device))
            else:
                # Weighted sum
                weighted_feat = sub_feat * sub_mask.unsqueeze(0)
                pooled = weighted_feat.sum(dim=(1, 2)) / mask_sum
                grid_embeddings.append(pooled)
                
        # Concatenate 4 vectors
        full_embedding = torch.cat(grid_embeddings, dim=0) # (4*C,)
        output_embeddings.append(full_embedding)
        
    # Stack into batch
    output_tensor = torch.stack(output_embeddings, dim=0) # (N, 4*C)
    
    # L2 Normalize
    normalized = F.normalize(output_tensor, p=2, dim=1)
    
    # Handle NaNs
    if torch.isnan(normalized).any():
        normalized = torch.where(
            torch.isnan(normalized), torch.zeros_like(normalized), normalized
        )
        
    return normalized


def assemble_pairs(
    images: Sequence[Path],
    image_root: Path,
    mask_root: Path,
    mask_lookup: Dict[str, Path],
    skip_missing: bool,
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for image_path in images:
        mask_path = resolve_mask_path(image_path, image_root, mask_root, mask_lookup)
        if mask_path is None:
            message = f"Mask for {image_path.name} is missing."
            if skip_missing:
                logging.warning(message)
                continue
            raise FileNotFoundError(message)
        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError("No valid image/mask pairs found.")

    return pairs


class FaceRegionDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        resize: int,
        center_crop: int,
        mask_threshold: float,
    ) -> None:
        self.pairs = list(pairs)
        self.resize = resize
        self.center_crop = center_crop
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path, mask_path = self.pairs[idx]
        image_tensor, mask_tensor = preprocess_pair(
            image_path,
            mask_path,
            self.resize,
            self.center_crop,
            self.mask_threshold,
            add_batch_dim=False,
        )
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "stem": image_path.stem,
        }


def collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    stems = [item["stem"] for item in batch]
    return {"image": images, "mask": masks, "stem": stems}


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_dir)
    mask_root = Path(args.mask_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA-enabled GPU is required for VGG feature extraction.")

    vgg_model = VGG19().to(device).eval()
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        logging.info("Using %d GPUs via DataParallel.", gpu_count)
        vgg = torch.nn.DataParallel(vgg_model)
    else:
        vgg = vgg_model

    images = list_images(image_root)
    if not images:
        raise FileNotFoundError(f"No images found under {image_root}.")

    mask_lookup = build_mask_lookup(mask_root)
    layer_index = LAYER_MAP[args.layer]

    pairs = assemble_pairs(
        images, image_root, mask_root, mask_lookup, args.skip_missing_mask
    )
    dataset = FaceRegionDataset(
        pairs, args.resize, args.center_crop, args.mask_threshold
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    logging.info(
        "Processing %d image/mask pairs using layer %s (batch size %d).",
        len(dataset),
        args.layer,
        args.batch_size,
    )

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        image_tensor = batch["image"].to(device, non_blocking=True)
        mask_tensor = batch["mask"].to(device, non_blocking=True)
        stems = batch["stem"]

        with torch.no_grad():
            feature_maps = vgg(image_tensor)
            feature_map = feature_maps[layer_index]
            embeddings = regional_grid_pool(feature_map, mask_tensor)

        embeddings = embeddings.cpu().numpy()
        for stem, embedding in zip(stems, embeddings):
            embedding_path = output_root / f"{stem}.npy"
            np.save(embedding_path, embedding)


if __name__ == "__main__":
    main()
