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

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
) -> (torch.Tensor, torch.Tensor):
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
    img_tensor = img_tensor.unsqueeze(0)

    mask_np = np.array(mask, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    mask_tensor = (mask_tensor >= mask_threshold).float()

    return img_tensor, mask_tensor


def masked_global_pool(
    feature_map: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    resized_mask = F.interpolate(mask, size=feature_map.shape[-2:], mode="nearest")
    mask_area = resized_mask.sum(dim=(2, 3)).clamp(min=1e-6)
    masked_features = feature_map * resized_mask
    pooled = masked_features.sum(dim=(2, 3)) / mask_area
    normalized = F.normalize(pooled, p=2, dim=1)
    if torch.isnan(normalized).any():
        normalized = torch.where(
            torch.isnan(normalized), torch.zeros_like(normalized), normalized
        )
    return normalized


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

    vgg = VGG19().to(device).eval()

    images = list_images(image_root)
    if not images:
        raise FileNotFoundError(f"No images found under {image_root}.")

    mask_lookup = build_mask_lookup(mask_root)
    layer_index = LAYER_MAP[args.layer]

    logging.info("Processing %d images with layer %s.", len(images), args.layer)

    for image_path in tqdm(images, desc="Extracting embeddings"):
        mask_path = resolve_mask_path(image_path, image_root, mask_root, mask_lookup)
        if mask_path is None:
            message = f"Mask for {image_path.name} is missing."
            if args.skip_missing_mask:
                logging.warning(message)
                continue
            raise FileNotFoundError(message)

        image_tensor, mask_tensor = preprocess_pair(
            image_path, mask_path, args.resize, args.center_crop, args.mask_threshold
        )

        image_tensor = image_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        with torch.no_grad():
            feature_maps = vgg(image_tensor)
            feature_map = feature_maps[layer_index]
            embedding = masked_global_pool(feature_map, mask_tensor)

        embedding_path = output_root / f"{image_path.stem}.npy"
        np.save(embedding_path, embedding.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    main()
