"""
Extract facial-region VGG-Face embeddings.

Pipeline
--------
1. Load 512x512 face images and their 256x256 binary masks.
2. Apply the same resize/center crop pipeline to both image and mask.
3. Encode the preprocessed image with VGG-Face and pick a feature map.
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

from model.vgg_face import VGGFace

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LAYER_MAP = {
    "conv1_1": "conv1_1",
    "conv3_3": "conv3_3",
    "conv4_3": "conv4_3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract masked VGG-Face embeddings for facial regions."
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
        default="conv4_3",
        choices=list(LAYER_MAP.keys()),
        help="VGG-Face feature layer used for region pooling.",
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Path to VGG-Face weights file (.pth).",
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

    # VGG-Face preprocessing usually involves subtracting mean [129.1863, 104.7624, 93.5940]
    # However, standard VGG preprocessing is often mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # The original script used mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5).
    # We should stick to what the user might expect or standard VGG-Face if known.
    # VGG-Face (Parkhi et al.) uses BGR mean subtraction [93.5940, 104.7624, 129.1863].
    # But since we don't have the weights and the user might provide weights trained differently,
    # I will stick to the normalization used in vgg_similarity.py for consistency, 
    # OR I should use the standard VGG-Face preprocessing if I assume standard weights.
    # Given the ambiguity, I'll stick to the previous script's normalization for now,
    # but add a comment.
    
    img_tensor = transforms.ToTensor()(image)  # [0, 1] RGB
    img_tensor = img_tensor * 255.0  # [0, 255] RGB
    img_tensor = img_tensor[[2, 1, 0], :, :]  # [0, 255] BGR
    
    # VGG-Face mean subtraction [93.5940, 104.7624, 129.1863]
    mean = torch.tensor([93.5940, 104.7624, 129.1863]).view(3, 1, 1)
    img_tensor = img_tensor - mean
    if add_batch_dim:
        img_tensor = img_tensor.unsqueeze(0)

    mask_np = np.array(mask, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
    if add_batch_dim:
        mask_tensor = mask_tensor.unsqueeze(0)
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
        logging.warning("CUDA not available. Using CPU, which might be slow.")

    vgg_model = VGGFace(weights_path=args.weights_path).to(device).eval()
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
    
    # Check if layer is valid
    if args.layer not in LAYER_MAP:
        raise ValueError(f"Invalid layer: {args.layer}. Choose from {list(LAYER_MAP.keys())}")

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
            # feature_maps is a dict
            feature_map = feature_maps[args.layer]
            embeddings = masked_global_pool(feature_map, mask_tensor)

        embeddings = embeddings.cpu().numpy()
        for stem, embedding in zip(stems, embeddings):
            embedding_path = output_root / f"{stem}.npy"
            np.save(embedding_path, embedding)


if __name__ == "__main__":
    main()
