import math
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.stylegan.model import Generator
from model.encoder.psp import pSp
from model.stylegan import lpips
from model.encoder.criteria import id_loss
from model.BeautyBank import BeautyBank
import model.contextual_loss.functional as FCX
from model.vgg import VGG19


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Extract refined makeup code for a single image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the aligned input image.")
    parser.add_argument(
        "--mask_root",
        type=str,
        default="./data/makeup/masks/vis",
        help="Root directory that contains face/combined_eyes/mouth subfolders.",
    )
    parser.add_argument("--face_mask_path", type=str, default=None, help="Explicit path to the face mask.")
    parser.add_argument("--eye_mask_path", type=str, default=None, help="Explicit path to the combined eyes mask.")
    parser.add_argument("--mouth_mask_path", type=str, default=None, help="Explicit path to the mouth mask.")
    parser.add_argument("--style", type=str, default="makeup", help="Style category to use when loading checkpoints.")
    parser.add_argument("--model_path", type=str, default="./checkpoint/", help="Folder that hosts pretrained weights.")
    parser.add_argument(
        "--finetuned_generator",
        type=str,
        default="finetune-000600.pt",
        help="Fine-tuned StyleGAN checkpoint used during demakeup.",
    )
    parser.add_argument(
        "--stylegan_ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="Original StyleGAN checkpoint for reconstruction.",
    )
    parser.add_argument("--encoder_ckpt", type=str, default=None, help="Path to the pSp encoder checkpoint.")
    parser.add_argument("--beautybank_ckpt", type=str, default=None, help="BeautyBank checkpoint path.")
    parser.add_argument("--truncation", type=float, default=0.5, help="Truncation used for StyleGAN sampling.")
    parser.add_argument("--demakeup_iter", type=int, default=300, help="Optimization steps for demakeup stage.")
    parser.add_argument("--refine_iter", type=int, default=300, help="Optimization steps for refinement stage.")
    parser.add_argument("--lr_color", type=float, default=0.01, help="Learning rate for color latent refinement.")
    parser.add_argument("--lr_structure", type=float, default=0.005, help="Learning rate for structure latent refinement.")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to store the refined makeup code (npy).")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU execution.")
    args = parser.parse_args()

    if args.encoder_ckpt is None:
        args.encoder_ckpt = os.path.join(args.model_path, "encoder.pt")
    if args.beautybank_ckpt is None:
        args.beautybank_ckpt = os.path.join(args.model_path, args.style, "generator.pt")
    args.finetuned_generator = os.path.join(args.model_path, args.style, args.finetuned_generator)
    args.stylegan_ckpt = os.path.join(args.model_path, args.stylegan_ckpt)
    args.id_ckpt = os.path.join(args.model_path, "model_ir_se50.pth")

    return args


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def noise_regularize(noises):
    loss = 0
    for noise in noises:
        size = noise.shape[2]
        current = noise
        while True:
            loss = (
                loss
                + (current * torch.roll(current, shifts=1, dims=3)).mean().pow(2)
                + (current * torch.roll(current, shifts=1, dims=2)).mean().pow(2)
            )
            if size <= 8:
                break
            current = current.reshape([-1, 1, size // 2, 2, size // 2, 2])
            current = current.mean([3, 5])
            size //= 2
    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std + 1e-8)


def resolve_mask_path(path: str, root: str, subfolder: str, image_name: str) -> str:
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask path not found: {path}")
        return path
    if root:
        candidate = os.path.join(root, subfolder, image_name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate {subfolder} mask for {image_name}.")


def load_image_and_masks(
    image_path: str,
    face_mask_path: str,
    eye_mask_path: str,
    mouth_mask_path: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0).float()),
        ]
    )

    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    face_mask = mask_transform(Image.open(face_mask_path).convert("RGB")).unsqueeze(0).to(device)
    eye_mask = mask_transform(Image.open(eye_mask_path).convert("RGB")).unsqueeze(0).to(device)
    mouth_mask = mask_transform(Image.open(mouth_mask_path).convert("RGB")).unsqueeze(0).to(device)
    return image, face_mask, eye_mask, mouth_mask


def build_demakeup_components(args, device):
    generator_prime = Generator(1024, 512, 8, 2).to(device).eval()
    generator = Generator(1024, 512, 8, 2).to(device).eval()
    generator_prime.load_state_dict(torch.load(args.finetuned_generator, map_location=device)["g_ema"])
    generator.load_state_dict(torch.load(args.stylegan_ckpt, map_location=device)["g_ema"])

    ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
    opts = ckpt["opts"]
    opts["checkpoint_path"] = args.encoder_ckpt
    from argparse import Namespace

    encoder = pSp(Namespace(**opts)).to(device).eval()
    encoder.load_state_dict(ckpt["state_dict"], strict=False)

    perceptual = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.type == "cuda")
    identity = id_loss.IDLoss(args.id_ckpt).to(device).eval()
    noises_single = generator.make_noise()
    return {
        "generator_prime": generator_prime,
        "generator": generator,
        "encoder": encoder,
        "perceptual": perceptual,
        "identity": identity,
        "noises": noises_single,
    }


def run_demakeup(
    tensors: Dict[str, torch.Tensor],
    args,
    device: torch.device,
    components: Dict[str, torch.nn.Module],
) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs = tensors["image"]
    masks = tensors["face"]
    generator_prime = components["generator_prime"]
    generator = components["generator"]
    encoder = components["encoder"]
    perceptual = components["perceptual"]
    identity = components["identity"]
    noises_single = components["noises"]

    with torch.no_grad():
        _, latent_e = encoder(imgs, randomize_noise=False, return_latents=True, z_plus_latent=True)

    noises = []
    for noise in noises_single:
        noise_batch = noise.repeat(imgs.shape[0], 1, 1, 1).normal_().to(device)
        noise_batch.requires_grad = True
        noises.append(noise_batch)

    latent = latent_e.detach().clone()
    latent.requires_grad = True

    optimizer = optim.Adam([latent] + noises, lr=0.1)
    pbar = tqdm(range(args.demakeup_iter), desc="Demakeup", ncols=100)
    for i in pbar:
        t = i / max(1, args.demakeup_iter)
        lr = get_lr(t, 0.1)
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = generator_prime([latent], input_is_latent=False, noise=noises, z_plus_latent=True)
        batch, channel, height, width = img_gen.shape
        if height > 256:
            factor = height // 256
            img_gen = img_gen.reshape(batch, channel, height // factor, factor, width // factor, factor)
            img_gen = img_gen.mean([3, 5])

        Lperc = perceptual(img_gen, imgs).sum()
        LID = identity(img_gen, imgs)
        Lreg = latent.std(dim=1).mean()
        Lnoise = noise_regularize(noises)
        mask_sum = masks.sum().item()
        mask_den = mask_sum if mask_sum > 0 else imgs.shape[0]
        L1_mask = (torch.abs(img_gen - imgs) * masks).sum() / mask_den

        loss = 1 * Lperc + 0.1 * LID + Lreg + 1e5 * Lnoise + 100 * L1_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        noise_normalize_(noises)
        pbar.set_postfix({"Lperc": f"{Lperc.item():.3f}", "L1": f"{L1_mask.item():.3f}"})

    with torch.no_grad():
        latent[:, 8:18] = latent_e[:, 8:18].detach()
        img_dsty, _ = generator(
            [latent.detach()],
            input_is_latent=False,
            truncation=args.truncation,
            truncation_latent=0,
            noise=noises,
            z_plus_latent=True,
        )
        img_dsty = F.adaptive_avg_pool2d(img_dsty.detach(), 256)
        _, latent_i = encoder(img_dsty, randomize_noise=False, return_latents=True, z_plus_latent=True)
        latent_i[:, 8:18] = latent_e[:, 8:18].detach()

    return latent_e.detach(), latent_i.detach()


def build_refine_components(args, device):
    generator = BeautyBank(1024, 512, 8, 2, res_index=6).to(device).eval()
    generator.load_state_dict(torch.load(args.beautybank_ckpt, map_location=device)["g_ema"])
    perceptual = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.type == "cuda")
    vggloss = VGG19().to(device).eval()
    noises_single = generator.make_noise()
    return {
        "generator": generator,
        "perceptual": perceptual,
        "vgg": vggloss,
        "noises": noises_single,
    }


def run_refinement(
    tensors: Dict[str, torch.Tensor],
    codes: Dict[str, torch.Tensor],
    args,
    device: torch.device,
    components: Dict[str, torch.nn.Module],
) -> torch.Tensor:
    generator = components["generator"]
    perceptual = components["perceptual"]
    vggloss = components["vgg"]
    noises_single = components["noises"]

    imgs = tensors["image"]
    masks = tensors["face"]
    eye_masks = tensors["eye"]
    mouth_masks = tensors["mouth"]

    makeups = codes["makeup"].to(device)
    barefaces = codes["bareface"].to(device)

    makeups_c = makeups[:, 7:].detach().clone().requires_grad_(True)
    makeups_s = makeups[:, 0:7].detach().clone().requires_grad_(True)

    noises = []
    for noise in noises_single:
        noise_batch = noise.repeat(imgs.shape[0], 1, 1, 1).normal_().to(device)
        noise_batch.requires_grad = True
        noises.append(noise_batch)

    optimizer = optim.Adam(
        [{"params": makeups_c, "lr": args.lr_color}, {"params": makeups_s, "lr": args.lr_structure}, {"params": noises, "lr": 0.1}]
    )

    with torch.no_grad():
        real_feats = vggloss(imgs)

    pbar = tqdm(range(args.refine_iter), desc="Refine", ncols=100)
    for _ in pbar:
        latent = torch.cat((makeups_s, makeups_c), dim=1)
        latent = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(latent.shape)
        img_gen, _ = generator([barefaces], latent, noise=noises, use_res=True, z_plus_latent=True)
        batch, channel, height, width = img_gen.shape
        if height > 256:
            factor = height // 256
            img_gen = img_gen.reshape(batch, channel, height // factor, factor, width // factor, factor)
            img_gen = img_gen.mean([3, 5])

        Lperc = perceptual(img_gen, imgs).sum()
        Lnoise = noise_regularize(noises)
        L1_mask = (torch.abs(img_gen - imgs) * masks).sum()
        Lperc_masked = perceptual(img_gen * masks, imgs * masks)
        L_eye_masked = perceptual(img_gen * eye_masks, imgs * eye_masks)
        L_mouth_masked = perceptual(img_gen * mouth_masks, imgs * mouth_masks)

        fake_feats = vggloss(img_gen)
        LCX = FCX.contextual_loss(fake_feats[2], real_feats[2].detach(), band_width=0.2, loss_type="cosine")

        loss = Lperc + LCX + 1e5 * Lnoise + 1e-4 * L1_mask + 100 * Lperc_masked + 100 * L_eye_masked + 100 * L_mouth_masked

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        noise_normalize_(noises)
        pbar.set_postfix({"Lp": f"{Lperc.item():.3f}", "LCX": f"{LCX.item():.3f}"})

    with torch.no_grad():
        latent = torch.cat((makeups_s, makeups_c), dim=1)
    return latent.detach()


def extract_makeup_code(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    image_name = os.path.basename(args.image_path)
    face_mask_path = resolve_mask_path(args.face_mask_path, args.mask_root, "face", image_name)
    eye_mask_path = resolve_mask_path(args.eye_mask_path, args.mask_root, "combined_eyes", image_name)
    mouth_mask_path = resolve_mask_path(args.mouth_mask_path, args.mask_root, "mouth", image_name)

    image, face_mask, eye_mask, mouth_mask = load_image_and_masks(
        args.image_path, face_mask_path, eye_mask_path, mouth_mask_path, device
    )
    tensors = {"image": image, "face": face_mask, "eye": eye_mask, "mouth": mouth_mask}

    demakeup_components = build_demakeup_components(args, device)
    makeup_code, bareface_code = run_demakeup(tensors, args, device, demakeup_components)

    refine_components = build_refine_components(args, device)
    refined_code = run_refinement(
        tensors,
        {"makeup": makeup_code, "bareface": bareface_code},
        args,
        device,
        refine_components,
    )

    result = refined_code.cpu().numpy()
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.save(args.save_path, result)
    return result


if __name__ == "__main__":
    arguments = parse_args()
    code = extract_makeup_code(arguments)
    print("Refined makeup code shape:", code.shape)
    if arguments.save_path:
        print(f"Saved refined code to {arguments.save_path}")
    else:
        np.set_printoptions(suppress=True, precision=4)
        print(code)
