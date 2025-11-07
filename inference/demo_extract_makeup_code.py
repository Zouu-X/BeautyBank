import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
from types import SimpleNamespace

import numpy as np
from PIL import Image

import torch
from torch import optim
from torchvision import transforms

# Encoder backbones
from model.encoder.encoders import psp_encoders
from model.BeautyBank import BeautyBank
from model.stylegan import lpips
import model.contextual_loss.functional as FCX
from model.vgg import VGG19


def _get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    return {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}


def build_encoder_from_ckpt(ckpt_path, device):
    """Build only the encoder using weights from a pSp checkpoint.

    Returns: (encoder_module, opts, latent_avg)
    - encoder_module: torch.nn.Module on `device`
    - opts: SimpleNamespace with needed fields (subset from checkpoint)
    - latent_avg: torch.Tensor or None
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Recreate minimal opts used by encoders
    raw_opts = ckpt.get('opts', {})
    # Ensure required fields exist with safe defaults
    required_defaults = {
        'output_size': 1024,
        'encoder_type': 'BackboneEncoderUsingLastLayerIntoWPlus',
        'input_nc': 3,
        'start_from_latent_avg': True,
        'learn_in_w': False,
    }
    for k, v in required_defaults.items():
        raw_opts.setdefault(k, v)

    # Derive n_styles like pSp does: 2*log2(res) - 2
    raw_opts['n_styles'] = int(np.log2(raw_opts['output_size']) * 2 - 2)
    raw_opts['device'] = device.type
    opts = SimpleNamespace(**raw_opts)

    # Instantiate encoder module
    if opts.encoder_type == 'GradualStyleEncoder':
        encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', opts)
    elif opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
        encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', opts)
    elif opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
        encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', opts)
    else:
        raise ValueError(f"Unsupported encoder_type: {opts.encoder_type}")

    # Load encoder weights
    enc_state = _get_keys(ckpt, 'encoder')
    encoder.load_state_dict(enc_state, strict=True)
    encoder.to(device).eval()

    # Load latent average if present
    latent_avg = ckpt.get('latent_avg', None)
    if latent_avg is not None:
        latent_avg = latent_avg.to(device)

    return encoder, opts, latent_avg


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def noise_regularize(noises):
    loss = 0
    for noise in noises:
        size = noise.shape[2]
        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )
            if size <= 8:
                break
            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2
    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)


def refine_single_image_makeup_code(img_path, save_path, device, iters=300, lr_color=0.01, lr_structure=0.005):
    # Fixed pretrained model paths
    enc_ckpt_path = "/db-mnt/mnt/efs-mount/home/xiangzou/beauty_bank/encoder.pt"
    # Try to locate BeautyBank generator next to encoder by default
    gen_ckpt_path = os.path.join(os.path.dirname(enc_ckpt_path), 'makeup', 'generator.pt')

    if not os.path.isfile(enc_ckpt_path):
        raise FileNotFoundError(f"Pretrained encoder not found at: {enc_ckpt_path}")
    if not os.path.isfile(gen_ckpt_path):
        raise FileNotFoundError(f"BeautyBank generator not found at: {gen_ckpt_path}")

    # Build encoder only
    encoder, opts, latent_avg = build_encoder_from_ckpt(enc_ckpt_path, device)

    # Load image and preprocess
    img = Image.open(img_path).convert('RGB')
    tfm = build_transform()
    imgs = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        codes = encoder(imgs)
        if getattr(opts, 'start_from_latent_avg', False) and (latent_avg is not None):
            if getattr(opts, 'learn_in_w', False):
                codes = codes + latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + latent_avg.repeat(codes.shape[0], 1, 1)

    # Use encoder codes as both initial makeup code and bareface code for this single image
    makeups = codes.detach()
    barefaces = codes.detach()

    # Build BeautyBank generator
    generator = BeautyBank(1024, 512, 8, 2, res_index=6).to(device).eval()
    ckpt = torch.load(gen_ckpt_path, map_location='cpu')
    generator.load_state_dict(ckpt["g_ema"])
    noises_single = generator.make_noise()

    # Losses
    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.type == 'cuda')
    vggloss = None
    if device.type == 'cuda':
        try:
            vggloss = VGG19().to(device).eval()
        except Exception:
            # If pretrained VGG19 weights are not available (e.g., no internet/cache), skip VGG-based loss
            vggloss = None

    with torch.no_grad():
        real_feats = vggloss(imgs) if vggloss is not None else None

    # Prepare noises per batch=1
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
    for noise in noises:
        noise.requires_grad = True

    # Split and set learnable parts
    makeups_c = makeups[:, 7:].detach().clone(); makeups_c.requires_grad = True
    makeups_s = makeups[:, 0:7].detach().clone(); makeups_s.requires_grad = True

    optimizer = optim.Adam([
        {'params': makeups_c, 'lr': lr_color},
        {'params': makeups_s, 'lr': lr_structure},
        {'params': noises, 'lr': 0.1},
    ])

    # Identity masks (no external mask files)
    ones_mask = torch.ones_like(imgs[:, :1, :, :])  # shape [1,1,256,256]
    masks = ones_mask
    eye_masks = ones_mask
    mouth_masks = ones_mask

    for i in range(iters):
        latent = torch.cat((makeups_s, makeups_c), dim=1)
        latent = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

        img_gen, _ = generator([barefaces], latent, noise=noises, use_res=True, z_plus_latent=True)

        batch, channel, height, width = img_gen.shape
        if height > 256:
            factor = height // 256
            img_gen = img_gen.reshape(batch, channel, height // factor, factor, width // factor, factor)
            img_gen = img_gen.mean([3, 5])

        Lperc = percept(img_gen, imgs).sum()
        Lnoise = noise_regularize(noises)

        L1_mask = (torch.abs(img_gen - imgs) * masks).sum()
        Lperc_masked = percept(img_gen * masks, imgs * masks)
        L_eye_masked = percept(img_gen * eye_masks, imgs * eye_masks)
        L_mouth_masked = percept(img_gen * mouth_masks, imgs * mouth_masks)

        if vggloss is not None and real_feats is not None:
            fake_feats = vggloss(img_gen)
            LCX = FCX.contextual_loss(fake_feats[2], real_feats[2].detach(), band_width=0.2, loss_type='cosine')
        else:
            LCX = torch.tensor(0.0, device=device)

        loss = Lperc + LCX + 1e5 * Lnoise + 1e-4 * L1_mask + 100 * Lperc_masked + 100 * L_eye_masked

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        noise_normalize_(noises)

    # Save refined makeup code (concatenated structure+color in Z+ space)
    with torch.no_grad():
        latent = torch.cat((makeups_s, makeups_c), dim=1)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        np.save(save_path, latent.detach().cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description='Demo: Refine single-image makeup latent code')
    parser.add_argument('--input', required=True, help='Path to input 1024x1024 image')
    parser.add_argument('--output', required=True, help='Path to save refined_img_makeup_code.npy')
    parser.add_argument('--iter', type=int, default=300, help='Optimization iterations')
    parser.add_argument('--lr_color', type=float, default=0.01, help='LR for color parts')
    parser.add_argument('--lr_structure', type=float, default=0.005, help='LR for structure parts')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    # Prefer CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    refine_single_image_makeup_code(
        args.input,
        args.output,
        device,
        iters=args.iter,
        lr_color=args.lr_color,
        lr_structure=args.lr_structure,
    )
    print(f"Saved refined makeup latent to: {args.output}")


if __name__ == '__main__':
    main()
