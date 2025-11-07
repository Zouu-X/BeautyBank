import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
from types import SimpleNamespace

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

# Encoder backbones
from model.encoder.encoders import psp_encoders


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


def extract_makeup_latent(img_path, save_path, device):
    # Fixed pretrained model path (as requested)
    ckpt_path = "/db-mnt/mnt/efs-mount/home/xiangzou/beauty_bank/encoder.pt"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained encoder not found at: {ckpt_path}")

    # Build encoder only
    encoder, opts, latent_avg = build_encoder_from_ckpt(ckpt_path, device)

    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    tfm = build_transform()
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        codes = encoder(x)
        # Align with pSp behavior: optionally add latent_avg
        if getattr(opts, 'start_from_latent_avg', False) and (latent_avg is not None):
            if getattr(opts, 'learn_in_w', False):
                codes = codes + latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + latent_avg.repeat(codes.shape[0], 1, 1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.save(save_path, codes.detach().cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description='Demo: Extract makeup latent code')
    parser.add_argument('--input', required=True, help='Path to input 1024x1024 image')
    parser.add_argument('--output', required=True, help='Path to save makeup_code.npy')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    # Prefer CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    extract_makeup_latent(args.input, args.output, device)
    print(f"Saved makeup latent to: {args.output}")


if __name__ == '__main__':
    main()

