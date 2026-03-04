import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import core.datasets as datasets
from utils import frame_utils
from raft import RAFT
from utils.utils import InputPadder, forward_interpolate
from demo import *  # if you use viz(), etc.


# -------------------------
# Corruptions (Gaussian noise / color jitter / blur)
# Images are float tensors in [0,255], shape [B,3,H,W]
# -------------------------
def _clamp255(x):
    return torch.clamp(x, 0.0, 255.0)

def _gaussian_kernel2d(ksize: int, sigma: float, device, dtype):
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    xs = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kernel1d = torch.exp(-(xs**2) / (2 * (sigma**2)))
    kernel1d = kernel1d / kernel1d.sum()
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    return kernel2d  # [K,K]

def gaussian_blur(img, ksize=11, sigma=2.0):
    # img: [B,3,H,W]
    if ksize <= 1 or sigma <= 0:
        return img
    device, dtype = img.device, img.dtype
    k2d = _gaussian_kernel2d(ksize, sigma, device, dtype)
    k = k2d[None, None, :, :]                         # [1,1,K,K]
    k = k.repeat(3, 1, 1, 1)                           # [3,1,K,K]
    pad = ksize // 2
    return F.conv2d(img, k, padding=pad, groups=3)

def color_jitter(img, brightness=0.0, contrast=0.0, saturation=0.0, seed=None):
    """
    Simple jitter in torch:
      brightness: factor in [1-b, 1+b]
      contrast:   factor in [1-c, 1+c]
      saturation: factor in [1-s, 1+s]
    Applies the SAME sampled factors to the whole batch (and you should call once and
    apply to both frames for temporal consistency).
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    out = img

    if brightness > 0:
        b = (1.0 + (2.0 * torch.rand(1, device=img.device) - 1.0) * brightness).item()
        out = out * b
        out = _clamp255(out)

    if contrast > 0:
        c = (1.0 + (2.0 * torch.rand(1, device=img.device) - 1.0) * contrast).item()
        mean = out.mean(dim=(2, 3), keepdim=True)
        out = (out - mean) * c + mean
        out = _clamp255(out)

    if saturation > 0:
        s = (1.0 + (2.0 * torch.rand(1, device=img.device) - 1.0) * saturation).item()
        # grayscale conversion
        gray = (0.2989 * out[:, 0:1] + 0.5870 * out[:, 1:2] + 0.1140 * out[:, 2:3])
        out = gray + s * (out - gray)
        out = _clamp255(out)

    return out

def apply_corruptions(image1, image2, args, sample_idx: int = 0):
    """
    image1,image2: [B,3,H,W] float in [0,255] on CUDA.
    args.corruptions: comma-separated list from {none, gaussian, colorjitter, blur}
    For colorjitter/blur we apply the same sampled transform to both frames.
    For gaussian noise we add independent noise to each frame (common robustness test).
    """
    names = []
    if args.corruptions is not None:
        names = [x.strip().lower() for x in args.corruptions.split(",") if x.strip()]
    if len(names) == 0 or names == ["none"]:
        return image1, image2

    # deterministic per-sample if desired
    seed = None
    if args.corruption_seed >= 0:
        seed = args.corruption_seed + int(sample_idx)

    out1, out2 = image1, image2

    # 1) color jitter
    if "colorjitter" in names:
        # same jitter params for both frames
        out1 = color_jitter(out1, args.jitter_brightness, args.jitter_contrast, args.jitter_saturation, seed=seed)
        out2 = color_jitter(out2, args.jitter_brightness, args.jitter_contrast, args.jitter_saturation, seed=seed)

    # 2) blur
    if "blur" in names:
        # same blur for both frames
        out1 = gaussian_blur(out1, ksize=args.blur_ksize, sigma=args.blur_sigma)
        out2 = gaussian_blur(out2, ksize=args.blur_ksize, sigma=args.blur_sigma)
        out1, out2 = _clamp255(out1), _clamp255(out2)

    # 3) gaussian noise
    if "gaussian" in names:
        if seed is not None:
            torch.manual_seed(int(seed))
        if args.noise_std > 0:
            out1 = _clamp255(out1 + torch.randn_like(out1) * args.noise_std)
            out2 = _clamp255(out2 + torch.randn_like(out2) * args.noise_std)

    return out1, out2


# -------------------------
# Existing submission helpers (unchanged)
# -------------------------
@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        flow_prev, sequence_prev = None, None

        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))
            os.makedirs(output_dir, exist_ok=True)
            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)
    os.makedirs(output_path, exist_ok=True)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        frame_utils.writeFlowKITTI(os.path.join(output_path, frame_id), flow)


@torch.no_grad()
def create_nuscenes_submission(model, iters=24, partition='front', output_path='nuscenes_submission'):
    model.eval()
    test_dataset = datasets.nuScenes(aug_params=None, partition=partition)
    output_path = os.path.join(output_path, partition)
    os.makedirs(output_path, exist_ok=True)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        frame_utils.writeFlow(os.path.join(output_path, frame_id), flow)


# -------------------------
# Your existing validators (chairs/sintel/kitti) can stay as-is
# (I’m not rewriting them to keep this focused on JHMDB + HD1K corruptions.)
# -------------------------

import time

@torch.inference_mode()
def validate_jhmdb(model, iters=24):
    """JHMDB evaluation with optional corruptions (gaussian/colorjitter/blur)."""
    model.eval()

    val_dataset = datasets.JHMDB(
        aug_params=None,
        root=args.jhmdb_root,
        use_flow=True,
        flow_mode=args.jhmdb_flow_mode,
        flow_img_scale=args.jhmdb_flow_scale,
        frame_ext=args.jhmdb_frame_ext,
        flow_ext=args.jhmdb_flow_ext,
    )

    max_n = len(val_dataset) if args.max_samples < 0 else min(len(val_dataset), args.max_samples)
    tag = args.corruptions if args.corruptions else "none"
    print(f"JHMDB pairs = {len(val_dataset)} | eval = {max_n} | corruptions = {tag}", flush=True)

    total_epe = 0.0
    total_px = 0
    cnt1 = cnt3 = cnt5 = 0

    t0 = time.time()
    for val_id in range(max_n):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image1 = image1[None].cuda(non_blocking=True)
        image2 = image2[None].cuda(non_blocking=True)
        flow_gt = flow_gt.cuda(non_blocking=True)
        if valid_gt is not None:
            valid_gt = (valid_gt.cuda(non_blocking=True) >= 0.5)

        # corrupt before padding
        image1, image2 = apply_corruptions(image1, image2, args, sample_idx=val_id)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0])  # stays on GPU: [2,H,W]

        epe = torch.sqrt(((flow - flow_gt) ** 2).sum(dim=0))  # [H,W]
        if valid_gt is not None:
            epe = epe[valid_gt]
        else:
            epe = epe.view(-1)

        total_epe += epe.sum().item()
        total_px += epe.numel()
        cnt1 += (epe < 1).sum().item()
        cnt3 += (epe < 3).sum().item()
        cnt5 += (epe < 5).sum().item()

        if val_id > 0 and (val_id % args.print_freq == 0):
            elapsed = time.time() - t0
            sp = elapsed / val_id
            eta = sp * (max_n - val_id)
            cur_epe = total_epe / max(total_px, 1)
            # print(f"[{val_id}/{max_n}] EPE={cur_epe:.4f} | {sp:.3f}s/sample | ETA~{eta/60:.1f} min", flush=True)

    epe_mean = total_epe / max(total_px, 1)
    px1 = cnt1 / max(total_px, 1)
    px3 = cnt3 / max(total_px, 1)
    px5 = cnt5 / max(total_px, 1)

    print(f"[JHMDB | {tag}] EPE: {epe_mean:.6f}, 1px: {px1:.6f}, 3px: {px3:.6f}, 5px: {px5:.6f}", flush=True)
    return {"jhmdb-epe": epe_mean, "jhmdb-1px": px1, "jhmdb-3px": px3, "jhmdb-5px": px5}


@torch.inference_mode()
def validate_hd1k(model, iters=24):
    """HD1K evaluation with optional corruptions (gaussian/colorjitter/blur)."""
    model.eval()

    val_dataset = datasets.HD1K(aug_params=None, root=args.hd1k_root)

    max_n = len(val_dataset) if args.max_samples < 0 else min(len(val_dataset), args.max_samples)
    tag = args.corruptions if args.corruptions else "none"
    print(f"HD1K pairs = {len(val_dataset)} | eval = {max_n} | corruptions = {tag}", flush=True)

    total_epe = 0.0
    total_px = 0
    cnt1 = cnt3 = cnt5 = 0

    t0 = time.time()
    for val_id in range(max_n):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image1 = image1[None].cuda(non_blocking=True)
        image2 = image2[None].cuda(non_blocking=True)
        flow_gt = flow_gt.cuda(non_blocking=True)
        if valid_gt is not None:
            valid_gt = (valid_gt.cuda(non_blocking=True) >= 0.5)

        image1, image2 = apply_corruptions(image1, image2, args, sample_idx=val_id)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0])  # GPU

        epe = torch.sqrt(((flow - flow_gt) ** 2).sum(dim=0))
        if valid_gt is not None:
            epe = epe[valid_gt]
        else:
            epe = epe.view(-1)

        total_epe += epe.sum().item()
        total_px += epe.numel()
        cnt1 += (epe < 1).sum().item()
        cnt3 += (epe < 3).sum().item()
        cnt5 += (epe < 5).sum().item()

        if val_id > 0 and (val_id % args.print_freq == 0):
            elapsed = time.time() - t0
            sp = elapsed / val_id
            eta = sp * (max_n - val_id)
            cur_epe = total_epe / max(total_px, 1)
            # print(f"[{val_id}/{max_n}] EPE={cur_epe:.4f} | {sp:.3f}s/sample | ETA~{eta/60:.1f} min", flush=True)

    epe_mean = total_epe / max(total_px, 1)
    px1 = cnt1 / max(total_px, 1)
    px3 = cnt3 / max(total_px, 1)
    px5 = cnt5 / max(total_px, 1)

    print(f"[HD1K | {tag}] EPE: {epe_mean:.6f}, 1px: {px1:.6f}, 3px: {px3:.6f}, 5px: {px5:.6f}", flush=True)
    return {"hd1k-epe": epe_mean, "hd1k-1px": px1, "hd1k-3px": px3, "hd1k-5px": px5}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--raft', help="checkpoint from the RAFT paper?", type=bool, default=True)
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Attack-related args (kept)
    parser.add_argument('--attack_type', type=str, default='None')
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('--channel', type=int, default=-1)

    # Model variants (kept)
    parser.add_argument('--fcbam', type=bool, default=False)
    parser.add_argument('--ccbam', type=bool, default=False)
    parser.add_argument('--deform', type=bool, default=False)

    # Output (kept)
    parser.add_argument('--output_path')
    parser.add_argument('--name', default="flow.png")
    parser.add_argument('--partition', type=str, default="front")

    # JHMDB args (kept)
    parser.add_argument('--jhmdb_root', type=str, default='/data/JHMDB')
    parser.add_argument('--jhmdb_flow_mode', type=str, default='auto')
    parser.add_argument('--jhmdb_flow_scale', type=float, default=20.0)
    parser.add_argument('--jhmdb_frame_ext', type=str, default='png')
    parser.add_argument('--jhmdb_flow_ext', type=str, default='jpg')

    # HD1K root
    parser.add_argument('--hd1k_root', type=str, default='../HD1k')
    parser.add_argument('--image_size', type=int, nargs='+', default=[640, 270])

    # Speed control
    parser.add_argument('--max_samples', type=int, default=-1, help='cap #pairs for fast eval')

    # Corruption controls
    parser.add_argument('--corruptions', type=str, default='none',
                        help="comma-separated: none, gaussian, colorjitter, blur (e.g., 'gaussian' or 'colorjitter,blur')")
    parser.add_argument('--corruption_seed', type=int, default=0,
                        help='>=0 makes corruptions deterministic per-sample; set -1 for fully random')

    # Gaussian noise (pixel std in [0,255] space)
    parser.add_argument('--noise_std', type=float, default=0.0)

    # Color jitter strengths (0 disables each)
    parser.add_argument('--jitter_brightness', type=float, default=0.0)
    parser.add_argument('--jitter_contrast', type=float, default=0.0)
    parser.add_argument('--jitter_saturation', type=float, default=0.0)

    # Blur
    parser.add_argument('--blur_ksize', type=int, default=11)
    parser.add_argument('--blur_sigma', type=float, default=2.0)

    parser.add_argument('--print_freq', type=int, default=500)

    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    if not args.raft:
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # Use inference_mode for speed when not attacking
    if args.attack_type == 'None':
        ctx = torch.inference_mode()
    else:
        ctx = torch.no_grad()  # keep old behavior; your attack code enables grads internally

    with ctx:
        if args.dataset == 'nuscenes':
            create_nuscenes_submission(model.module, partition=args.partition)

        elif args.dataset == 'jhmdb':
            validate_jhmdb(model.module, iters=24)

        elif args.dataset in ('hd1k', 'h1dk', 'H1DK'):
            validate_hd1k(model.module, iters=24)
        elif args.dataset == 'chairs':
            validate_chairs(model.module)
        elif args.dataset == 'sintel':
            validate_sintel(model.module, train=False)
        elif args.dataset == 'kitti':
            validate_kitti(model.module)

