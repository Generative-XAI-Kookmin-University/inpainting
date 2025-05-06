#!/usr/bin/env python3

"""
Extended version of test.py with LPIPS, FID, PSNR, and SSIM metrics.
Based on RePaint: Inpainting using Denoising Diffusion Probabilistic Models

This version uses the official LPIPS implementation and clean-fid for FID calculation.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import lpips
from cleanfid import fid
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
) 

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def main(conf: conf_mgt.Default_Conf):
    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))
    
    print("Setting up metrics...")
    lpips_model = lpips.LPIPS(net='alex').to(device)

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    dset = 'eval'
    eval_name = conf.get_default_eval_name()
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    gt_dir = os.path.join('./log', conf.name, 'temp_gt')
    sr_dir = os.path.join('./log', conf.name, 'temp_sr')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(sr_dir, exist_ok=True)
    
    all_lpips_scores = []
    all_psnr_scores = []
    all_ssim_scores = []

    for batch_idx, batch in enumerate(tqdm(iter(dl), desc="Processing batches")):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}
        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        
        srs = result['sample'] 
        gts = model_kwargs.get('gt')
        
        for i in range(batch_size):
            sr_img = srs[i:i+1]
            gt_img = gts[i:i+1]

            sr_norm = (sr_img + 1) / 2.0
            gt_norm = (gt_img + 1) / 2.0

            with th.no_grad():
                lpips_score = lpips_model(sr_norm, gt_norm).item()
            all_lpips_scores.append(lpips_score)

            sr_np = sr_norm.squeeze().permute(1, 2, 0).cpu().numpy()
            gt_np = gt_norm.squeeze().permute(1, 2, 0).cpu().numpy()

            psnr_score = psnr(gt_np, sr_np, data_range=1.0)
            all_psnr_scores.append(psnr_score)

            ssim_score = ssim(gt_np, sr_np, data_range=1.0, channel_axis=2)
            all_ssim_scores.append(ssim_score)

        srs_uint8 = toU8(srs)
        gts_uint8 = toU8(gts)
        lrs_uint8 = toU8(gts * gt_keep_mask + (-1) * th.ones_like(gts) * (1 - gt_keep_mask))
        gt_keep_masks_uint8 = toU8((gt_keep_mask * 2 - 1))

        conf.eval_imswrite(
            srs=srs_uint8, gts=gts_uint8, lrs=lrs_uint8, gt_keep_masks=gt_keep_masks_uint8,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

        for i in range(batch_size):
            gt_img = (gts_uint8[i] * 1.0).astype(np.uint8)
            sr_img = (srs_uint8[i] * 1.0).astype(np.uint8)

            img_name = f"{batch_idx:04d}_{i:02d}.png"
            
            Image.fromarray(gt_img).save(os.path.join(gt_dir, img_name))
            Image.fromarray(sr_img).save(os.path.join(sr_dir, img_name))

    mean_lpips = sum(all_lpips_scores) / len(all_lpips_scores)
    mean_psnr = sum(all_psnr_scores) / len(all_psnr_scores)
    mean_ssim = sum(all_ssim_scores) / len(all_ssim_scores)

    print("Calculating FID score...")
    fid_score = fid.compute_fid(gt_dir, sr_dir, device=device)

    print(f"Metrics on {eval_name}:")
    print(f"LPIPS: {mean_lpips:.4f}")
    print(f"PSNR: {mean_psnr:.4f} dB")
    print(f"SSIM: {mean_ssim:.4f}")
    print(f"FID: {fid_score:.4f}")

    metrics_dir = os.path.join('./log', conf.name, 'thick')
    os.makedirs(metrics_dir, exist_ok=True)
    
    with open(os.path.join(metrics_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Dataset: {eval_name}\n")
        f.write(f"LPIPS: {mean_lpips:.4f}\n")
        f.write(f"PSNR: {mean_psnr:.4f} dB\n")
        f.write(f"SSIM: {mean_ssim:.4f}\n")
        f.write(f"FID: {fid_score:.4f}\n")

    import shutil
    shutil.rmtree(gt_dir)
    shutil.rmtree(sr_dir)
    
    print("Sampling and evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)