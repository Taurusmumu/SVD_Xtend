#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
os.environ['CUDA_VISIBLE_DEVICES']='2'
import sys
import pandas as pd
import numpy as np
from functools import partial
import diffusers
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from scripts.calculate_metrics import FID, prepare_logger
from accelerate import PartialState
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from common_metrics_on_video_quality.calculate_psnr import calculate_psnr, calculate_psnr_vertical
from common_metrics_on_video_quality.calculate_ssim import calculate_ssim, calculate_ssim_vertical
from common_metrics_on_video_quality.calculate_lpips import calculate_lpips, calculate_lpips_vertical

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
distributed_state = PartialState()
device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')


def sample_gt_images(gt_dir, sample_file_path):
    from scripts.dataloaders import GTSampleDataset
    dataset = GTSampleDataset(data_dir=gt_dir, sample_file_path=sample_file_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    for batch in tqdm(dataloader):
        source_paths = batch["source_paths"]
        save_paths = batch["save_paths"]
        for img, path in zip(source_paths, save_paths):
            for i in range(len(path)):
                s_path = img[i]
                o_path = path[i]
                image = Image.open(s_path).convert("RGB")
                os.makedirs(os.path.dirname(o_path), exist_ok=True)
                image.save(o_path)
    print()

def load_videos_from_folder(dir_path, num_frames = 3):
    videos = []

    for slide in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, slide)):
            continue
        slide_path = os.path.join(dir_path, slide)

        for patch in os.listdir(slide_path):
            patch_path = os.path.join(slide_path, patch)
            frames = sorted(os.listdir(patch_path))
            frame_paths = [os.path.join(patch_path, frame) for frame in frames]
            frame_np = [np.array(Image.open(frame_path).convert('RGB')) for frame_path in frame_paths]
            videos.append(np.stack(frame_np))

    return np.stack(videos)

def convert_to_tensor(no_array):
    array = torch.from_numpy(no_array)
    array = array.permute(0, 1, 4, 2, 3)
    array = array.float() / 255.0  # Normalize to [0, 1]
    return array


if __name__ == "__main__":
    size = 256
    sample_file_done, sample_GT_done, sample_pred_done = True, True, True
    sample_num = 5000
    sample_file_path = f"/home/compu/jiamu/SVD_Xtend/image_process/blur_data_sampled_{sample_num}.csv"
    blur_data_path = "/home/compu/jiamu/SVD_Xtend/image_process/blur_data1.csv"
    split_file_path = "/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt"
    output_dir = "/ssd2/AMC_zstack_2_patches/output_for_metrics/"
    ckp_v = 80000
    version = "output0526"
    pretrained_model_path = f"/ssd2/AMC_zstack_2_patches/{version}/checkpoint-{ckp_v}"
    pretrained_model_name = "stabilityai/stable-video-diffusion-img2vid-xt"
    gt_dir = os.path.join(output_dir, "GT")
    fake_dir = os.path.join(output_dir, "Pred", f"{version}_{ckp_v}")
    log_path = os.path.join(fake_dir, "test_logs.txt")
    os.makedirs(fake_dir, exist_ok=True)
    logger = prepare_logger(log_path)

    if sample_file_done is False:
        split_data = pd.read_csv(split_file_path)
        split_data_train = split_data.loc[split_data["train"] == "train"]
        split_data_train_slides = list(split_data_train.iloc[:, 0])

        df = pd.read_csv(blur_data_path)
        df_train = df.loc[df["slide_name"].isin(split_data_train_slides)]
        df_sample = df_train.sample(n=sample_num, random_state=42)
        df_sample.to_csv(sample_file_path, index=False)

    if sample_GT_done is False:
        sample_gt_images(gt_dir, sample_file_path)

    if sample_pred_done is False:
        start = time.time()

        feature_extractor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name, subfolder="feature_extractor"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            pretrained_model_name, subfolder="image_encoder"
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            pretrained_model_name, subfolder="vae", variant="fp16")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            pretrained_model_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
        )

        # Freeze vae and image_encoder
        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        # # generator = torch.Generator("cuda:0").manual_seed(8)
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            pretrained_model_name,
            unet=unet,
            image_encoder=image_encoder,
            vae=vae,
            torch_dtype=torch.float32
        )
        # pipeline.set_progress_bar_config(disable=True)

        pipeline.to(distributed_state.device)

        for slide in os.listdir(gt_dir):
            slide_dir = os.path.join(gt_dir, slide)
            for patch in os.listdir(slide_dir):
                patch_dir = os.path.join(slide_dir, patch)

                pred_frames = []
                for frame_path in os.listdir(patch_dir):
                    pred_frames.append(os.path.join(patch_dir, frame_path))

                mid_frame = pred_frames[1]
                video_frames = pipeline(
                    load_image(mid_frame).resize((size, size)),
                    height=size,
                    width=size,
                    num_frames=len(pred_frames),
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.02,
                    # generator=generator,
                ).frames[0]

                out_path = os.path.join(
                    fake_dir, slide, patch,
                )
                os.makedirs(out_path, exist_ok=True)
                for i in range(len(pred_frames)):
                    img = video_frames[i]
                    # img = np.array(img)
                    # img = Image.fromarray(img)
                    img.save(os.path.join(fake_dir, slide, patch, f"{pred_frames[i].split('/')[-1]}"))

        end = time.time()
        elapsed = end - start
        print(f"Inference took {elapsed:.2f} seconds") # 7 hours

    start = time.time()
    gt_np = "/ssd2/AMC_zstack_2_patches/output_for_metrics/real_videos.npy"
    if not os.path.isfile(gt_np):
        real_videos = load_videos_from_folder(gt_dir)
        np.save(gt_np, real_videos.astype(np.float32))

    pred_np = f"/ssd2/AMC_zstack_2_patches/output_for_metrics/pred_videos_{version}_{ckp_v}.npy"
    if not os.path.isfile(pred_np):
        fake_videos = load_videos_from_folder(fake_dir)
        np.save(pred_np, fake_videos.astype(np.float32))

    real_videos = np.load(gt_np).astype(np.float32)   # [N, T, H, W, C]
    fake_videos = np.load(pred_np).astype(np.float32)

    # Convert to TensorFlow tensors
    real_tensor = convert_to_tensor(real_videos)
    fake_tensor = convert_to_tensor(fake_videos)
    import json
    result = {}
    only_final = False
    # only_final = True
    # result['fvd'] = calculate_fvd(real_tensor, fake_tensor, device, method='styleganv', only_final=only_final)
    result['ssim'] = calculate_ssim(real_tensor, fake_tensor, only_final=only_final)
    result['psnr'] = calculate_psnr(real_tensor, fake_tensor, only_final=only_final)
    result['lpips'] = calculate_lpips(real_tensor, fake_tensor, device, only_final=only_final)
    result['ssim_final'] = np.mean(result['ssim']['value'])
    result['psnr_final'] = np.mean(result['psnr']['value'])
    result['lpips_final'] = np.mean(result['lpips']['value'])

    result['ssim_vertical'] = calculate_ssim_vertical(real_tensor, fake_tensor, only_final=only_final)
    result['psnr_vertical'] = calculate_psnr_vertical(real_tensor, fake_tensor, only_final=only_final)
    result['lpips_vertical'] = calculate_lpips_vertical(real_tensor, fake_tensor, device, only_final=only_final)
    result['ssim_vertical_final'] = np.mean(result['ssim_vertical']['value'])
    result['psnr_vertical_final'] = np.mean(result['psnr_vertical']['value'])
    result['lpips_vertical_final'] = np.mean(result['lpips_vertical']['value'])

    print(json.dumps(result, indent=4))
    with open(f"/ssd2/AMC_zstack_2_patches/output_for_metrics/log_{version}_{ckp_v}.json", "w") as f:
        json.dump(result, f, indent=4)

    end_fvd = time.time()
    elapsed = end_fvd - start
    print(f"FVD took {elapsed:.2f} seconds")

    start = time.time()
    metric = FID(gt_dir, fake_dir, logger, distributed_state.device)
    fid = metric.compute()
    result['fid'] = fid
    with open(f"/ssd2/AMC_zstack_2_patches/output_for_metrics/log_{version}_{ckp_v}.json", "w") as f:
        json.dump(result, f, indent=4)

    end_fid = time.time()
    elapsed = end_fid - start
    print(f"FID took {elapsed:.2f} seconds")








