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
import csv
import json
import logging
import os
import time
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from fid_score import FID
from accelerate import PartialState
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from common_metrics_on_video_quality.calculate_psnr import calculate_psnr, calculate_psnr_vertical
from common_metrics_on_video_quality.calculate_ssim import calculate_ssim, calculate_ssim_vertical
from common_metrics_on_video_quality.calculate_lpips import calculate_lpips, calculate_lpips_vertical
from dataloaders import GTSampleDataset

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
distributed_state = PartialState()
device = distributed_state.device


def load_videos_from_folder(dir_path, match_sampled_pair_file=None):
    videos, blur_degrees = [], []

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
            cur_bd = match_sampled_pair_file.loc[(match_sampled_pair_file["slide_name"] == slide) & (match_sampled_pair_file["patch_name"] == patch)]["blur_degree"].values
            blur_degrees.append(np.stack(cur_bd))

    return np.stack(videos), np.stack(blur_degrees)


def convert_to_tensor(no_array):
    array = torch.from_numpy(no_array)
    array = array.permute(0, 1, 4, 2, 3)
    array = array.float() / 255.0  # Normalize to [0, 1]
    return array

def calculate_metrics(match_sampled_pair_path,
                      gt_np_2048,
                      gt_np_2048_bd,
                      gt_dir,
                      pred_np_2048,
                      fake_match_dir,
                      metric_log,
                      device,
                      threshold=0.2,
                      only_final = True,
                      sample_num_frame=10000):
    print("Calculating metrics for FVD...")
    result = {}
    match_sampled_pair_file_ori = pd.read_csv(match_sampled_pair_path)
    if not os.path.isfile(gt_np_2048):
        real_videos, blur_degrees = load_videos_from_folder(gt_dir, match_sampled_pair_file_ori)
        np.save(gt_np_2048, real_videos.astype(np.float32))
        np.save(gt_np_2048_bd, blur_degrees.astype(np.float32))
    else:
        real_videos = np.load(gt_np_2048).astype(np.float32)

    if not os.path.isfile(pred_np_2048):
        fake_videos, blur_degrees = load_videos_from_folder(fake_match_dir, match_sampled_pair_file_ori)
        np.save(pred_np_2048, fake_videos.astype(np.float32))
    else:
        fake_videos = np.load(pred_np_2048).astype(np.float32)

    real_tensor = convert_to_tensor(real_videos)
    fake_tensor = convert_to_tensor(fake_videos)
    blur_degrees = np.load(gt_np_2048_bd).astype(np.float32)

    result['fvd'] = calculate_fvd(real_tensor, fake_tensor, device, method='styleganv', only_final=only_final)
    with open(metric_log, "w") as f:
        json.dump(result, f, indent=4)

    start = time.time()
    result['ssim'] = calculate_ssim(real_tensor, fake_tensor, only_final=only_final, blur_degrees=blur_degrees)
    result['psnr'] = calculate_psnr(real_tensor, fake_tensor, only_final=only_final, blur_degrees=blur_degrees)
    result['lpips'] = calculate_lpips(real_tensor, fake_tensor, device, only_final=only_final, blur_degrees=blur_degrees)
    result['ssim_final'] = np.mean(result['ssim']['value'])
    result['psnr_final'] = np.mean(result['psnr']['value'])
    result['lpips_final'] = np.mean(result['lpips']['value'])
    print(json.dumps(result, indent=4))
    with open(metric_log, "w") as f:
        json.dump(result, f, indent=4)

    result['ssim_vertical'] = calculate_ssim_vertical(real_tensor, fake_tensor, only_final=only_final, blur_degrees=blur_degrees)
    result['psnr_vertical'] = calculate_psnr_vertical(real_tensor, fake_tensor, only_final=only_final, blur_degrees=blur_degrees)
    result['lpips_vertical'] = calculate_lpips_vertical(real_tensor, fake_tensor, device, only_final=only_final, blur_degrees=blur_degrees)
    result['ssim_vertical_final'] = np.mean(result['ssim_vertical']['value'])
    result['psnr_vertical_final'] = np.mean(result['psnr_vertical']['value'])
    result['lpips_vertical_final'] = np.mean(result['lpips_vertical']['value'])

    print(json.dumps(result, indent=4))
    with open(metric_log, "w") as f:
        json.dump(result, f, indent=4)

    end_smi = time.time()
    elapsed = end_smi - start
    print(f"Similarity took {elapsed:.2f} seconds")

    print("Calculating FID...")
    match_sampled_pair_file = match_sampled_pair_file_ori.loc[match_sampled_pair_file_ori["blur_degree"] <= threshold]
    match_sampled_pair_file = match_sampled_pair_file.sample(n=sample_num_frame, random_state=42)
    gt_dir_list = list(match_sampled_pair_file["gt_path"])
    pred_dir_list = list(match_sampled_pair_file["pred_path"])

    start = time.time()
    metric = FID(gt_dir_list, pred_dir_list, device)
    fid = metric.compute(num_samples=sample_num_frame)
    result['fid'] = fid
    with open(metric_log, "w") as f:
        json.dump(result, f, indent=4)
    #
    end_fid = time.time()
    elapsed = end_fid - start
    print(f"FID took {elapsed:.2f} seconds")


if __name__ == "__main__":
    layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09", "z10", "z11", "z12", "z13", "z14",
              "z15", "z16", "z17", "z18"]
    size, sample_num, sample_num_frame = 256, 2048, 10000
    split_file_path = "/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt"
    data_root_path = "/ssd2/AMC_zstack_2_patches/pngs_mid"
    pretrained_model_name = "stabilityai/stable-video-diffusion-img2vid-xt"
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name, subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name, subfolder="image_encoder"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name, subfolder="vae", variant="fp16")
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    generator = torch.Generator(device).manual_seed(8)

    blur_data_path = "/home/compu/jiamu/SVD_Xtend/image_process/blur_data5.csv"
    ckp_v = 400000
    version = "output0701"
    num_frame = 11
    output_dir = f"/ssd2/AMC_zstack_2_patches/output_for_metrics/{version}"
    os.makedirs(output_dir, exist_ok=True)
    match_sampled_2048_path = f"{output_dir}/match_sampled_{sample_num}.csv"
    match_sampled_pair_path = f"{output_dir}/Pred_match/match_sampled_pair_{ckp_v}.csv"
    pretrained_model_path = f"/ssd2/AMC_zstack_2_patches/{version}/checkpoint-{ckp_v}"

    gt_np_2048 = f"{output_dir}/real_videos.npy"
    gt_np_2048_bd = f"{output_dir}/real_videos_blur_degree.npy"
    pred_np_2048 = f"{output_dir}/Pred_match/pred_videos_{ckp_v}.npy"
    metric_log = f"{output_dir}/Pred_match/log_{ckp_v}.json"

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    unet.requires_grad_(False)
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_name,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=torch.float32,
        generator = generator
    )
    # pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)
    pipeline.enable_model_cpu_offload()

    gt_dir = os.path.join(output_dir, "GT")
    fake_match_dir = os.path.join(output_dir, "Pred_match", f"{ckp_v}")
    log_path = os.path.join(fake_match_dir, "test_logs.txt")
    os.makedirs(fake_match_dir, exist_ok=True)

    if os.path.isfile(match_sampled_2048_path) is False:
        split_data = pd.read_csv(split_file_path)
        split_data_train = split_data.loc[split_data["train"] == "train"]
        split_data_train_slides = list(split_data_train.iloc[:, 0])

        df = pd.read_csv(blur_data_path)
        df_train = df.loc[df["slide_name"].isin(split_data_train_slides)]
        start_layers = np.array(df_train["start_indices"])
        end_layers = np.array(df_train["end_indices"])
        valid_index = np.where((start_layers >= 0) & (end_layers <= len(layers) - 1))[0]
        df_valid = df_train.iloc[valid_index]
        df_sample = df_valid.sample(n=sample_num, random_state=42)
        df_sample.to_csv(match_sampled_2048_path, index=False)

    # start to sample 2048 3D Images
    start = time.time()
    dataset = GTSampleDataset(data_dir=data_root_path,sample_file_path=match_sampled_2048_path, num_frames=num_frame)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) # batch_size = 1
    fieldnames = ["slide_name", "patch_name", "frame", "gt_path", "pred_path", "blur_degree"]
    if not os.path.isfile(match_sampled_pair_path):
        with open(match_sampled_pair_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"]
        blur_degrees = batch["blur_degrees"]
        slide_name = batch["slide_name"][0]
        patch_name = batch["patch_name"][0]
        frames = batch["frames"]

        gt_path = os.path.join(gt_dir, slide_name, patch_name)
        if not (os.path.exists(gt_path) and len(os.listdir(gt_path)) == num_frame):
            # save GT frames
            os.makedirs(gt_path, exist_ok=True)
            for i in range(len(frames)):
                img = pixel_values[i][0]
                img = img.cpu().numpy()
                img = np.clip(img, 0, 255).astype(np.uint8)
                img = transforms.ToPILImage()(img)
                # pixel_values[i] = img
                img.save(os.path.join(gt_path, f"{frames[i][0]}.png"))

        pred_path = os.path.join(fake_match_dir, slide_name, patch_name)
        mid_frame = pixel_values[len(pixel_values)//2][0]
        mid_frame = mid_frame.cpu().numpy()
        mid_frame = np.clip(mid_frame, 0, 255).astype(np.uint8)
        mid_frame = transforms.ToPILImage()(mid_frame)

        if os.path.exists(pred_path) and len(os.listdir(pred_path)) == num_frame:
            continue

        video_frames = pipeline(
            load_image(mid_frame).resize((size, size)),
            height=size,
            width=size,
            num_frames=len(frames),
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02,
            # generator=generator,
        ).frames[0]
        os.makedirs(pred_path, exist_ok=True)
        for i in range(len(frames)):
            img = video_frames[i]
            # img = np.array(img)
            # img = Image.fromarray(img)
            img.save(os.path.join(pred_path, f"{frames[i][0]}.png"))

        # save the pair
        with open(match_sampled_pair_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for i in range(len(frames)):
                writer.writerow({
                    "slide_name": slide_name,
                    "patch_name": patch_name,
                    "frame": frames[i][0],
                    "gt_path": os.path.join(gt_path, f"{frames[i][0]}.png"),
                    "pred_path": os.path.join(pred_path, f"{frames[i][0]}.png"),
                    "blur_degree": blur_degrees[i].item()
                })

    end = time.time()
    elapsed = end - start
    print(f"Match pred Inference took {elapsed:.2f} seconds")

    calculate_metrics(match_sampled_pair_path,
                      gt_np_2048,
                      gt_np_2048_bd,
                      gt_dir,
                      pred_np_2048,
                      fake_match_dir,
                      metric_log,
                      device,
                      threshold=0.2,
                      only_final=True,
                      sample_num_frame=10000)












