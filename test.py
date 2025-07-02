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
device = distributed_state.device


def sample_gt_images(gt_dir, sample_file_path, num_frame):
    from scripts.dataloaders import GTSampleDataset
    dataset = GTSampleDataset(data_dir=gt_dir, sample_file_path=sample_file_path, num_frames=num_frame)
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
    layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09", "z10", "z11", "z12", "z13", "z14",
              "z15", "z16", "z17", "z18"]
    size = 256
    sample_num = 2048
    sample_num_frame = 10000
    blur_data_path = "/home/compu/jiamu/SVD_Xtend/image_process/blur_data4.csv"
    split_file_path = "/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt"
    data_root_path = "/ssd2/AMC_zstack_2_patches/pngs_mid"
    sample_GT_done = True

    # sample_pred_match_done, metric_set_1_done = False, False
    # sample_pred_random_done, metric_set_2_done = True, True

    sample_pred_match_done, metric_set_1_done = True, False
    sample_pred_random_done, metric_set_2_done = True, True

    ckp_v = 200000
    version = "output0623"
    num_frame = 11
    sample_file_gt_path = f"/home/compu/jiamu/SVD_Xtend/image_process/{version}/blur_data_sampled_{sample_num}.csv"
    sample_file_random_path = f"/home/compu/jiamu/SVD_Xtend/image_process/{version}/blur_data_sampled1_{sample_num}.csv"
    os.makedirs(f"/home/compu/jiamu/SVD_Xtend/image_process/{version}/", exist_ok=True)
    output_dir = f"/ssd2/AMC_zstack_2_patches/output_for_metrics/{version}"
    pretrained_model_path = f"/ssd2/AMC_zstack_2_patches/{version}/checkpoint-{ckp_v}"
    pretrained_model_name = "stabilityai/stable-video-diffusion-img2vid-xt"
    gt_np = f"/ssd2/AMC_zstack_2_patches/output_for_metrics/{version}/real_videos.npy"
    pred_match_np = f"/ssd2/AMC_zstack_2_patches/output_for_metrics/{version}/pred_match_videos_{version}_{ckp_v}.npy"
    metric_set_1_log = f"/ssd2/AMC_zstack_2_patches/output_for_metrics/{version}/log_match_{version}_{ckp_v}.json"

    pred_random_np = f"/ssd2/AMC_zstack_2_patches/output_for_metrics/{version}/pred_random_videos_{version}_{ckp_v}.npy"
    metric_set_2_log = f"/ssd2/AMC_zstack_2_patches/output_for_metrics/{version}/log_random_{version}_{ckp_v}.json"

    gt_dir = os.path.join(output_dir, "GT")
    fake_match_dir = os.path.join(output_dir, "Pred_match", f"{ckp_v}")
    log_path = os.path.join(fake_match_dir, "test_logs.txt")
    os.makedirs(fake_match_dir, exist_ok=True)
    logger = prepare_logger(log_path)

    fake_random_dir = os.path.join(output_dir, "Pred_random", f"{ckp_v}")
    log_path1 = os.path.join(fake_random_dir, "test_logs.txt")
    os.makedirs(fake_random_dir, exist_ok=True)
    logger1 = prepare_logger(log_path1)

    if os.path.isfile(sample_file_gt_path) is False:
        split_data = pd.read_csv(split_file_path)
        split_data_train = split_data.loc[split_data["train"] == "train"]
        split_data_train_slides = list(split_data_train.iloc[:, 0])

        df = pd.read_csv(blur_data_path)
        df_train = df.loc[df["slide_name"].isin(split_data_train_slides)]
        df_sample = df_train.sample(n=sample_num, random_state=42)
        df_sample.to_csv(sample_file_gt_path, index=False)

    if os.path.isfile(sample_file_random_path) is False:
        split_data = pd.read_csv(split_file_path)
        split_data_train = split_data.loc[split_data["train"] == "train"]
        split_data_train_slides = list(split_data_train.iloc[:, 0])

        df = pd.read_csv(blur_data_path)
        df_train = df.loc[df["slide_name"].isin(split_data_train_slides)]
        df_sample = df_train.sample(n=sample_num, random_state=24)
        df_sample.to_csv(sample_file_random_path, index=False)

    if sample_GT_done is False:
        sample_gt_images(gt_dir, sample_file_gt_path, num_frame)

    if sample_pred_match_done is False:
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

        pipeline.to(device)
        pipeline.enable_model_cpu_offload()
        with distributed_state.main_process_first():
            # for i in tqdm(range(len(os.listdir(gt_dir)))):
            #     if i % distributed_state.num_processes != distributed_state.process_index:
            #         continue

            from scripts.dataloaders import GTSampleDataset

            dataset = GTSampleDataset(data_dir=gt_dir, sample_file_path=sample_file_gt_path, num_frames=num_frame)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
            for batch in tqdm(dataloader):
                middle_frames = batch["middle_frames"]
                slides = batch["slide_name"]
                patches = batch["patch_name"]
                start_indexes = batch["start_index"]
                for i in range(len(slides)):
                    slide = slides[i]
                    patch = patches[i]
                    middle_frame = middle_frames[i]
                    start_index = start_indexes[i]
                    frames = layers[start_index: start_index + num_frame]
                    out_path = os.path.join(
                        fake_match_dir, slide, patch,
                    )
                    if os.path.exists(out_path) and len(os.listdir(out_path)) == num_frame:
                        continue

                    video_frames = pipeline(
                        load_image(middle_frame).resize((size, size)),
                        height=size,
                        width=size,
                        num_frames=len(frames),
                        decode_chunk_size=8,
                        motion_bucket_id=127,
                        fps=7,
                        noise_aug_strength=0.02,
                        # generator=generator,
                    ).frames[0]
                    os.makedirs(out_path, exist_ok=True)
                    for i in range(len(frames)):
                        img = video_frames[i]
                        # img = np.array(img)
                        # img = Image.fromarray(img)
                        img.save(os.path.join(fake_match_dir, slide, patch, f"{frames[i]}.png"))

            end = time.time()
            elapsed = end - start
            print(f"Match pred Inference took {elapsed:.2f} seconds") # 7 hours

    if metric_set_1_done is False:

        if distributed_state.is_main_process:
            print("Calculating metrics for match predictions...")
            start = time.time()
            if not os.path.isfile(gt_np):
                real_videos = load_videos_from_folder(gt_dir)
                np.save(gt_np, real_videos.astype(np.float32))

            if not os.path.isfile(pred_match_np):
                fake_videos = load_videos_from_folder(fake_match_dir)
                np.save(pred_match_np, fake_videos.astype(np.float32))

            real_videos = np.load(gt_np).astype(np.float32)   # [N, T, H, W, C]
            fake_videos = np.load(pred_match_np).astype(np.float32)

            # Convert to TensorFlow tensors
            real_tensor = convert_to_tensor(real_videos)
            fake_tensor = convert_to_tensor(fake_videos)
            import json
            result = {}
            # result['fvd'] = calculate_fvd(real_tensor, fake_tensor, device, method='styleganv', only_final=only_final)
            only_final = False
            # only_final = True
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
            with open(metric_set_1_log, "w") as f:
                json.dump(result, f, indent=4)

            end_smi = time.time()
            elapsed = end_smi - start
            print(f"Similarity took {elapsed:.2f} seconds")

    if sample_pred_random_done is False:

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

        pipeline.to(device)
        pipeline.enable_model_cpu_offload()

        random_df = pd.read_csv(sample_file_random_path)
        layers = list(random_df.columns[2:-2])
        slides = list(random_df["slide_name"])
        patches = list(random_df["patch_name"])
        start_layers = list(random_df["start_indices"])

        with distributed_state.main_process_first():
            for i in tqdm(range(len(slides))):
                # if i % distributed_state.num_processes != distributed_state.process_index:
                #     continue

                slide, patch = slides[i], patches[i]
                print(slide)
                out_path = os.path.join(
                    fake_random_dir, slide, patch.split('.')[0]
                )
                if os.path.exists(out_path) and len(os.listdir(out_path)) == num_frame:
                    continue

                frames = layers[start_layers[i]: start_layers[i] + num_frame]
                mid_frame = frames[(len(frames) - 1) // 2]
                source_path = os.path.join(data_root_path, slide, mid_frame, patch)
                video_frames = pipeline(
                    load_image(source_path).resize((size, size)),
                    height=size,
                    width=size,
                    num_frames=num_frame,
                    decode_chunk_size=8,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.02,
                    # generator=generator,
                ).frames[0]
                os.makedirs(out_path, exist_ok=True)
                for f in range(num_frame):
                    img = video_frames[f]
                    print(os.path.join(fake_random_dir, slide, patch.split('.')[0], f"{frames[f]}.png"))
                    img.save(os.path.join(fake_random_dir, slide, patch.split('.')[0], f"{frames[f]}.png"))

            end = time.time()
            elapsed = end - start
            print(f"Random pred Inference took {elapsed:.2f} seconds") # 7 hours

    if metric_set_2_done is False:

        if distributed_state.is_main_process:
            print("Calculating metrics for random predictions...")
            start = time.time()
            import json
            result = {}
            metric = FID(gt_dir, fake_random_dir, logger1, device)
            fid = metric.compute(num_samples=sample_num_frame)
            result['fid'] = fid
            with open(metric_set_2_log, "w") as f:
                json.dump(result, f, indent=4)
            #
            end_fid = time.time()
            elapsed = end_fid - start
            print(f"FID took {elapsed:.2f} seconds")

            if not os.path.isfile(gt_np):
                real_videos = load_videos_from_folder(gt_dir)
                np.save(gt_np, real_videos.astype(np.float32))
            else:
                real_videos = np.load(gt_np).astype(np.float32)

            if not os.path.isfile(pred_random_np):
                fake_random_videos = load_videos_from_folder(fake_random_dir)
                np.save(pred_random_np, fake_random_videos.astype(np.float32))
            else:
                fake_random_videos = np.load(pred_random_np).astype(np.float32)

            real_tensor = convert_to_tensor(real_videos)
            fake_tensor = convert_to_tensor(fake_random_videos)

            end_fvd = time.time()
            only_final = True
            result['fvd'] = calculate_fvd(real_tensor, fake_tensor, device, method='styleganv', only_final=only_final)
            with open(metric_set_2_log, "w") as f:
                json.dump(result, f, indent=4)
            elapsed = end_fvd - end_fid
            print(f"FVD took {elapsed:.2f} seconds")








