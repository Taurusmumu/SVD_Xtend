import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import csv
import PIL.Image as Image
from pipes import load_pipe
from utils import load_config, save_results, load_videos_from_folder, sample_gt, load_image, convert_to_tensor
from dataloaders import ProstateSampleDataset, custom_collate_fn
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd


def main(config_path):
    """
    Main function to run the evaluation pipeline.
    """
    # --- 1. Setup ---
    print("Loading configuration...")
    config = load_config(config_path)

    # Create output directory
    os.makedirs(os.path.join(config['output_path'], config['output_folder']), exist_ok=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    print("Loading dataset...")
    dataset = ProstateSampleDataset(
        data_dir=config['data_root_path'],
        split='test'
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             collate_fn=custom_collate_fn,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=0
                                             )  # batch_size = 1
    # --- 3. Load Model ---
    print("Loading model...")
    if len(dataloader) > 0:
        pipeline = load_pipe(config, device)

    # --- 5. Run Evaluation Loop ---
    print("Starting evaluation...")
    pred_video_path = os.path.join(config['output_path'], config['output_folder'])
    os.makedirs(pred_video_path, exist_ok=True)

    start = time.time()

    for batch in tqdm(dataloader):
        mid_frame = batch["pixel_value"][0] # PIL Image
        gt_video_frames = [mid_frame for i in range(config['num_frame'])]
        base_dir = batch["base_dir"][0]
        base_patch = batch["base_patch"][0]
        condition_mask = torch.tensor([False, False, False, False, False, True, False, False, False, False, False],
                            device=device)
        # if not patch_name == "patch_1719_11981_54514":
        #     continue

        pred_path = os.path.join(pred_video_path, base_dir, base_patch)
        if os.path.exists(pred_path) and len(os.listdir(pred_path)) == config['num_frame']:
            continue

        pred_video_frames = pipeline(
            load_image(mid_frame).resize((config['size'], config['size'])),
            height=config['size'],
            width=config['size'],
            num_frames=config['num_frame'],
            motion_bucket_id=1.7,
            fps=7,
            noise_aug_strength=0.02,
            gt_images=gt_video_frames,
            cond=condition_mask
        ).frames[0] # list of PIL

        os.makedirs(pred_path, exist_ok=True)
        for i in range(config['num_frame']):
            img = pred_video_frames[i]
            img = np.array(img)
            img = Image.fromarray(img)
            img.save(os.path.join(pred_path, f"{i}.jpg"))

    end = time.time()
    elapsed = end - start
    print(f"Match pred Inference took {elapsed:.2f} seconds")


    print("--- Generation complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Image Generation Using Pretrained SVD.")
    parser.add_argument('--config', type=str, default="./configs/unet_lora_32.yaml", help="Path to the evaluation YAML config file.")
    args = parser.parse_args()

    main(args.config)
