import os
import argparse

import PIL.Image

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
import cv2
from eval_sampler import DistributedEvalSampler
from accelerate import PartialState
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import csv
import PIL.Image as Image
from metrics import MetricsCalculator, VBench
from pipes import load_pipe
from utils import load_config, save_results, load_videos_from_folder, sample_gt, load_image, convert_to_tensor
from dataloaders import GTSampleDataset, custom_collate_fn
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd


def main(config_path):
    """
    Main function to run the evaluation pipeline.
    """
    distributed_state = PartialState()
    # --- 1. Setup ---
    print("Loading configuration...")
    config = load_config(config_path)

    # Create output directory
    os.makedirs(config['output_path'], exist_ok=True)

    # Set up device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = distributed_state.device
    print(f"Using device: {device}")

    # sample GT to evaluate the pair Pred
    sample_gt(config)

    # --- 2. Load Data ---
    print("Loading dataset...")
    dataset = GTSampleDataset(data_dir=config['data_root_path'],
                              blur_threshold=config['blur_threshold'],
                              sample_file_path=config['video_sampled_path'],
                              num_frames=config['num_frame'])
    sampler = DistributedEvalSampler(dataset, rank=distributed_state.process_index,
                                     num_replicas=distributed_state.num_processes)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             collate_fn=custom_collate_fn,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0,
                                             sampler=sampler
                                             )  # batch_size = 1

    # --- 3. Load Model ---
    print("Loading model...")
    if len(dataloader) > 0:
        pipeline = load_pipe(config, device)

    # --- 4. Initialize Metrics ---
    metrics_calculator = MetricsCalculator(config['metrics_to_run'], device)

    # --- 5. Run Evaluation Loop ---
    print("Starting evaluation...")
    gt_video_path = config['gt_folder']
    pred_video_path = os.path.join(config['output_path'], config['pred_folder'])
    os.makedirs(gt_video_path, exist_ok=True)
    os.makedirs(pred_video_path, exist_ok=True)

    start = time.time()
    fieldnames = ["slide_name", "patch_name", "frame", "gt_path", "pred_path", "blur_degree"]
    for metric in config["metrics_to_run"]:
        fieldnames.append(metric)

     # Create result CSV file if it doesn't exist
    result_pair_path = os.path.join(config['output_path'], config['pred_folder'], config['result_pair_path'])
    if not os.path.isfile(result_pair_path):
        with open(result_pair_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # for batch in tqdm(dataloader):
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        gt_video_frames = batch["pixel_values"][0] # PIL Image
        blur_degrees = batch["blur_degrees"][0]
        blur_mask = np.array([float(blur_degree) for blur_degree in blur_degrees]) < config['blur_threshold']
        condition_mask = None
        if config["is_project"] is True:
            # print("using projected condition")
            condition_mask = torch.tensor([False, False, False, False, False, True, False, False, False, False, False],
                                device=device)
        slide_name = batch["slide_name"][0]
        patch_name = batch["patch_name"][0]
        frames = batch["frames"][0]
        # if not patch_name == "patch_1719_11981_54514":
        #     continue

        gt_path = os.path.join(gt_video_path, slide_name, patch_name)
        if not (os.path.exists(gt_path) and len(os.listdir(gt_path)) == config['num_frame']):
            # save GT frames
            os.makedirs(gt_path, exist_ok=True)
            for i in range(len(frames)):
                img = gt_video_frames[i]
                img.save(os.path.join(gt_path, f"{frames[i]}.png"))

        pred_path = os.path.join(pred_video_path, slide_name, patch_name)
        mid_frame = gt_video_frames[len(gt_video_frames)//2]

        if not (os.path.exists(pred_path) and len(os.listdir(pred_path)) == config['num_frame']):
            # save Pred frames
            pred_video_frames, _ = pipeline(
                load_image(mid_frame).resize((config['size'], config['size'])),
                height=config['size'],
                width=config['size'],
                num_frames=len(frames),
                motion_bucket_id=1.7,
                fps=7,
                noise_aug_strength=0.02,
                gt_images=gt_video_frames,
                cond=condition_mask
            ) # list of PIL

            os.makedirs(pred_path, exist_ok=True)
            for i in range(len(frames)):
                img = pred_video_frames[i]
                img = np.array(img)
                img = Image.fromarray(img)
                img.save(os.path.join(pred_path, f"{frames[i]}.png"))

        pred_video_frames = [PIL.Image.open(os.path.join(pred_path, frame) for frame in os.listdir(pred_path))]
        scores = metrics_calculator.evaluate(gt_video_frames, pred_video_frames, blur_mask)
        # save the data
        with open(result_pair_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for i in range(len(frames)):
                writer.writerow({
                    "slide_name": slide_name,
                    "patch_name": patch_name,
                    "frame": frames[i],
                    "gt_path": os.path.join(gt_path, f"{frames[i]}.png"),
                    "pred_path": os.path.join(pred_path, f"{frames[i]}.png"),
                    "blur_degree": blur_degrees[i],
                    "ssim": scores["ssim"],
                    "psnr": scores["psnr"],
                    "lpips": scores["lpips"],
                    "ssim_ver": scores["ssim_ver"],
                    "psnr_ver": scores["psnr_ver"],
                    "lpips_ver": scores["lpips_ver"],
                    "ssim_m": scores["ssim_m"][i],
                    "psnr_m": scores["psnr_m"][i],
                    "lpips_m": scores["lpips_m"][i],
                })

    end = time.time()
    elapsed = end - start
    print(f"Match pred Inference took {elapsed:.2f} seconds")


    # --- 6. Aggregate and Save Results ---
    output_data = {}

    read_np, fake_np = load_videos_from_folder(gt_video_path), load_videos_from_folder(pred_video_path)
    real_tensor, fake_tensor = convert_to_tensor(read_np), convert_to_tensor(fake_np)

    vb_calculator = VBench(config["metrics_to_run_vbench"], device)
    vb_real_result = vb_calculator.evaluate(read_np, config)
    output_data["vb_real"] = vb_real_result
    vb_pred_result = vb_calculator.evaluate(fake_np, config)
    output_data["vb_pred"] = vb_pred_result

    # FID and FVD
    metric_file = pd.read_csv(result_pair_path)
    for metric in config["metrics_to_run"]:
        output_data[metric] = np.mean(metric_file[metric].values)

    from common_metrics_on_video_quality.fid_score import FID
    metric_file = metric_file.loc[(metric_file["blur_degree"] <= config["blur_threshold"])
                                  & (metric_file["blur_degree"] >= 0)]
    metric = FID(metric_file["gt_path"].values, metric_file["pred_path"].values, device)
    fid = metric.compute(num_samples=config["sample_num_frame"])

    fvd = calculate_fvd(real_tensor, fake_tensor, device, method='styleganv', only_final=True)
    output_data["fid"] = fid
    output_data["fvd"] = fvd

    # Save detailed and summary results

    save_results(output_data, config)
    print("--- Evaluation complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SVD model evaluation.")
    parser.add_argument('--config', type=str, default="./configs/unet_dir.yaml", help="Path to the evaluation YAML config file.")
    args = parser.parse_args()

    main(args.config)
