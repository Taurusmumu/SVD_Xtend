import os
import argparse
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
from utils import get_from_gen_done, load_config, save_results, load_videos_from_folder, sample_gt_prostate, load_image, convert_to_tensor
from dataloaders import ProstateHarvardDataset, custom_collate_fn_prostate, ProstateAggc22Dataset
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
from common_metrics_on_video_quality.fid_score import FID


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
    sample_gt_prostate(config)

    # --- 2. Load Data ---
    print("Loading dataset...")
    if "harvard" in config['data_root_path']:
        dataset = ProstateHarvardDataset(data_dir=config['data_root_path'],
                                          sample_file_path=config['img_sampled_path'],
                                          num_frames=config['num_frame'])
    if "agg" in config['data_root_path']:
        dataset = ProstateAggc22Dataset(data_dir=config['data_root_path'],
                                          sample_file_path=config['img_sampled_path'],
                                          num_frames=config['num_frame'])
     # Distributed sampler
    sampler = DistributedEvalSampler(dataset, rank=distributed_state.process_index,
                                     num_replicas=distributed_state.num_processes)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             collate_fn=custom_collate_fn_prostate,
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
    fieldnames = ["slide_name", "patch_name", "frame", "gt_path", "pred_path", "class"]
    for metric in config["metrics_to_run"]:
        fieldnames.append(metric)

     # Create result CSV file if it doesn't exist
    result_pair_path = os.path.join(config['output_path'], config['pred_folder'], config['result_pair_path'])
    if not os.path.isfile(result_pair_path):
        with open(result_pair_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    frames = [f"z{i:02d}" for i in range(config["num_frame"])]
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        gt_frame = batch["pixel_values"][0] # PIL Image
        slide_name = batch["slide_name"][0]
        patch_name = batch["patch_name"][0]
        cls = batch["cls"][0]
        condition_mask, gt_video_frames = None, None
        if config["is_project"] is True:
            # print("using projected condition")
            condition_mask = torch.tensor([False, False, False, False, False, True, False, False, False, False, False],
                                device=device)
            gt_video_frames = [gt_frame for i in range(config['num_frame'])]

        # if not patch_name == "patch_1719_11981_54514":
        #     continue

        gt_path = os.path.join(gt_video_path, slide_name, f'{patch_name}.png')
        if not os.path.exists(gt_path):
            # save GT frames
            os.makedirs(os.path.join(gt_video_path, slide_name), exist_ok=True)
            gt_frame.save(gt_path)

        pred_path = os.path.join(pred_video_path, slide_name, patch_name)
        if not (os.path.exists(pred_path) and len(os.listdir(pred_path)) == config['num_frame']):

            pred_video_frames = get_from_gen_done(os.path.join(config['gen_done_path'], slide_name, patch_name))

            if pred_video_frames is None:
                # save Pred frames
                pred_video_frames, _ = pipeline(
                    load_image(gt_frame).resize((config['size'], config['size'])),
                    height=config['size'],
                    width=config['size'],
                    num_frames=config["num_frame"],
                    motion_bucket_id=1.7,
                    fps=7,
                    noise_aug_strength=0.02,
                    gt_images=gt_video_frames,
                    cond=condition_mask
                ) # list of PIL

            scores = metrics_calculator.evaluate(gt_video_frames, pred_video_frames, None)
            # save the data
            with open(result_pair_path, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for i in range(config["num_frame"]):
                    writer.writerow({
                        "slide_name": slide_name,
                        "patch_name": patch_name,
                        "frame": frames[i],
                        "gt_path": gt_path,
                        "pred_path": os.path.join(pred_path, f"{frames[i]}.png"),
                        "class": cls,
                        "ssim_m": scores["ssim_m"][i],
                        "psnr_m": scores["psnr_m"][i],
                        "lpips_m": scores["lpips_m"][i],
                    })

            os.makedirs(pred_path, exist_ok=True)
            for i in range(len(frames)):
                img = pred_video_frames[i]
                img.save(os.path.join(pred_path, f"{frames[i]}.png"))


    end = time.time()
    elapsed = end - start
    print(f"Match pred Inference took {elapsed:.2f} seconds")

    # --- 6. Aggregate and Save Results ---
    output_data = {}
    if "harvard" in config['data_root_path']:
        cls_list = [-1, 0, 1, 2, 3]  # -1 for overall
    if "aggc" in config['data_root_path']:
        cls_list = [-1, 1, 2, 3, 4, 5]
    for cls in cls_list:
        cls_data = {}
        target_cls = cls if cls != -1 else None
        fake_np = load_videos_from_folder(pred_video_path, target_cls)
        vb_calculator = VBench(config["metrics_to_run_vbench"], device)
        vb_pred_result = vb_calculator.evaluate(fake_np, config)
        cls_data["vb_pred"] = vb_pred_result

        # FID and FVD
        metric_file = pd.read_csv(result_pair_path)
        metric_file = metric_file if target_cls is None else metric_file.loc[metric_file["class"] == cls]
        gt_paths = set(metric_file["gt_path"].values)
        metrics = {}
        for metric in config["metrics_to_run"]:
            metrics[f'{metric}_mean_per_frame'] = []
            metrics[f'{metric}_mean_per_video'] = []

        for gt_path in gt_paths:
            video = metric_file.loc[metric_file["gt_path"] == gt_path]
            for metric in config["metrics_to_run"]:
                arr = video[metric].values
                metrics[f'{metric}_mean_per_video'].append(np.mean(arr))
                metrics[f'{metric}_mean_per_frame'].append(arr)

        for metric in config["metrics_to_run"]:
            metrics[f'{metric}_mean_per_frame'] = np.mean(metrics[f'{metric}_mean_per_frame'], axis=0).tolist()
            metrics[f'{metric}_mean_per_video'] = np.mean(metrics[f'{metric}_mean_per_video'])

        cls_data['metric'] = metrics

        pred_path = metric_file["pred_path"].values
        # metric_file = metric_file.loc[(metric_file["blur_degree"] <= config["blur_threshold"])
        #                               & (metric_file["blur_degree"] > 0)]
        metric = FID(list(gt_paths), list(pred_path), device)
        fid = metric.compute(num_samples=config["sample_num_frame"])
        cls_data["fid"] = fid

        pred_path_middle = [fn for fn in pred_path if "z05" in fn]
        metric = FID(list(gt_paths), list(pred_path_middle), device)
        fid_middle = metric.compute(num_samples=config["sample_num_frame"])
        cls_data["fid_middle"] = fid_middle
        output_data[f"class_{cls}"] = cls_data

    save_results(output_data, config)
    print("--- Evaluation complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SVD model evaluation.")
    parser.add_argument('--config', type=str, default="./configs/unet_lora_32_aggc2022.yaml", help="Path to the evaluation YAML config file.")
    args = parser.parse_args()

    main(args.config)
