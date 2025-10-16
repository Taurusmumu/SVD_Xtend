import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
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
from metrics import MetricsCalculator, VBench
from pipes import load_pipe
from utils import load_config, save_results, load_videos_from_folder, sample_gt, load_image, convert_to_tensor
from dataloaders import GTSampleDataset, custom_collate_fn
from common_metrics_on_video_quality.calculate_fvd import calculate_fvd


def main(config_path):
    """
    Main function to run the evaluation pipeline.
    """
    # --- 1. Setup ---
    print("Loading configuration...")
    config = load_config(config_path)

    # Create output directory
    os.makedirs(config['output_path'], exist_ok=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # sample GT to evaluate the pair Pred
    sample_gt(config)

    # --- 2. Load Data ---
    print("Loading dataset...")
    dataset = GTSampleDataset(data_dir=config['data_root_path'],
                              sample_file_path=os.path.join(config['output_path'], config['video_sampled_path']),
                              num_frames=config['num_frame'])
    dataloader = torch.utils.data.DataLoader(dataset,
                                             collate_fn=custom_collate_fn,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=0
                                             )  # batch_size = 1

    # --- 3. Load Model ---
    print("Loading model...")
    if len(dataloader) > 0:
        pipeline = load_pipe(config, device)

    # --- 4. Initialize Metrics ---
    metrics_calculator = MetricsCalculator(config['metrics_to_run'], device)

    # --- 5. Run Evaluation Loop ---
    print("Starting evaluation...")
    gt_video_path = os.path.join(config['output_path'], config['gt_folder'])
    pred_video_path = os.path.join(config['output_path'], config['pred_folder'])
    os.makedirs(gt_video_path, exist_ok=True)
    os.makedirs(pred_video_path, exist_ok=True)
    all_results = []

    start = time.time()
    fieldnames = ["slide_name", "patch_name", "frame", "gt_path", "pred_path", "blur_degree"]
    for metric in config["metrics_to_run"]:
        fieldnames.append(metric)
        # fieldnames.append(f"{metric}_ver")

     # Create result CSV file if it doesn't exist
    result_pair_path = os.path.join(config['output_path'], config['pred_folder'], config['result_pair_path'])
    if not os.path.isfile(result_pair_path):
        with open(result_pair_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for batch in tqdm(dataloader):
        gt_video_frames = batch["pixel_values"][0] # PIL Image
        blur_degrees = batch["blur_degrees"][0]
        slide_name = batch["slide_name"][0]
        patch_name = batch["patch_name"][0]
        frames = batch["frames"][0]

        gt_path = os.path.join(gt_video_path, slide_name, patch_name)
        if not (os.path.exists(gt_path) and len(os.listdir(gt_path)) == config['num_frame']):
            # save GT frames
            os.makedirs(gt_path, exist_ok=True)
            for i in range(len(frames)):
                img = gt_video_frames[i]
                img.save(os.path.join(gt_path, f"{frames[i]}.png"))

        pred_path = os.path.join(pred_video_path, slide_name, patch_name)
        mid_frame = gt_video_frames[len(gt_video_frames)//2]

        if os.path.exists(pred_path) and len(os.listdir(pred_path)) == config['num_frame']:
            continue

        pred_video_frames = pipeline(
            load_image(mid_frame).resize((config['size'], config['size'])),
            height=config['size'],
            width=config['size'],
            num_frames=len(frames),
            decode_chunk_size=8,
            motion_bucket_id=1.7,
            fps=7,
            noise_aug_strength=0.02,
        ).frames[0] # list of PIL

        scores = metrics_calculator.evaluate(gt_video_frames, pred_video_frames)

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
                    "ssim": scores["ssim"][i],
                    "psnr": scores["psnr"][i],
                    "lpips": scores["lpips"][i],
                    "ssim_ver": scores["ssim_ver"][i],
                    "psnr_ver": scores["psnr_ver"][i],
                    "lpips_ver": scores["lpips_ver"][i],
                })

        os.makedirs(pred_path, exist_ok=True)
        for i in range(len(frames)):
            img = pred_video_frames[i]
            img = np.array(img)
            img = Image.fromarray(img)
            img.save(os.path.join(pred_path, f"{frames[i]}.png"))

    end = time.time()
    elapsed = end - start
    print(f"Match pred Inference took {elapsed:.2f} seconds")


    # --- 6. Aggregate and Save Results ---
    output_data = {
    }

    read_np, fake_np = load_videos_from_folder(gt_video_path), load_videos_from_folder(pred_video_path)
    real_tensor, fake_tensor = convert_to_tensor(read_np), convert_to_tensor(fake_np)

    vb_calculator = VBench(config["metrics_to_run_vbench"], device)
    vb_real_result = vb_calculator.evaluate(read_np, config)
    output_data["vb_real"] = vb_real_result
    vb_pred_result = vb_calculator.evaluate(fake_np, config)
    output_data["vb_pred"] = vb_pred_result

    # FID and FVD
    metric_file = pd.read_csv(result_pair_path)
    metric_file = metric_file.loc[(metric_file["blur_degree"] <= config["blur_threshold"])
                                  & (metric_file["blur_degree"] > 0)]
    # right_idx = np.where(metric_file['psnr'].values > 1)[0]
    # metric_file_new = metric_file.iloc[right_idx] # metric_file_new["patch_name"].values
    # metric_file = metric_file.sample(config["sample_num_frame"], random_state=42)
    for metric in config["metrics_to_run"]:
        output_data[metric] = np.mean(metric_file[metric].values)

    # metric_file_correct = metric_file.iloc[np.where(metric_file['psnr'].values > 1)[0]]
    # metric_file_wrong = metric_file.iloc[np.where(metric_file['psnr'].values <= 1)[0]]
    #
    # def load_videos_from_folder_new(dir_path, patch_list):
    #     videos = []
    #     for slide in os.listdir(dir_path):
    #         if not os.path.isdir(os.path.join(dir_path, slide)):
    #             continue
    #         slide_path = os.path.join(dir_path, slide)
    #         for patch in os.listdir(slide_path):
    #             if patch not in patch_list:
    #                 continue
    #             patch_path = os.path.join(slide_path, patch)
    #
    #             frames = sorted(os.listdir(patch_path))
    #             frame_paths = [os.path.join(patch_path, frame) for frame in frames]
    #             frame_np = [np.array(Image.open(frame_path).convert('RGB')) for frame_path in frame_paths]
    #             videos.append(np.stack(frame_np))
    #     return np.stack(videos)
    #
    # gt_correct_np, pred_correct_np = load_videos_from_folder_new(gt_video_path, metric_file_correct[
    #     "patch_name"].values), load_videos_from_folder_new(pred_video_path, metric_file_correct["patch_name"].values)
    # gt_wrong_np, pred_wrong_np = load_videos_from_folder_new(gt_video_path, metric_file_wrong[
    #     "patch_name"].values), load_videos_from_folder_new(pred_video_path, metric_file_wrong["patch_name"].values)
    # print(np.mean(gt_correct_np))
    # print(np.mean(pred_correct_np))
    # print(np.mean(gt_wrong_np))
    # print(np.mean(pred_wrong_np))
    #
    # from common_metrics_on_video_quality.calculate_psnr import calculate_psnr, calculate_psnr_vertical
    # from common_metrics_on_video_quality.calculate_ssim import calculate_ssim, calculate_ssim_vertical
    # from common_metrics_on_video_quality.calculate_lpips import calculate_lpips, calculate_lpips_vertical
    # result = {}
    # only_final = True
    # result['ssim'] = calculate_ssim(real_tensor, fake_tensor, only_final=only_final)
    # result['psnr'] = calculate_psnr(real_tensor, fake_tensor, only_final=only_final)
    # result['lpips'] = calculate_lpips(real_tensor, fake_tensor, device, only_final=only_final)
    # result['ssim_final'] = np.mean(result['ssim']['value'])
    # result['psnr_final'] = np.mean(result['psnr']['value'])
    # result['lpips_final'] = np.mean(result['lpips']['value'])

    from common_metrics_on_video_quality.fid_score import FID
    metric = FID(metric_file["gt_path"].values, metric_file["pred_path"].values, device)
    fid = metric.compute(num_samples=config["sample_num_frame"])
    # metrics_dict_fid = torch_fidelity.calculate_metrics(
    #     input1=gt_video_path,
    #     input2=pred_video_path,
    #     cuda=torch.cuda.is_available(),
    #     isc=False,
    #     fid=True,
    #     kid=False,
    #     verbose=False,
    # )


    fvd = calculate_fvd(real_tensor, fake_tensor, device, method='styleganv', only_final=True)
    output_data["fid"] = fid
    output_data["fvd"] = fvd

    # Save detailed and summary results

    save_results(output_data, config)
    print("--- Evaluation complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SVD model evaluation.")
    parser.add_argument('--config', type=str, default="./configs/output1004_unet_temp_lora32.yaml", help="Path to the evaluation YAML config file.")
    args = parser.parse_args()

    main(args.config)
