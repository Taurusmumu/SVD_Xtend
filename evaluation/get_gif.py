import os
import argparse
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from utils import load_config, export_to_gif
import pandas as pd



def main(config_path):
    """
    Main function to run the evaluation pipeline.
    """
    # --- 1. Setup ---
    print("Loading configuration...")
    config = load_config(config_path)
    video_sampled_path = os.path.join(config['output_path'], config['video_sampled_path'])
    video_sampled = pd.read_csv(video_sampled_path)
    example_gif = video_sampled.sample(100, random_state=42)

    slide_patch_pairs = []
    for index, row in example_gif.iterrows():
        slide = row['slide_name']
        patch = row['patch_name'].split('.')[0]
        slide_patch_pairs.append((slide, patch))

    for info in slide_patch_pairs:
        patch_name = info[1]
        slide = info[0]
        gt_video_path = os.path.join(config['output_path'], config['gt_folder'], slide, patch_name)
        pred_video_path = os.path.join(config['output_path'], config['pred_folder'], slide, patch_name)
        if not os.path.exists(gt_video_path) or not os.path.exists(pred_video_path):
            continue

        gif_path = os.path.join(config['output_path'], config['gif_folder'], slide)
        gt_gif_path = os.path.join(gif_path, f"{patch_name}_GT.gif")
        pred_gif_path = os.path.join(gif_path, f"{patch_name}_Pred.gif")
        os.makedirs(gif_path, exist_ok=True)

        frames, pred_frames = [], []
        for path in sorted(os.listdir(gt_video_path)):
            frame_path = os.path.join(gt_video_path, path)
            print(frame_path)
            frame = Image.open(frame_path)
            draw = ImageDraw.Draw(frame)
            draw.text((10, 10), f"{path.split('.')[0]}", (255, 255, 255), font=ImageFont.truetype("arial.ttf", 20))
            frames.append(frame)

        export_to_gif(frames, gt_gif_path, fps=7)

        for path in sorted(os.listdir(pred_video_path)):
            frame_path = os.path.join(pred_video_path, path)
            print(frame_path)
            frame = Image.open(frame_path)
            draw = ImageDraw.Draw(frame)
            draw.text((10, 10), f"{path.split('.')[0]}", (255, 255, 255), font=ImageFont.truetype("arial.ttf", 20))
            pred_frames.append(frame)

        export_to_gif(pred_frames, pred_gif_path, fps=7)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Getting Gif.")
    parser.add_argument('--config', type=str, default="./configs/output1004_unet_temp_lora32_aug.yaml", help="Path to the evaluation YAML config file.")
    args = parser.parse_args()

    main(args.config)



