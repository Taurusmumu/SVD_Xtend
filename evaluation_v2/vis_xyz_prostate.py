import PIL.Image
import numpy as np
import imageio.v3 as iio
import imageio
import os
import argparse
from utils import load_config, export_to_gif
import pandas as pd
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def gen_xyz_gif(paths, out_dir, base_name, duration=150):

    frames = []
    for path in paths:
        frame = Image.open(path)
        draw = ImageDraw.Draw(frame)
        draw.text((10, 10), f"{path.split('/')[-1].split('.')[0]}", (255, 255, 255), font=ImageFont.truetype("arial.ttf", 20))
        frames.append(frame)
    export_to_gif(frames, os.path.join(out_dir, f"{base_name}_xy.gif"), duration=duration)
    # stack_xy = np.stack(frames, axis=0)
    stack = np.stack([iio.imread(p) for p in paths], axis=0)  # (Z, H, W, 3)
    if stack.dtype != np.float32:
        stack = stack.astype(np.float32) / 255.0

    def make_gif(frames, path, fps=5):
        frames_uint8 = [(np.clip(f * 255, 0, 255)).astype(np.uint8) for f in frames]
        imageio.mimsave(path, frames_uint8, duration=duration)

    # frames_xy = [stack[z, :, :, :] for z in range(stack.shape[0])]
    # make_gif(frames_xy, os.path.join(out_dir, f"{base_name}_xy.gif"))

    frames_xz = [np.rot90(np.transpose(stack[:, y, :, :], (1, 0, 2)), k=1) for y in range(stack.shape[1])]
    make_gif(frames_xz, os.path.join(out_dir, f"{base_name}_xz.gif"))

    frames_yz = [np.transpose(stack[:, :, x, :], (1, 0, 2)) for x in range(stack.shape[2])]
    make_gif(frames_yz, os.path.join(out_dir,  f"{base_name}_yz.gif"))


def main(config_path):

    print("Loading configuration...")
    config = load_config(config_path)
    video_sampled_path = config['img_sampled_path']
    video_sampled = pd.read_csv(video_sampled_path)
    example_gif = video_sampled.sample(25, random_state=42)

    slide_patch_pairs = []
    # slide_patch_pairs = [("24S 067519;A;4;;FA0824;1_241225_005143", "patch_4690_19612_32041")]
    for index, row in example_gif.iterrows():
        fn = row['file_path']
        slide = '/'.join(fn.split('/')[-3:-1]) if 'agg' in config['data_root_path'] else fn.split('/')[-2]
        patch = fn.split('/')[-1].split('.')[0]
        slide_patch_pairs.append((slide, patch))

    for info in slide_patch_pairs:
        patch_name = info[1]
        slide = info[0]
        gt_img_path = os.path.join(config['output_path'], config['gt_folder'], slide, f'{patch_name}.png')
        pred_video_path = os.path.join(config['gen_done_path'], slide, patch_name)

        out_dir = os.path.join(config['output_path'], config['gif_folder'], slide, patch_name)
        os.makedirs(out_dir, exist_ok=True)
        Image.open(gt_img_path).save(os.path.join(out_dir, f"gt_xy.png"))

        pred_paths = [f'{pred_video_path}/{i}.jpg' for i in range(len(os.listdir(pred_video_path)))]
        gen_xyz_gif(pred_paths, out_dir, "pred_6", duration=150)
        gen_xyz_gif(pred_paths[5:-5], out_dir, "pred_3", duration=150)

        # paths = [f"/home/kwaklab_103/Desktop/patch_9197_25059_39671_pred/z{i:02d}.png" for i in range(6, 17)]
        # out_dir = "/home/kwaklab_103/Desktop/zstack_gifs_pred"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Getting Gif.")
    parser.add_argument('--config', type=str, default="./configs/unet_lora_32_prostate.yaml", help="Path to the evaluation YAML config file.")
    args = parser.parse_args()

    main(args.config)