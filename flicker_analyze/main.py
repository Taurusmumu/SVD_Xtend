import os.path
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path,
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def normalize_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    normalized = (image / 255.0 - 0.5) * 2.0
    return normalized

def get_first_derivative(image_3d):
    differences = np.diff(image_3d, axis=1)
    differences_mean = np.mean(np.abs(differences), axis=1)
    differences_max = np.max(np.abs(differences), axis=1)
    return differences, differences_mean, differences_max

def get_second_derivative(image_3d):
    differences = np.diff(image_3d, n=2, axis=1)
    differences_mean = np.mean(np.abs(differences), axis=1)
    differences_max = np.max(np.abs(differences), axis=1)
    return differences, differences_mean, differences_max


if __name__ == "__main__":

    sampled_fn = "/ssd1/AMC_zstack_2_patches/output_for_metrics/output1004_unet_temp_lora32/gt_video_sampled.csv"
    data_root_gt = "/ssd1/AMC_zstack_2_patches/output_for_metrics/output1004_unet_temp_lora32/gt"
    data_root_pred = "/ssd1/AMC_zstack_2_patches/output_for_metrics/output1004_unet_temp_lora32/40000_v2_proj"
    data_output = "/ssd2/AMC_zstack_2_patches/output_for_metrics/check_flicker"
    slide_name = "24S 050169;A;6;;FA0824;1_241226_150108"
    patch_name = "patch_1623_10027_32574"

    total_gt, total_pred = [], []


    df = pd.read_csv(sampled_fn)
    for index, row in df.iterrows():
        slide_name = row['slide_name']
        patch_name = row['patch_name'].split('.')[0]
        img_list_gt, img_list_pred = [], []
        for fn in os.listdir(os.path.join(data_root_gt, slide_name, patch_name)):
            image = normalize_image(os.path.join(data_root_gt, slide_name, patch_name, fn))
            img_list_gt.append(image)

            image = normalize_image(os.path.join(data_root_pred, slide_name, patch_name, fn))
            img_list_pred.append(image)
        total_gt.append(img_list_gt)
        total_pred.append(img_list_pred)
    # os.makedirs(os.path.join(data_output, slide_name), exist_ok=True)
    # output_path = os.path.join(data_output, slide_name, f"{patch_name}_flicker.gif")
    # export_to_gif(img_list, output_path, fps=7)

    total_gt = np.array(total_gt)
    total_pred = np.array(total_pred)

    pred_diff, pred_diff_mean, pred_diff_max = get_first_derivative(total_pred)
    gt_diff, gt_diff_mean, gt_diff_max = get_first_derivative(total_gt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))
    # Plot 1: The instability heatmap
    im = ax1.hist(gt_diff_mean.flatten(), range=[0,0.5], bins=100)
    ax1.set_title(f'GT (mean: {np.round(np.mean(gt_diff_max), 2)}, std: {np.round(np.std(gt_diff_max), 2)})')
    ax1.set_ylabel('Histogram Values', fontsize=20)
    ax1.set_xlabel('First Derivative Mean', fontsize=20)
    im2 = ax2.hist(pred_diff_mean.flatten(), range=[0,0.5], bins=100)
    ax2.set_title('Pred', fontsize=20)
    ax2.set_ylabel('Histogram Values', fontsize=20)
    ax2.set_xlabel('First Derivative Mean', fontsize=20)
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))
    # Plot 1: The instability heatmap
    im = ax1.hist(gt_diff_max.flatten(), bins=50)
    ax1.set_title('GT')
    ax1.set_ylabel('Histogram Values', fontsize=20)
    ax1.set_xlabel('First Derivative Max', fontsize=20)
    im2 = ax2.hist(pred_diff_max.flatten(), bins=50)
    ax2.set_title('Pred', fontsize=20)
    ax2.set_ylabel('Histogram Values', fontsize=20)
    ax2.set_xlabel('First Derivative Max', fontsize=20)
    plt.tight_layout()
    plt.show()

    pred_diff, pred_diff_mean, pred_diff_max = get_second_derivative(total_pred)
    gt_diff, gt_diff_mean, gt_diff_max = get_second_derivative(total_gt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))
    # Plot 1: The instability heatmap
    im = ax1.hist(gt_diff_mean.flatten(), bins=50)
    ax1.set_title('GT')
    ax1.set_ylabel('Histogram Values')
    ax1.set_xlabel('Second Derivative Mean')
    im2 = ax2.hist(pred_diff_mean.flatten(), bins=50)
    ax2.set_title('Pred')
    ax2.set_ylabel('Histogram Values')
    ax2.set_xlabel('Second Derivative Mean')
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))
    # Plot 1: The instability heatmap
    im = ax1.hist(gt_diff_max.flatten(), bins=50)
    ax1.set_title('GT')
    ax1.set_ylabel('Histogram Values')
    ax1.set_xlabel('Second Derivative Max')
    im2 = ax2.hist(pred_diff_max.flatten(), bins=50)
    ax2.set_title('Pred')
    ax2.set_ylabel('Histogram Values')
    ax2.set_xlabel('Second Derivative Max')
    plt.tight_layout()
    plt.show()


