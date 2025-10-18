import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import skimage.io
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def calculate_tissue_ratio(image_path: str, low_s: int = 50, high_v: int = 200) -> float:
    """
    Calculates the ratio of tissue to the total image area.

    Args:
        image_path (str): The path to the input image file.
        low_s (int): The threshold for saturation to distinguish background.
                     Lower values are more lenient.
        high_v (int): The threshold for value (brightness) to distinguish background.
                      Higher values target brighter areas.

    Returns:
        float: A value between 0.0 and 1.0 representing the tissue ratio.
    """
    # 1. Read the image
    image_rgb = cv2.imread(image_path)
    if image_rgb is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 2. Convert the image from BGR to HSV color space
    # OpenCV reads images as BGR, so we'll work with that and convert to HSV.
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV)

    # 3. Define thresholds to create a mask for the background
    # The background is typically white, which has low saturation and high value (brightness).
    # Hue can be anything, so we span its full range (0-179 in OpenCV).
    lower_bound = np.array([0, 0, high_v])
    upper_bound = np.array([179, low_s, 255])

    # 4. Create the background mask and invert it to get the tissue mask
    background_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    tissue_mask = cv2.bitwise_not(background_mask)

    # 5. Calculate the ratio
    # Get the total number of pixels in the image
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]

    # Count the number of non-zero (white) pixels in the tissue mask
    tissue_pixels = cv2.countNonZero(tissue_mask)

    # Avoid division by zero for completely black images
    if total_pixels == 0:
        return 0.0

    tissue_ratio = tissue_pixels / total_pixels

    return tissue_ratio


def check_contrast(clear_layer, slide_name, patch_name, target_layers, root_dir):
    try:
        frame1_path = os.path.join(root_dir, slide_name, clear_layer, patch_name)
        frame1 = cv2.imread(frame1_path)
        if frame1 is None:
            print(f"Error: Could not read reference frame: {frame1_path}. Skipping patch.")
            return None

        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        mag_list = []
        # print("Clear layer is: ", clear_layer)
        # print(f"Aligning all layers to reference: {clear_layer}/{patch_name}")

        for l_idx, next_layer in enumerate(target_layers):
            frame2_path = os.path.join(root_dir, slide_name, next_layer, patch_name)
            # try:
            frame2 = cv2.imread(frame2_path)
            if frame2 is None:
                print(f"  - Warning: Missing frame for layer {next_layer}. Appending score -1.")
                return None
            # Convert to grayscale
            # Create a grayscale version specifically forframe1 registration
            next_frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame_gray, None, 0.5, 3, 8, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag[np.where(mag == np.inf)] = 0
            mag_list.append(float(np.mean(mag)))

        return mag_list

    except Exception as e:
        print(f"Error processing {slide_name} {patch_name}: {e}")
        return None


if __name__ == '__main__':

    anno_path = "./blur_motion_data4.csv"
    root_dir = "/ssd2/AMC_zstack_2_patches_warp/pngs_mid"
    text_file_path = "/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt"
    target_layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09",
                     "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]

    blur_degree_dict = {}
    start_layer_dict = {}
    motion_dict = {}

    flag = 0
    with open(anno_path, "r") as rf:
        rf.readline()

        for line in tqdm(rf.readlines(), desc="Processing data"):
            # flag += 1
            # if flag == 100:
            #     break
            line_split = line.strip().split(",")
            slide_name = line_split[0]
            patch_name = line_split[1]
            start_layer = int(line_split[-3])
            end_layer = int(line_split[-2])
            clear_layer = int(line_split[-1])
            # if not patch_name == "patch_0_4694_18733.png":
            #     continue
            frame1_path = os.path.join(root_dir, slide_name, target_layers[clear_layer], patch_name)
            ratio = calculate_tissue_ratio(frame1_path, low_s=50, high_v=200)

            if ratio >= 0.5:
                if slide_name not in motion_dict:
                    start_layer_dict[slide_name] = {}
                    blur_degree_dict[slide_name] = {}
                    motion_dict[slide_name] = {}

                motion_blur_list = [score for score in line_split[2:-3]]
                blur_scores = [float(score.split(';')[1]) for score in motion_blur_list]
                motion_scores = [float(score.split(';')[1]) for score in motion_blur_list]

                if (-1 in motion_scores) or (-1 in blur_scores):
                    print(f"Skipping {slide_name}/{patch_name} due to missing data.")
                    continue

                start_layer_dict[slide_name][patch_name] = (start_layer, end_layer, clear_layer)
                blur_degree_dict[slide_name][patch_name] = blur_scores
                motion_dict[slide_name][patch_name] = motion_scores

    with open("./blur_motion_data5.csv", "w") as wf:
        wf.write("slide_name,patch_name,{},start_indices,end_indices,min_indices\n".format(",".join(target_layers)))
        for slide_name, slide_data in blur_degree_dict.items():
            for patch_name, blur_scores in slide_data.items():
                motion_scores = motion_dict[slide_name][patch_name]
                (start_layer, end_layer, clear_layer) = start_layer_dict[slide_name][patch_name]
                combined_scores = []
                for l_idx, layer in enumerate(target_layers):
                    combined_scores.append(f"{motion_scores[l_idx]};{blur_scores[l_idx]}")
                scores = ",".join(combined_scores)
                wf.write("{},{},{},{},{},{}\n".format(slide_name, patch_name, scores, start_layer, end_layer, clear_layer))
