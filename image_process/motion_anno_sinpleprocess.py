import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import skimage.io
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


def calc_score(layer_dir, next_layer_dir, patch_name):
    try:
        frame1_path = os.path.join(layer_dir, patch_name)
        frame2_path = os.path.join(next_layer_dir, patch_name)
        frame1 = cv2.imread(frame1_path)

        # Convert the first frame to grayscale
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Create an HSV image for visualization, same as the original script
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255  # Set saturation to maximum

        # --- MODIFICATION: Loop through the rest of the image paths ---
        # We start from the second image (index 1)
        # Read the next frame in the sequence
        frame2 = cv2.imread(frame2_path)
        # Convert to grayscale
        next_frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # --- CORE LOGIC (UNCHANGED) ---
        # Calculate dense optical flow using Farneback method

        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # global_dx = np.median(flow[..., 0])
        # global_dy = np.median(flow[..., 1])
        # flow[..., 0] -= global_dx
        # flow[..., 1] -= global_dy

        # Convert flow vectors from cartesian (dx, dy) to polar (magnitude, angle)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Map angle to Hue and magnitude to Value
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the HSV image back to BGR for display
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # --- DISPLAY (SLIGHTLY MODIFIED) ---
        # Show the optical flow visualization
        skimage.io.imsave(f'{frame1_path.split("/")[-2]}_original.png', frame1)
        skimage.io.imsave(f'{frame1_path.split("/")[-2]}_optical_flow_hsv.png', bgr)

    except Exception as e:
        print(f"Error processing {patch_name}: {e}")
        score2 = -1
    return patch_name

def estimate_motion(clear_layer, slide_name, patch_name, target_layers, motion_dict, root_dir, motion_dict1):
    try:
        frame1_path = os.path.join(root_dir, slide_name, clear_layer, patch_name)
        frame1 = cv2.imread(frame1_path)
        if frame1 is None:
            print(f"Error: Could not read reference frame: {frame1_path}. Skipping patch.")
            return [-1] * len(target_layers)  # Return list of zeros on error

        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        mag_list = []
        mag_list1 = []
        # print("Clear layer is: ", clear_layer)
        # print(f"Aligning all layers to reference: {clear_layer}/{patch_name}")

        for l_idx, next_layer in enumerate(target_layers):
            frame2_path = os.path.join(root_dir, slide_name, next_layer, patch_name)
            # Convert the first frame to grayscale
            # try:
            frame2 = cv2.imread(frame2_path)
            if frame2 is None:
                print(f"  - Warning: Missing frame for layer {next_layer}. Appending score -1.")
                mag_list.append(-1)
                continue
            # Convert to grayscale
            # Create a grayscale version specifically for registration
            next_frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag[np.where(mag == np.inf)] = 0
            mag_list.append(float(np.mean(mag)))

            # --- 3. Calculate the transformation matrix using grayscale images ---
            warp_mode = cv2.MOTION_TRANSLATION
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                # Calculate the transform that aligns next_frame_gray with prvs (the clear layer)
                (cc, warp_matrix) = cv2.findTransformECC(prvs, next_frame_gray, warp_matrix, warp_mode, criteria)
            except cv2.error:
                # If registration fails, the matrix remains an identity matrix (no shift)
                print(f"  - Warning: ECC registration failed for {next_layer}. Using identity matrix.")

            # --- 4. Apply the matrix to warp the COLOR image ---
            h, w = next_frame_gray.shape
            ### MODIFIED: We apply the transform to the original color frame.
            aligned_color_frame = cv2.warpAffine(frame2, warp_matrix, (w, h),
                                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            # --- CORE LOGIC (UNCHANGED) ---
            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag[np.where(mag == np.inf)] = 0
            mag_list1.append(float(np.mean(mag)))
            # except:
            #     mag_list.append(-1)
            # global_dx = np.median(flow[..., 0])
            # global_dy = np.median(flow[..., 1])
            # flow[..., 0] -= global_dx
            # flow[..., 1] -= global_dy
            # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            # mag[np.where(mag == np.inf)] = 0
            # mag_list1.append(float(np.mean(mag)))

            # Convert flow vectors from cartesian (dx, dy) to polar (magnitude, angle)

            # # Map angle to Hue and magnitude to Value
            # Create an HSV image for visualization, same as the original script
            # hsv = np.zeros_like(frame1)
            # hsv[..., 1] = 255  # Set saturation to maximum
            # hsv[..., 0] = ang * 180 / np.pi / 2
            # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            #
            # # Convert the HSV image back to BGR for display
            # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            # --- DISPLAY (SLIGHTLY MODIFIED) ---
            # Show the optical flow visualization
            # skimage.io.imsave(f'{frame1_path.split("/")[-2]}_original.png', frame1)
            os.makedirs(f'/ssd2/AMC_zstack_2_patches/pngs_mid/{slide_name}/{next_layer}/', exist_ok=True)
            skimage.io.imsave(f'/ssd2/AMC_zstack_2_patches/pngs_mid/{slide_name}/{next_layer}/{patch_name}', aligned_color_frame)

        motion_dict[slide_name][patch_name] = mag_list
        motion_dict1[slide_name][patch_name] = mag_list1

    except Exception as e:
        print(f"Error processing {slide_name} {patch_name}: {e}")
        motion_dict[slide_name][patch_name] = list(np.full(len(target_layers), -1.0))
        motion_dict1[slide_name][patch_name] = list(np.full(len(target_layers), -1.0))


if __name__ == '__main__':
    # --- MODIFICATION: Specify image path and file type ---
    # IMPORTANT: Replace 'path/to/your/images' with the actual folder path.
    # Replace '*.png' with your file extension if it's different (e.g., '*.jpg', '*.tif').
    anno_path = "./blur_data5.csv"
    root_dir = "/ssd2/AMC_zstack_2_patches/pngs_mid"
    text_file_path = "/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt"
    target_layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09",
                     "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]

    # n_jobs = mp.cpu_count() * 80 //
    blur_degree_dict = {}
    start_layer_dict = {}
    motion_dict = {}
    motion_dict1 = {}
    with open(anno_path, "r") as rf:
        rf.readline()
        for line in tqdm(rf.readlines(), desc="Processing data"):
            line_split = line.strip().split(",")
            slide_name = line_split[0]
            patch_name = line_split[1]
            start_layer = int(line_split[-3])
            end_layer = int(line_split[-2])
            clear_layer = int(line_split[-1])

            if slide_name not in motion_dict:
                start_layer_dict[slide_name] = {}
                blur_degree_dict[slide_name] = {}
                motion_dict[slide_name] = {}
                motion_dict1[slide_name] = {}
            start_layer_dict[slide_name][patch_name] = (start_layer, end_layer, clear_layer)
            blur_degree_dict[slide_name][patch_name] = [float(score) for score in line_split[2:-3]]
            estimate_motion(target_layers[clear_layer], slide_name, patch_name, target_layers, motion_dict, root_dir, motion_dict1)

    print()
