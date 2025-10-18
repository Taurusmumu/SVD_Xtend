import numpy as np
import cv2
import os
from tqdm import tqdm
import skimage.io
import multiprocessing as mp
import PIL.Image
from skimage.exposure import is_low_contrast
import csv


# --- Placeholder for your motion calculation logic ---
def calculate_motion(volume_3d):
    """
    Calculates motion scores for a 3D image volume.

    Args:
        volume_3d (np.array): A 3D numpy array of shape (frames, H, W, C).

    Returns:
        list: A list of motion scores, one for each frame transition.
              For this demo, it returns a list of random floats.
    """
    mid_frame = volume_3d[volume_3d.shape[0] // 2]
    prvs = cv2.cvtColor(mid_frame, cv2.COLOR_RGB2GRAY)
    mag_list = []

    for i in range(len(volume_3d)):
        frame2 = volume_3d[i]
        next_frame_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        mag_total = np.zeros_like(next_frame_gray)
        for j in range(0, 256, 8):
            flow = cv2.calcOpticalFlowFarneback(prvs[:, j: j + 8], next_frame_gray[:, j: j + 8], None, 0.5, 3, 8, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag[np.where(mag == np.inf)] = 0
            mag_total[:, j: j + 8] = mag

        mag_list.append(float(np.mean(mag_total)))

    mag_list = list(np.array(mag_list) - mag_list[volume_3d.shape[0] // 2])
    return mag_list


# This function is the "worker" that will be run on each core.
def flip_img_generate(args):
    """
    Processes a single image patch, calculates motion, and returns the results.
    """
    root_dir, out_dir, slide_name, patch_name, target_layers, start_layer, slice_bef, slice_aft = args

    try:
        input_3D = []
        if np.random.rand() > 0.5:
            target_size = (470, 256)  # (width, height)
        else:
            target_size = (256, 470)  # (width, height)

        for layer in target_layers[start_layer: start_layer + slice_bef]:
            img_path = os.path.join(root_dir, slide_name, layer, patch_name)
            try:
                img = PIL.Image.open(img_path).convert("RGB")
                img = np.array(img)
            except (PIL.UnidentifiedImageError, FileNotFoundError):
                return None  # Skip this job if a file is missing/corrupt

            if is_low_contrast(img, 0.2):
                return None  # Skip if low contrast

            resized_slice = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            input_3D.append(resized_slice)

        if len(input_3D) != slice_bef:
            return None  # Skip if not enough valid frames

        input_3D = np.stack(input_3D, axis=0)
        output_3D = np.zeros((11, 256, int(256 / slice_aft * slice_bef), 3), dtype=np.uint8)
        output_3D_resized = np.zeros((11, 256, 256, 3), dtype=np.uint8)

        img_h, img_w = input_3D.shape[1], input_3D.shape[2]

        if img_h > img_w:  # img_w should be the longer side
            input_3D = np.transpose(input_3D, (0, 2, 1, 3))
            img_w, img_h = img_h, img_w

        fill_idx = 0
        for i in range(0, img_w - 11, 11):
            slice_3d = input_3D[:, :, i: i + 11, :]

            if is_low_contrast(slice_3d, 0.2):
                continue

            transposed_slice = np.transpose(slice_3d, (2, 1, 0, 3))
            # if transposed_slice.shape[1] != 256:
            #     temp_resized = np.zeros((11, 256, 16, 3), dtype=np.uint8)
            #     for j in range(transposed_slice.shape[0]):
            #         temp_resized[j] = cv2.resize(transposed_slice[j], (16, 256), interpolation=cv2.INTER_LINEAR)
            #     transposed_slice = temp_resized

            if fill_idx + slice_bef <= (256 / slice_aft * slice_bef):
                output_3D[:, :, fill_idx:fill_idx + slice_bef, :] = transposed_slice

            fill_idx += slice_bef

        for i in range(len(output_3D)):
            output_3D_resized[i] = cv2.resize(output_3D[i], (256, 256), interpolation=cv2.INTER_LINEAR)

        # if np.random.rand() > 0.5:
        #     output_3D = np.transpose(output_3D, (0, 2, 1, 3))

        out_patch_dir = os.path.join(out_dir, slide_name, patch_name.split('.')[0])
        os.makedirs(out_patch_dir, exist_ok=True)

        for i in range(len(output_3D_resized)):
            save_path = os.path.join(out_patch_dir, f'{target_layers[start_layer + i]}.png')
            skimage.io.imsave(save_path, output_3D_resized[i], check_contrast=False)

        # --- New: Calculate motion and prepare the result ---
        motion_scores = calculate_motion(output_3D)
        final_motion_scores = np.zeros_like(target_layers, dtype=float)
        final_motion_scores[start_layer:start_layer + len(motion_scores)] = motion_scores

        return (slide_name, patch_name, list(final_motion_scores))

    except Exception as e:
        # print(f"Error processing {slide_name}/{patch_name}: {e}")
        return None


if __name__ == '__main__':
    # --- Configuration ---
    anno_path = "./blur_data5.csv"
    root_dir = "/ssd2/AMC_zstack_2_patches_warp/pngs_mid"
    out_dir = "/ssd2/AMC_zstack_2_patches_flip_v2/pngs_mid"
    motion_anno_path = "motion_annotations_v2.csv"  # Output CSV file
    target_layers = ["z{:02d}".format(i) for i in range(19)]
    # FRAME_NUM = 16
    slice_bef, slice_aft = 14, 8
    N_JOBS = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1
    print(f"Using {N_JOBS} parallel processes.")

    # --- 1. Prepare all tasks in a list ---
    tasks = []
    print("Preparing tasks from annotation file...")
    with open(anno_path, "r") as rf:
        rf.readline()  # Skip header
        lines = rf.readlines()
        for line in tqdm(lines, desc="Reading annotations"):
            line_split = line.strip().split(",")
            slide_name, patch_name = line_split[0], line_split[1]

            blur_scores = [float(score) for score in line_split[2:-3]]
            min_sum = float('inf')
            min_index = 0
            for i in range(len(blur_scores) - slice_bef + 1):
                section_sum = np.sum(blur_scores[i: i + slice_bef])
                if section_sum < min_sum:
                    min_sum = section_sum
                    min_index = i

            task_args = (root_dir, out_dir, slide_name, patch_name, target_layers, min_index, slice_bef, slice_aft)
            tasks.append(task_args)

    print(f"\nCreated {len(tasks)} tasks to process.")

    # --- 2. Run tasks in parallel and collect results ---
    print("Starting image processing...")
    results = []
    with mp.Pool(processes=N_JOBS) as pool:
        # The pool returns results as they are completed.
        for result in tqdm(pool.imap_unordered(flip_img_generate, tasks), total=len(tasks), desc="Processing Patches"):
            # Only append if the worker function didn't return None (i.e., it was successful)
            if result:
                results.append(result)

    # --- 3. Write collected results to a CSV file ---
    print(f"\nWriting {len(results)} motion annotations to {motion_anno_path}...")
    with open(motion_anno_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['slide_name', 'patch_name'] + target_layers)
        # Write the data rows
        for slide_name, patch_name, scores in results:
            # Convert the list of float scores to a single semi-colon separated string
            # scores_str = ";".join(map(str, scores))
            writer.writerow([slide_name, patch_name] + scores)

    print("\n✅ All tasks completed.")

