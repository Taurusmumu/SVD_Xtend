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
import matplotlib.pyplot as plt
from skimage.exposure import is_low_contrast


def estimate_motion(clear_layer, slide_name, patch_name, target_layers, motion_dict, root_dir, motion_dict1):
    try:
        frame1_path = os.path.join(root_dir, slide_name, clear_layer, patch_name)
        frame1 = cv2.imread(frame1_path)
        if frame1 is None or is_low_contrast(frame1):
            print(f"Error: Could not read reference frame: {frame1_path}. Skipping patch.")
            motion_dict[slide_name][patch_name] = list(np.full(len(target_layers), -1.0))
            motion_dict1[slide_name][patch_name] = list(np.full(len(target_layers), -1.0))
            return

        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        mag_list = []
        mag_list1 = []
        # print(f"Aligning all layers to reference: {clear_layer}/{patch_name}")

        for l_idx, next_layer in enumerate(target_layers):
            frame2_path = os.path.join(root_dir, slide_name, next_layer, patch_name)
            frame2 = cv2.imread(frame2_path)
            if frame2 is None:
                print(f"  - Warning: Missing frame for {slide_name} layer {next_layer}/{patch_name}. Appending score -1.")
                mag_list.append(-1)
                mag_list1.append(-1)
                continue

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
                print(f"  - Warning: ECC registration failed for {slide_name} layer {next_layer}/{patch_name}. Using identity matrix.")

            # --- 4. Apply the matrix to warp the COLOR image ---
            h, w = next_frame_gray.shape
            ### MODIFIED: We apply the transform to the original color frame.
            aligned_color_frame = cv2.warpAffine(frame2, warp_matrix, (w, h), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
            # aligned_color_frame = cv2.cvtColor(aligned_color_frame, cv2.COLOR_BGR2RGB)
            aligned_gray_frame = cv2.cvtColor(aligned_color_frame, cv2.COLOR_BGR2GRAY)
            # --- CORE LOGIC (UNCHANGED) ---
            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(prvs, aligned_gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag[np.where(mag == np.inf)] = 0
            mag_list1.append(float(np.mean(mag)))

            output_dir = os.path.join(f"/ssd2/AMC_zstack_2_patches_warp/pngs_mid3", slide_name, next_layer)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, patch_name)
            # print(output_path)
            cv2.imwrite(output_path, aligned_color_frame)
            # skimage.io.imsave(output_path, aligned_color_frame)

        motion_dict[slide_name][patch_name] = mag_list
        motion_dict1[slide_name][patch_name] = mag_list1

    except Exception as e:
        print(f"Error processing {slide_name} layer {next_layer}/{patch_name}: {e}")
        motion_dict[slide_name][patch_name] = list(np.full(len(target_layers), -1.0))
        motion_dict1[slide_name][patch_name] = list(np.full(len(target_layers), -1.0))


# Assume estimate_motion is defined elsewhere, for example:
# from your_motion_module import estimate_motion

def process_row(row, target_layers, root_dir):
    """
    Worker function to process a single row of data from the DataFrame.
    This function will be executed in a separate process.
    """
    # Extract data from the row
    slide_name = row['slide_name']
    patch_name = row['patch_name']
    start_layer = row['start_indices']
    end_layer = row['end_indices']
    clear_layer = row['min_indices']

    # The list of scores is now a single string 'v1 v2 v3...'. We need to parse it.
    blur_scores_str = row['blur_scores']
    blur_scores = [float(score) for score in blur_scores_str.split(' ')]

    # Initialize dictionaries for the motion results for this single row
    motion_dict_result = {}
    motion_dict1_result = {}
    if slide_name not in motion_dict_result:
        motion_dict_result[slide_name] = {}
        motion_dict1_result[slide_name] = {}

    # This is the slow part that will now run in parallel
    estimate_motion(target_layers[clear_layer], slide_name, patch_name, target_layers,
                    motion_dict_result, root_dir, motion_dict1_result)

    # Return all the processed data needed to build the main dictionaries
    return (slide_name, patch_name, start_layer, end_layer, clear_layer,
            blur_scores, motion_dict_result, motion_dict1_result)


if __name__ == "__main__":
    # Use if __name__ == "__main__": to protect your code when using multiprocessing

    # --- Configuration ---
    anno_path = "./blur_data5.csv"
    root_dir = "/ssd2/AMC_zstack_2_patches/pngs_mid"
    target_layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09",
                     "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]

    # --- 1. Fast Reading with Pandas ---
    print("Reading CSV file with pandas...")
    # We need to handle the space-separated list of scores. We'll read them as one string first.
    # Let's define the column names based on your original code's parsing logic.
    # Assuming the format is: slide,patch,score1,score2,...,scoreN,start,end,clear
    # We will read all scores into a single column to simplify.

    # A more robust way is to read the CSV and combine score columns
    df = pd.read_csv(anno_path)

    # Assuming column names are like 'slide_name', 'patch_name', 'score_1', ..., 'start_layer', etc.
    # We'll create a single column of space-separated scores.
    # Identify score columns (assuming they are between patch_name and start_layer)
    score_cols = df.columns[2:-3]
    df['blur_scores'] = df[score_cols].astype(str).agg(' '.join, axis=1)

    # Keep only the columns we need for processing
    df_proc = df[['slide_name', 'patch_name', 'start_indices', 'end_indices', 'min_indices', 'blur_scores']]

    # --- 2. Parallel Processing ---
    # Determine the number of processes to use
    n_jobs = mp.cpu_count()  # Use all available CPU cores
    print(f"Starting parallel processing with {n_jobs} cores...")

    # `partial` is a helper to "pre-fill" the arguments of our worker function
    # that are the same for every row.
    worker_func = partial(process_row, target_layers=target_layers, root_dir=root_dir)

    # Create the pool of worker processes
    with mp.Pool(processes=n_jobs) as pool:
        # `imap_unordered` is memory-efficient and gives results as they are completed.
        # `tqdm` provides a progress bar.
        results = list(tqdm(pool.imap_unordered(worker_func, [row for _, row in df_proc.iterrows()]),
                            total=len(df_proc),
                            desc="Estimating motion"))

    # --- 3. Aggregate Results ---
    print("Aggregating results into dictionaries...")
    start_layer_dict = {}
    blur_degree_dict = {}
    motion_dict = {}
    motion_dict1 = {}

    for result in tqdm(results, desc="Building dictionaries"):
        (slide_name, patch_name, start_layer, end_layer, clear_layer,
         blur_scores, motion_res, motion1_res) = result

        # Initialize nested dictionaries if the slide_name is new
        if slide_name not in motion_dict:
            start_layer_dict[slide_name] = {}
            blur_degree_dict[slide_name] = {}
            motion_dict[slide_name] = {}
            motion_dict1[slide_name] = {}

        # Populate the dictionaries
        start_layer_dict[slide_name][patch_name] = (start_layer, end_layer, clear_layer)
        blur_degree_dict[slide_name][patch_name] = blur_scores

        # The worker function returns dictionaries with the motion results.
        # We need to merge them into our main dictionaries.
        if slide_name in motion_res:
            motion_dict[slide_name].update(motion_res[slide_name])
        if slide_name in motion1_res:
            motion_dict1[slide_name].update(motion1_res[slide_name])

    print("Processing complete.")

    with open("./blur_motion_data_0913.csv", "w") as wf:
        wf.write("slide_name,patch_name,{},start_indices,end_indices,min_indices\n".format(",".join(target_layers)))
        for slide_name, slide_data in blur_degree_dict.items():
            for patch_name, blur_scores in slide_data.items():
                motion_scores = motion_dict[slide_name][patch_name]
                (start_layer, end_layer, clear_layer) = start_layer_dict[slide_name][patch_name]
                combined_scores = []
                for l_idx, layer in enumerate(target_layers):
                    combined_scores.append(f"{motion_scores[l_idx]};{blur_scores[l_idx]}")
                scores = ",".join(combined_scores)
                wf.write("{},{},{}\n".format(slide_name, patch_name, scores, start_layer, end_layer, clear_layer))

    with open("./blur_motion_data1_0913.csv", "w") as wf:
        wf.write("slide_name,patch_name,{},start_indices,end_indices,min_indices\n".format(",".join(target_layers)))
        for slide_name, slide_data in blur_degree_dict.items():
            for patch_name, blur_scores in slide_data.items():
                motion_scores = motion_dict1[slide_name][patch_name]
                (start_layer, end_layer, clear_layer) = start_layer_dict[slide_name][patch_name]
                combined_scores = []
                for l_idx, layer in enumerate(target_layers):
                    combined_scores.append(f"{motion_scores[l_idx]};{blur_scores[l_idx]}")
                scores = ",".join(combined_scores)
                wf.write("{},{},{}\n".format(slide_name, patch_name, scores, start_layer, end_layer, clear_layer))

    data_hist = []
    data_hist1 = []
    bool_data = []
    for slide_name, slide_data in blur_degree_dict.items():
        for patch_name, blur_scores in slide_data.items():
            motion_scores = motion_dict[slide_name][patch_name]
            motion_scores1 = motion_dict1[slide_name][patch_name]
            (start_layer, end_layer, clear_layer) = start_layer_dict[slide_name][patch_name]
            l0 = []
            l1 = []
            b = []
            for idx in range(start_layer, end_layer + 1):
                if idx < 0 or idx >= len(target_layers) or motion_scores[idx] > 10 or motion_scores1[idx] > 10:
                    b.append(False)
                    l0.append(0)
                    l1.append(0)
                else:
                    b.append(True)
                    l0.append(motion_scores[idx])
                    l1.append(motion_scores1[idx])
            data_hist.append(l0)
            data_hist1.append(l1)
            bool_data.append(b)
    data_hist = np.array(data_hist)
    data_hist1 = np.array(data_hist1)
    bool_data = np.array(bool_data)

    result = []
    for i in range(data_hist.shape[1]):
        result.append(np.mean(data_hist[:, i][bool_data[:, i]]))

    plt.figure(figsize=(8, 5))
    plt.bar([-5,-4,-3,-2,-1,0,1,2,3,4,5], result)
    plt.title('before wapring')
    plt.xlabel('Layer')
    plt.ylabel('Avg Motion Score')
    plt.show()

    result1 = []
    for i in range(data_hist.shape[1]):
        result1.append(np.mean(data_hist1[:, i][bool_data[:, i]]))

    plt.figure(figsize=(8, 5))
    plt.bar([-5,-4,-3,-2,-1,0,1,2,3,4,5], result1)
    plt.title('After wapring')
    plt.xlabel('Layer')
    plt.ylabel('Avg Motion Score')
    plt.show()