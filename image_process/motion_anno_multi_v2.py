import numpy as np
import cv2
import os
from tqdm import tqdm
import multiprocessing as mp


def estimate_motion(clear_layer, slide_name, patch_name, target_layers, root_dir):
    """
    Calculates the optical flow between a reference frame (clear_layer) and all other target_layers.
    This function remains unchanged from the original script.
    """
    try:
        frame1_path = os.path.join(root_dir, slide_name, clear_layer, patch_name)
        frame1 = cv2.imread(frame1_path)
        if frame1 is None:
            # Suppress print statements in multiprocessing to avoid cluttered output
            # print(f"Error: Could not read reference frame: {frame1_path}. Skipping patch.")
            return None

        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        mag_list = []

        for next_layer in target_layers:
            frame2_path = os.path.join(root_dir, slide_name, next_layer, patch_name)
            frame2 = cv2.imread(frame2_path)
            if frame2 is None:
                # print(f"  - Warning: Missing frame for layer {next_layer}. Appending score -1.")
                mag_list.append(-1)
                continue

            next_frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame_gray, None, 0.5, 3, 8, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag[np.where(mag == np.inf)] = 0
            mag_list.append(float(np.mean(mag)))

        return mag_list

    except Exception as e:
        # print(f"Error processing {slide_name} {patch_name}: {e}")
        return None


def process_line(args):
    """
    Worker function for a single process. It parses a line from the input CSV,
    calls the motion estimation function, and returns the combined results.
    """
    # Unpack arguments
    line, target_layers, root_dir = args

    # Parse data from the line
    line_split = line.strip().split(",")
    slide_name = line_split[0]
    patch_name = line_split[1]
    start_layer = int(line_split[-3])
    end_layer = int(line_split[-2])
    clear_layer = int(line_split[-1])

    # Calculate motion scores
    motion_scores = estimate_motion(target_layers[clear_layer], slide_name, patch_name, target_layers, root_dir)

    # If motion estimation was successful, prepare the result for writing
    if motion_scores is not None:
        motion_blur_list = [score for score in line_split[2:-3]]
        blur_scores = [float(score.split(';')[1]) for score in motion_blur_list]

        # Combine motion and blur scores into the required string format
        combined_scores_str = ",".join([f"{motion_scores[i]};{blur_scores[i]}" for i in range(len(target_layers))])

        # Return a formatted string ready to be written to the output file
        return f"{slide_name},{patch_name},{combined_scores_str},{start_layer},{end_layer},{clear_layer}\n"

    return None


if __name__ == '__main__':
    # --- Configuration ---
    anno_path = "./blur_motion_data3.csv"
    output_path = "./blur_motion_data4_mp.csv"  # Changed output filename
    root_dir = "/ssd2/AMC_zstack_2_patches_warp/pngs_mid"
    target_layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09",
                     "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]

    # Use most of the available CPU cores, leaving one free for system stability
    n_jobs = max(1, mp.cpu_count() - 1)

    # --- Task Preparation ---
    print("Preparing tasks...")
    with open(anno_path, "r") as rf:
        header = rf.readline()
        lines = rf.readlines()

    # Create a list of arguments for the worker function
    # Each item in the list is a tuple containing all the info needed for one line
    tasks = [(line, target_layers, root_dir) for line in lines]

    # To test with a smaller subset, uncomment the following line:
    # tasks = tasks[:500]

    print(f"Starting processing for {len(tasks)} patches using {n_jobs} processes...")

    # --- Multiprocessing Execution ---
    with open(output_path, "w") as wf:
        # Write the header to the new CSV file
        wf.write("slide_name,patch_name,{},start_indices,end_indices,min_indices\n".format(",".join(target_layers)))

        # Create a pool of worker processes
        with mp.Pool(processes=n_jobs) as pool:
            # Use imap_unordered to process tasks in parallel and get results as they complete
            # Wrap with tqdm to create a progress bar
            for result_line in tqdm(pool.imap_unordered(process_line, tasks), total=len(tasks)):
                # If the worker function returned a valid result (not None), write it to the file
                if result_line:
                    wf.write(result_line)

    print(f"Processing complete. Results saved to {output_path}")