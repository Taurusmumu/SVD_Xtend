import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import re


def get_wsi_dimensions(patch_paths):
    """
    Scans all patch filenames to determine the full WSI dimensions.
    Args:
        patch_paths (list): A list of paths to the patch files.
    Returns:
        tuple: (max_x, max_y, patch_width, patch_height) at full resolution.
    """
    max_x, max_y = 0, 0
    patch_w, patch_h = 0, 0

    if not patch_paths:
        return 0, 0, 0, 0

    # Get patch dimensions from the first valid patch
    first_patch = cv2.imread(patch_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_patch is None:
        raise IOError(f"Could not read the first patch to determine dimensions: {patch_paths[0]}")
    patch_h, patch_w = first_patch.shape

    # Regex to find coordinates like '12345_67890.png'
    coord_pattern = re.compile(r'(\d+)_(\d+)\.png$')

    for path in patch_paths:
        match = coord_pattern.search(os.path.basename(path))
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    # The full dimension is the coordinate of the last patch + its size
    full_width = max_x + patch_w
    full_height = max_y + patch_h

    return full_width, full_height, patch_w, patch_h


def get_motion_score(patch1_path, patch2_path):
    """
    Calculates a single motion score for a pair of patches.
    The score is the mean magnitude of the optical flow vectors.
    Args:
        patch1_path (str): Path to the patch from the first layer.
        patch2_path (str): Path to the corresponding patch from the second layer.
    Returns:
        float: The mean motion score, or None if an error occurs.
    """
    try:
        prvs = cv2.imread(patch1_path, cv2.IMREAD_GRAYSCALE)
        next_frame = cv2.imread(patch2_path, cv2.IMREAD_GRAYSCALE)

        if prvs is None or next_frame is None:
            # This can happen if a patch is missing in one of the layers
            return None

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(cv2.imread(patch1_path))
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the HSV image back to BGR for display
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return bgr

    except Exception as e:
        print(f"Error processing patches {patch1_path} and {patch2_path}: {e}")
        return None


def reconstruct_motion_map_for_slide(slide_name, root_dir, target_layers, downsample_factor=16):
    """
    Main function to reconstruct WSI-level motion maps for a single slide.
    """
    print(f"\nProcessing Slide: {slide_name}")

    # Iterate through pairs of layers
    for i in range(len(target_layers) - 1):
        layer1_name = target_layers[i]
        layer2_name = target_layers[i + 1]

        print(f"  Comparing {layer1_name} -> {layer2_name}")

        layer1_path = os.path.join(root_dir, slide_name, layer1_name)
        layer2_path = os.path.join(root_dir, slide_name, layer2_name)

        if not os.path.isdir(layer1_path):
            print(f"    Warning: Directory not found, skipping: {layer1_path}")
            continue

        # Get all patch paths from the first layer to define the scope
        patch_paths_layer1 = glob.glob(os.path.join(layer1_path, "*.png"))

        # 1. Determine the full WSI dimensions from the patch coordinates
        wsi_w, wsi_h, patch_w, patch_h = get_wsi_dimensions(patch_paths_layer1)

        if wsi_w == 0 or wsi_h == 0:
            print("    Could not determine WSI dimensions. Skipping.")
            continue

        # 2. Create a new large canvas for the downsampled score map
        canvas_w = wsi_w // downsample_factor
        canvas_h = wsi_h // downsample_factor

        # Use a float canvas to store precise scores before visualization
        motion_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

        # Regex to extract coordinates
        coord_pattern = re.compile(r'(\d+)_(\d+)\.png$')

        # 3. Iterate through patches, calculate score, and fill canvas
        for patch1_path in tqdm(patch_paths_layer1, desc=f"    Processing {layer1_name}"):
            patch_basename = os.path.basename(patch1_path)
            patch2_path = os.path.join(layer2_path, patch_basename)

            score = get_motion_score(patch1_path, patch2_path)

            if score is not None:
                # Get patch coordinates from filename
                match = coord_pattern.search(patch_basename)
                if match:
                    x = int(match.group(1))
                    y = int(match.group(2))

                    # Calculate position on the downsampled canvas
                    canvas_x = x // downsample_factor
                    canvas_y = y // downsample_factor

                    # Calculate the size of the patch on the canvas
                    patch_canvas_w = patch_w // downsample_factor
                    patch_canvas_h = patch_h // downsample_factor

                    # Fill the corresponding rectangle on the canvas with the score
                    # Ensure the region does not go out of bounds
                    end_y = min(canvas_y + patch_canvas_h, canvas_h)
                    end_x = min(canvas_x + patch_canvas_w, canvas_w)
                    motion_canvas[canvas_y:end_y, canvas_x:end_x] = \
                        cv2.resize(score, (patch_w // downsample_factor, patch_h // downsample_factor))

        normalized_canvas = motion_canvas
        # # 4. Normalize and visualize the completed canvas
        # # Avoid division by zero if the canvas is empty or has no motion
        # max_score = np.max(motion_canvas)
        # if max_score > 0:
        #     # Normalize to 0-255 range for visualization
        #     normalized_canvas = (motion_canvas / max_score * 255).astype(np.uint8)
        # else:
        #     normalized_canvas = np.zeros_like(motion_canvas, dtype=np.uint8)

        # Apply a colormap for better visualization (JET or INFERNO are good choices)
        # heatmap_viz = cv2.applyColorMap(normalized_canvas, cv2.COLORMAP_JET)

        # 5. Save the resulting WSI-level score map
        output_dir = os.path.join("./wsi_motion_maps", slide_name)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"motion_map_{layer1_name}_to_{layer2_name}.png"
        cv2.imwrite(os.path.join(output_dir, output_filename), motion_canvas)

    print(f"Finished processing for slide {slide_name}.")


if __name__ == '__main__':
    # --- Configuration ---
    # The root directory containing folders for each slide
    ROOT_PATCH_DIR = "/ssd2/AMC_zstack_2_patches/pngs_mid"

    # List of slide names you want to process
    SLIDE_NAMES = ["24S 048630;E;10;;FA0824;1_241226_161645", "24S 048905;E;7;;FA0824;1_241226_155450"]  # <-- TODO: CHANGE THIS TO YOUR SLIDE FOLDER NAMES

    # The names of your z-stack layer folders
    TARGET_LAYERS = [f"z{i:02d}" for i in range(19)]

    # How much smaller the output WSI map should be
    DOWNSAMPLE_FACTOR = 16

    # --- Run the reconstruction process for each slide ---
    for slide_name in SLIDE_NAMES:
        reconstruct_motion_map_for_slide(slide_name, ROOT_PATCH_DIR, TARGET_LAYERS, DOWNSAMPLE_FACTOR)

