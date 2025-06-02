import os

import numpy as np
from PIL import Image
import pandas as pd

def from_number_to_layer(layer_number):
    """
    Convert a layer number to a string with leading zeros.

    Args:
    - layer_number (int): The layer number to convert.

    Returns:
    - str: The layer number as a string with leading zeros.
    """
    return str(layer_number).zfill(2)

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


if __name__ == "__main__":
    root_path = "/ssd2/AMC_zstack_2_patches/pngs_mid"
    pred_path = "/ssd2/AMC_zstack_2_patches/output_for_metrics/Pred/output0526_100000"
    output_gif_path = "/ssd2/AMC_zstack_2_patches/output0526/pairs/"
    os.makedirs(output_gif_path, exist_ok=True)
    info_sets = []
    info_sets.append({
        "patch_name": "patch_4138_15580_45977",
        "silde": '24S 074983;D;8;;FA0824;1_241225_093459',
        "start_layer": 0,
    })
    info_sets.append({
        "patch_name": "patch_286_9553_41896",
        "silde": '24S 059505;E;5;;FA0824;1_241226_011806',
        "start_layer": 10,
    })

    info_sets.append({
        "patch_name": "patch_8292_19201_39940",
        "silde": '24S 071781;E;7;;FA0824;1_241225_064219',
        "start_layer": 4,
    })

    info_sets.append({
        "patch_name": "patch_3606_14819_13303",
        "silde": '24S 048905;E;7;;FA0824;1_241226_155450',
        "start_layer": 8,
    })

    for info in info_sets:
        patch_name = info["patch_name"]
        slide = info["silde"]
        start_layer = info["start_layer"]

        frames = []
        for layer in range(start_layer, start_layer + 19, 2):
            frame_path = f"{root_path}/{slide}/z{from_number_to_layer(layer)}/{patch_name}.png"
            print(frame_path)
            frame = Image.open(frame_path)
            frames.append(frame)

        export_to_gif(frames, output_gif_path + f"/{patch_name}_GT.gif", fps=10)

        pred_frames = []
        pred_folder_path = os.path.join(pred_path, slide, patch_name)
        for frame in os.listdir(pred_folder_path):
            frame_path = f"{pred_folder_path}/{frame}"
            print(frame_path)
            frame = Image.open(frame_path)
            pred_frames.append(frame)
        export_to_gif(pred_frames, output_gif_path + f"/{patch_name}_Pred.gif", fps=10)


