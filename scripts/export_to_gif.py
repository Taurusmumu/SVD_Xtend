import os

import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

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
    pred_path = "/ssd2/AMC_zstack_2_patches/output_for_metrics/output0623/Pred_random/200000/"
    output_gif_path = "/ssd2/AMC_zstack_2_patches/output0623/pairs/"
    num_frame = 11
    os.makedirs(output_gif_path, exist_ok=True)
    info_sets = []
    # info_sets.append({
    #     "patch_name": "patch_6446_23008_34514",
    #     "silde": '24S 053791;A;12;;FA0824;1_241226_081326',
    #     "start_layer": 0,
    # })
    # info_sets.append({
    #     "patch_name": "patch_9993_29408_33490",
    #     "silde": '24S 053791;A;12;;FA0824;1_241226_081326',
    #     "start_layer": 0,
    # })
    #
    # info_sets.append({
    #     "patch_name": "patch_6507_20844_15426",
    #     "silde": '24S 075098;A;8;;FA0824;1_241225_103552',
    #     "start_layer": 0,
    # })
    #
    # info_sets.append({
    #     "patch_name": "patch_6570_17725_29207",
    #     "silde": '24S 064735;E;11;;FA0824;1_241224_190437',
    #     "start_layer": 5,
    # })

    info_sets.append({
        "patch_name": "patch_6114_17494_46381",
        "silde": '24S 048630;E;10;;FA0824;1_241226_161645',
        "start_layer": 0,
    })

    for info in info_sets:
        patch_name = info["patch_name"]
        slide = info["silde"]
        start_layer = info["start_layer"]

        frames = []
        index = 0
        for layer in range(start_layer, start_layer + num_frame, 1):
            frame_path = f"{root_path}/{slide}/z{from_number_to_layer(layer)}/{patch_name}.png"
            print(frame_path)
            frame = Image.open(frame_path)
            draw = ImageDraw.Draw(frame)
            draw.text((10, 10), "Layer {}".format(index + 1), (255, 255, 255), font=ImageFont.load_default())
            frames.append(frame)
            index += 1

        export_to_gif(frames, output_gif_path + f"/{patch_name}_GT.gif", fps=10)

        pred_frames = []
        pred_folder_path = os.path.join(pred_path, slide, patch_name)
        index = 0
        for frame in range(start_layer, start_layer + num_frame, 1):
            frame_path = f"{pred_folder_path}/z{from_number_to_layer(frame)}.png"
            print(frame_path)
            frame = Image.open(frame_path)
            pred_frames.append(frame)
        export_to_gif(pred_frames, output_gif_path + f"/{patch_name}_Pred.gif", fps=10)


