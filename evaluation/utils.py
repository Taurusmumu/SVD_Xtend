import yaml
import json
import os
from PIL import Image
import PIL
import numpy as np
import pandas as pd
import torch


def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results, config):
    """Saves the final evaluation results to a JSON file."""
    results_path = os.path.join(config["output_path"], config["pred_folder"], config["result_log_path"])
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_path}")


def save_video_as_gif(video_tensor, output_path, filename):
    """
    Saves a video tensor as a GIF.
    Args:
        video_tensor (torch.Tensor): Video tensor of shape (F, C, H, W) and range [-1, 1].
        output_path (str): The directory to save the GIF in.
        filename (str): The base name for the file.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # De-normalize from [-1, 1] to [0, 255] and convert to numpy
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
    video_np = (video_np + 1) / 2.0 * 255.0
    video_np = video_np.astype(np.uint8)

    frames = [Image.fromarray(frame) for frame in video_np]

    full_path = os.path.join(output_path, f"{filename}_recon.gif")
    frames[0].save(
        full_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )

def sample_gt(config):
    video_sampled_path = os.path.join(config['output_path'], config['video_sampled_path'])
    if os.path.isfile(video_sampled_path) is False:
        split_data = pd.read_csv(config['split_file_path'])
        split_data_train = split_data.loc[split_data["train"] == "train"]
        split_data_train_slides = list(split_data_train.iloc[:, 0])

        df = pd.read_csv(config['blur_motion_path'])
        df_train = df.loc[df["slide_name"].isin(split_data_train_slides)]
        start_layers = np.array(df_train["start_indices"])
        end_layers = np.array(df_train["end_indices"])
        valid_index = np.where((start_layers >= 0) & (end_layers <= len(config['layers']) - 1))[0]
        df_valid = df_train.iloc[valid_index]
        df_sample = df_valid.sample(n=config['sample_num'], random_state=42)
        df_sample.to_csv(video_sampled_path, index=False)


def load_videos_from_folder(dir_path):
    videos = []

    for slide in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, slide)):
            continue
        slide_path = os.path.join(dir_path, slide)

        for patch in os.listdir(slide_path):
            patch_path = os.path.join(slide_path, patch)
            frames = sorted(os.listdir(patch_path))
            frame_paths = [os.path.join(patch_path, frame) for frame in frames]
            frame_np = [np.array(Image.open(frame_path).convert('RGB')) for frame_path in frame_paths]
            videos.append(np.stack(frame_np))

    return np.stack(videos)


def load_gt_videos(data_root_path, slide_name, patch_name, frames):
    videos = []
    for frame in frames:
        file_path = os.path.join(data_root_path, slide_name, frame, patch_name)
        image = Image.open(file_path).convert("RGB")
        videos.append(image)
    return videos



def load_image(
    image
) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
        convert_method (Callable[[PIL.Image.Image], PIL.Image.Image], optional):
            A conversion method to apply to the image after loading it.
            When set to `None` the image will be converted "RGB".

    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
        )

    image = PIL.ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    return image


def convert_to_tensor(no_array):
    array = torch.from_numpy(no_array)
    array = array.permute(0, 1, 4, 2, 3)
    array = array.float() / 255.0  # Normalize to [0, 1]
    return array

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