import yaml
import json
import os
from PIL import Image
import PIL
import numpy as np
import pandas as pd
import torch
import glob


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
    video_sampled_path = config['video_sampled_path']
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

def sample_gt_prostate(config):
    img_sampled_path = config['img_sampled_path']
    if os.path.isfile(img_sampled_path) is False:
        data_root_path = config['data_root_path']
        cls_0, cls_1, cls_2, cls_3, cls_4, cls_5 = [], [], [], [], [], []
        if "agg" in config['data_root_path']:
            total_path_list = []
            for metadata_path in config['metadata_path']:
                metadata = pd.read_csv(metadata_path)
                total_path_list += list(metadata['file_path'])
            # Class 0: 0, Class 1: 6459, Class 2: 6169, Class 3: 13008, Class 4: 19527, Class 5: 2073

        if "harvard" in config['data_root_path']:
            total_path_list = glob.glob(os.path.join(data_root_path, '*', '*.jpg'))
            # Class 0: 2076, Class 1: 6303, Class 2: 4541, Class 3: 2383

        for file_name in total_path_list:
            cls = file_name.split('.')[0].split('_')[-1]
            if cls == '0':
                cls_0.append(file_name)
            elif cls == '1':
                cls_1.append(file_name)
            elif cls == '2':
                cls_2.append(file_name)
            elif cls == '3':
                cls_3.append(file_name)
            elif cls == '4':
                cls_4.append(file_name)
            elif cls == '5':
                cls_5.append(file_name)

        print(f"Class 0: {len(cls_0)}, Class 1: {len(cls_1)}, Class 2: {len(cls_2)}, Class 3: {len(cls_3)}, Class 4: {len(cls_4)}, Class 5: {len(cls_5)}")

        cls_0_sample = [] if len(cls_0) == 0 else np.random.choice(cls_0, size=int(config['sample_num'] * 0.25), replace=False)
        cls_1_sample = [] if len(cls_1) == 0 else np.random.choice(cls_1, size=int(config['sample_num'] * 0.25), replace=False)
        cls_2_sample = [] if len(cls_2) == 0 else np.random.choice(cls_2, size=int(config['sample_num'] * 0.25), replace=False)
        cls_3_sample = [] if len(cls_3) == 0 else np.random.choice(cls_3, size=int(config['sample_num'] * 0.25), replace=False)
        cls_4_sample = [] if len(cls_4) == 0 else np.random.choice(cls_4, size=int(config['sample_num'] * 0.25), replace=False)
        cls_5_sample = [] if len(cls_5) == 0 else np.random.choice(cls_5, size=int(config['sample_num'] * 0.25),
                                                                   replace=False)
        labels = [0] * len(cls_0_sample) + [1] * len(cls_1_sample) + [2] * len(cls_2_sample) + [3] * len(
            cls_3_sample) + [4] * len(cls_4_sample) + [5] * len(cls_5_sample)
        file_paths = list(cls_0_sample) + list(cls_1_sample) + list(cls_2_sample) + list(cls_3_sample) + list(
            cls_4_sample) + list(cls_5_sample)

        df = pd.DataFrame({'file_path': file_paths, 'class': labels})
        df.to_csv(img_sampled_path, index=False)

def get_from_gen_done(gen_done_path):
    if not os.path.exists(gen_done_path):
        return None

    frames = [f'{i}.jpg' for i in range(5, 16)]
    video_frames = []
    for frame in frames:
        frame_path = os.path.join(gen_done_path, frame)
        image = Image.open(frame_path).convert("RGB")
        video_frames.append(image)
    return video_frames


def load_videos_from_folder(dir_path, label=None):
    videos = []

    for slide in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(dir_path, slide)):
            continue
        slide_path = os.path.join(dir_path, slide)
        for sub_slide in os.listdir(slide_path):
            sub_slide_path = os.path.join(slide_path, sub_slide)
            for patch in os.listdir(sub_slide_path):
                cls = patch.split('_')[-1]
                if label is not None and str(label) != cls:
                    continue
                patch_path = os.path.join(sub_slide_path, patch)
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

def export_to_gif(frames, output_gif_path, duration):
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
                       duration=duration,
                       loop=0)