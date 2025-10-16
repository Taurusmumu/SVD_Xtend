import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of data with strings.
    """
    # Separate the different parts of the batch
    pixel_values = [item['pixel_values'] for item in batch]
    blur_degrees = [item['blur_degrees'] for item in batch]
    slide_names = [item['slide_name'] for item in batch]
    patch_names = [item['patch_name'] for item in batch]
    frames = [item['frames'] for item in batch]

    # Stack the numerical/array data into tensors
    # Assuming pixel_values are lists of numpy arrays, we stack them
    # pixel_values_batch = torch.from_numpy(np.array(pixel_values))
    # blur_degrees_batch = torch.tensor(blur_degrees, dtype=torch.float32)

    # Return a dictionary where strings are kept as a list
    return {
        'pixel_values': pixel_values,
        'blur_degrees': blur_degrees,
        "slide_name": slide_names, # This is now a list of strings
        "patch_name": patch_names, # This is now a list of strings
        "frames": frames,
    }


class GTSampleDataset(Dataset):

    def __init__(
            self, img_size=256, num_frames=11, channels=3, blur_threshold=0.2,
            data_dir="/ssd2/AMC_zstack_2_patches_warp/pngs_mid",
            sample_file_path="/home/compu/jiamu/SVD_Xtend/image_process/blur_motion_data3.csv",
    ):
        self.blur_threshold = blur_threshold
        self.channels = channels
        self.sample_frames = num_frames
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.CenterCrop((img_size-3, img_size-3)),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda img: img.convert("RGB")),  # _convert_to_rgb
            # transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.48145466, 0.4578275, 0.40821073),
            #     std=(0.26862954, 0.26130258, 0.27577711)
            # )
            # transforms.Normalize([0.5], [0.5]),
        ])

        start_layer_dict = {}
        blur_degree_dict = {}
        motion_degree_dict = {}
        img_dict = {}
        with open(sample_file_path, "r") as rf:
            rf.readline()
            for line in rf:
                line_split = line.strip().split(",")
                slide_name = line_split[0]
                patch_name = line_split[1]
                start_layer = int(line_split[-3])
                end_layer = int(line_split[-2])
                clear_layer = int(line_split[-1])

                if slide_name not in start_layer_dict:
                    start_layer_dict[slide_name] = {}
                    motion_degree_dict[slide_name] = {}
                    blur_degree_dict[slide_name] = {}
                    img_dict[slide_name] = {}
                start_layer_dict[slide_name][patch_name] = (start_layer, end_layer, clear_layer)
                motion_blur_score = [score for score in line_split[2:-3]]
                motion_degree_dict[slide_name][patch_name] = [score.split(';')[0] for score in motion_blur_score]
                blur_degree_dict[slide_name][patch_name] = [score.split(';')[1] for score in motion_blur_score]

                slide_dir = os.path.join(data_dir, slide_name)
                for layer in os.listdir(slide_dir):
                    patch_path = os.path.join(slide_dir, layer, patch_name)
                    if patch_name not in img_dict[slide_name]:
                        img_dict[slide_name][patch_name] = {}
                    img_dict[slide_name][patch_name][layer] = patch_path

        print("Loading image paths from \"{}\".".format(data_dir))

        self.samples = []
        self.clear_frames = []
        self.blur_degrees = []
        for slide_name, slide_data in img_dict.items():
            for patch_name, patch_data in slide_data.items():
                try:
                    start_layer = start_layer_dict[slide_name][patch_name][0]
                    end_layer = start_layer_dict[slide_name][patch_name][1]
                    min_layer = start_layer_dict[slide_name][patch_name][2]
                except KeyError as e:
                    continue

                # Filter out patches that do not have full 11 frames
                if start_layer < 0 or (end_layer - start_layer + 1) < self.sample_frames:
                    assert KeyError

                # Current 3D image
                img_3D, blurs = [], []
                for i, (layer, patch_path) in enumerate(sorted(patch_data.items())):

                    if i >= start_layer:
                        img_3D.append(patch_path)
                        blurs.append(blur_degree_dict[slide_name][patch_name][i])

                    if len(img_3D) == self.sample_frames:
                        break

                assert len(img_3D) == self.sample_frames
                assert len(blurs) == self.sample_frames
                self.samples.append(img_3D)
                self.blur_degrees.append(blurs)

        print("{} samples loaded.".format(len(self.samples)))

    def __getitem__(self, index):
        selected_frames = self.samples[index]
        blur_degrees = self.blur_degrees[index]
        # blur_bools = np.array(blur_degrees) <= self.blur_threshold
        pixel_values = []
        # Load and process each frame
        frames = []
        for i, frame_path in enumerate(selected_frames):
            slide_name, layer, patch_name = frame_path.split('/')[-3:]
            frames.append(layer)
            with Image.open(frame_path) as img:
                img = self.transform(img)
                pixel_values.append(img)

        mid_frame = pixel_values[len(pixel_values)//2]
        output = {
            'pixel_values': pixel_values,
            'blur_degrees': list(blur_degrees),
            "slide_name": slide_name,
            "patch_name": patch_name.split('.')[0],
            "frames": frames, # layer.png
        }

        return output

    def __len__(self):
        return len(self.samples)
        # return 0


if __name__ == "__main__":
    GT_dir = "/ssd2/AMC_zstack_2_patches/output_for_metrics/output_0604/GT"
    Pred_dir = "/ssd2/AMC_zstack_2_patches/output_for_metrics/output_0604/Pred/output0526_80000"
    # real_videos = load_videos_from_folder(GT_dir)
    # np.save("/ssd2/AMC_zstack_2_patches/output_for_metrics/real_videos.npy", real_videos.astype(np.float32))

    # fake_videos = load_videos_from_folder(Pred_dir)
    # np.save("/ssd2/AMC_zstack_2_patches/output_for_metrics/fake_videos_0526_80000.npy", fake_videos.astype(np.float32))


    sample_num = 5000

    sample_file_path = f"/home/compu/jiamu/SVD_Xtend/image_process/blur_data_sampled_{sample_num}.csv"
    output_dir = "/ssd2/AMC_zstack_2_patches/output_for_metrics/GT"
    dataset = GTSampleDataset(data_dir=output_dir, sample_file_path=sample_file_path)
    print(f"Dataset length: {len(dataset)}")
    sample_image = dataset[0]
    print(f"Sample image shape: {sample_image.shape}")