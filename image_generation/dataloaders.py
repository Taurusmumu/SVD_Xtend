import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import glob

def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of data with strings.
    """
    # Separate the different parts of the batch
    pixel_value = [item['pixel_value'] for item in batch]
    base_dir = [item['base_dir'] for item in batch]
    base_patch = [item['base_patch'] for item in batch]

    # Stack the numerical/array data into tensors
    # Assuming pixel_values are lists of numpy arrays, we stack them
    # pixel_values_batch = torch.from_numpy(np.array(pixel_values))
    # blur_degrees_batch = torch.tensor(blur_degrees, dtype=torch.float32)

    # Return a dictionary where strings are kept as a list
    return {
        'pixel_value': pixel_value,
        'base_dir': base_dir,
        "base_patch": base_patch, # This is now a list of strings
    }


class ProstateSampleDataset(Dataset):

    def __init__(
            self, img_size=256, channels=3,
            split='train', # 'train' or 'validation' or 'test'
            data_dir="/ssd2/AMC_zstack_2_patches_warp/pngs_mid"
    ):
        self.channels = channels
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda img: img.convert("RGB")),  # _convert_to_rgb
            # transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.48145466, 0.4578275, 0.40821073),
            #     std=(0.26862954, 0.26130258, 0.27577711)
            # )
            # transforms.Normalize([0.5], [0.5]),
        ])

        self.samples = []
        for dir in os.listdir(data_dir):
            if split not in dir:
                continue
            split_dir = os.path.join(data_dir, dir)
            for file_name in glob.glob(os.path.join(split_dir, '*', '*', '*.jpg')):
                self.samples.append(file_name)
            for file_name in glob.glob(os.path.join(split_dir, '*', '*.jpg')):
                self.samples.append(file_name)

        print("{} samples loaded.".format(len(self.samples)))

    def __getitem__(self, index):
        fn = self.samples[index]
        with Image.open(fn) as img:
            img = self.transform(img)
        base_dir = '/'.join(os.path.dirname(fn).split('/')[3:])
        base_patch = os.path.basename(fn).split('.jpg')[0]
        output = {
            'pixel_value': img,
            'base_dir': base_dir,
            "base_patch": base_patch,
        }

        return output

    def __len__(self):
        return len(self.samples)
        # return 0


if __name__ == "__main__":

    data_dir = "/ssd1/prostate_harvard/"
    dataset = ProstateSampleDataset(data_dir=data_dir, split='train')
    print(f"Dataset length: {len(dataset)}")
    sample_image = dataset[0]
    print(f"Sample image shape: {sample_image.shape}")