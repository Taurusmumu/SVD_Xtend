import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AMCDataset(Dataset):
    def __init__(
            self, split, img_size=512,
            data_dir="/ssd2/AMC_zstack_2_patches/pngs_mid",
            start_layer_path="/home/compu/jiamu/SVD_Xtend/image_process/blur_data1.csv",
            split_file="/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt",
    ):
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((img_size, img_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),  # _convert_to_rgb
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        split_dict = {}
        with open(split_file, "r") as rf:
            for line in rf.readlines():
                line = line.strip().split(",")
                slide_name = line[0]
                sudo_base_layer = line[1]
                slide_split = line[2]
                split_dict[slide_name] = slide_split

        print("Loading layers info from \"{}\".".format(start_layer_path))
        start_layer_dict = {}
        with open(start_layer_path, "r") as rf:
            rf.readline()
            for line in rf:
                line_split = line.strip().split(",")
                slide_name = line_split[0]
                patch_name = line_split[1]
                start_layer = int(line_split[-1])

                if slide_name not in start_layer_dict:
                    start_layer_dict[slide_name] = {}
                start_layer_dict[slide_name][patch_name] = start_layer

        print("Loading image paths from \"{}\".".format(data_dir))
        img_dict = {}
        for slide_name in os.listdir(data_dir):
            if (self.split is not None and self.split != "all"
                    and split_dict[slide_name] != self.split):
                continue

            img_dict[slide_name] = {}
            slide_dir = os.path.join(data_dir, slide_name)
            for layer in os.listdir(slide_dir):
                layer_dir = os.path.join(slide_dir, layer)
                for patch_name in os.listdir(layer_dir):
                    patch_path = os.path.join(layer_dir, patch_name)
                    if patch_name not in img_dict[slide_name]:
                        img_dict[slide_name][patch_name] = {}
                    img_dict[slide_name][patch_name][layer] = patch_path

        self.samples = []
        for slide_name, slide_data in img_dict.items():
            for patch_name, patch_data in slide_data.items():
                try:
                    start_layer = start_layer_dict[slide_name][patch_name]
                except KeyError as e:
                    continue

                patch_imgs = []
                for i, (layer, patch_path) in enumerate(sorted(patch_data.items())):
                    if i >= start_layer:
                        patch_imgs.append(patch_path)
                    if len(patch_imgs) == 9:
                        break
                self.samples.append(patch_imgs)

        print("{} samples loaded.".format(len(self.samples)))

    def __getitem__(self, index):
        img_paths = self.samples[index]
        if self.split == "train":
            img_path = random.choice(img_paths)
        else:
            img_path = img_paths[len(img_paths) // 2]

        image = Image.open(img_path)
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train_dataset = AMCDataset(split="train", img_size=512)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=1, pin_memory=True, shuffle=True)

    val_dataset = AMCDataset(split="val", img_size=512)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, num_workers=1, pin_memory=True, shuffle=False)

    test_dataset = AMCDataset(split="test", img_size=512)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=1, pin_memory=True, shuffle=False)

    for batch in train_loader:
        print(batch.shape)
