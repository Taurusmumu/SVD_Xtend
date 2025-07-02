import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

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

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)

class AMCDataset(Dataset):
    def __init__(
            self, split, img_size=256, sample_frames=11, channels=3,blur_threshold=0.2,
            data_dir="/ssd2/AMC_zstack_2_patches/pngs_mid",
            start_layer_path="/home/compu/jiamu/SVD_Xtend/image_process/blur_data5.csv",
            split_file="/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt",
            # num_layers=19,
    ):
        self.blur_threshold = blur_threshold
        self.channels = channels
        self.sample_frames = sample_frames
        self.img_size = img_size
        self.split = split
        self.transform = transforms.Compose([
            # transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((img_size, img_size)),
            # transforms.Lambda(lambda img: img.convert("RGB")),  # _convert_to_rgb
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.48145466, 0.4578275, 0.40821073),
            #     std=(0.26862954, 0.26130258, 0.27577711)
            # )
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_middle = transforms.Compose([
            # transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((img_size, img_size)),
            # transforms.Lambda(lambda img: img.convert("RGB")),  # _convert_to_rgb
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # self.num_layers = num_layers

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
        blur_degree_dict = {}
        with open(start_layer_path, "r") as rf:
            rf.readline()
            for line in rf:
                line_split = line.strip().split(",")
                slide_name = line_split[0]
                patch_name = line_split[1]
                start_layer = int(line_split[-3])
                end_layer = int(line_split[-2])
                clear_layer = int(line_split[-1])
                # if patch_name == "patch_47_6181_4930.png" or patch_name == "patch_956_7003_23258.png" or \
                #         patch_name == "patch_6507_20844_15426.png" or patch_name == "patch_6570_17725_29207.png":
                #     print(patch_name)
                #     print(slide_name)
                #     print(start_layer)
                #     print(clear_frame)

                if slide_name not in start_layer_dict:
                    start_layer_dict[slide_name] = {}
                    blur_degree_dict[slide_name] = {}
                start_layer_dict[slide_name][patch_name] = (start_layer, end_layer, clear_layer)
                blur_degree_dict[slide_name][patch_name] = [float(score) for score in line_split[2:-3]]

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

                patch_imgs = []
                blurs = []
                valid_clear_frame = False
                for i, (layer, patch_path) in enumerate(sorted(patch_data.items())):
                    while start_layer < 0:
                        patch_imgs.append(patch_path)
                        blurs.append(np.Inf)
                        start_layer += 1

                    if i >= start_layer:
                        patch_imgs.append(patch_path)
                        blurs.append(blur_degree_dict[slide_name][patch_name][i])

                    # if i == clear_frame:
                    #     self.clear_frames.append(clear_frame - start_layer)
                    #     valid_clear_frame = True

                    if len(patch_imgs) == sample_frames:
                        break

                    if (i == len(patch_data.items()) - 1) and (len(patch_imgs) < sample_frames):
                        while len(patch_imgs) < sample_frames:
                            patch_imgs.append(patch_path)
                            blurs.append(np.Inf)

                assert len(patch_imgs) == sample_frames
                assert len(blurs) == sample_frames
                self.samples.append(patch_imgs)
                self.blur_degrees.append(blurs)

        print("{} samples loaded.".format(len(self.samples)))

    def __getitem__(self, index):
        selected_frames = self.samples[index]
        blur_degrees = self.blur_degrees[index]
        blur_degrees = np.array(blur_degrees) <= self.blur_threshold

        # selected_frames, start_layer = self.samples[index]
        # alpha = start_layer / self.num_layers

        # p = random.choice(['A', 'B', 'C'])
        # if p == 'A':
        #     selected_frames = [frames[0], frames[2], frames[4]]
        # elif p == 'B':
        #     selected_frames = [frames[2], frames[4], frames[6]]
        # else:
        #     selected_frames = [frames[4], frames[6], frames[8]]

        # q = random.uniform(0, 1)
        # if q > 0.5:
        #     selected_frames = selected_frames[::-1]

        pixel_values = torch.empty((self.sample_frames, self.channels, self.img_size, self.img_size))

        # Load and process each frame
        for i, frame_path in enumerate(selected_frames):
            with Image.open(frame_path) as img:
                # Resize the image and convert it to a tensor
                img_tensor = self.transform(img)
                pixel_values[i] = img_tensor

        mid_frame = pixel_values[len(pixel_values)//2]
        # clear_frame = pixel_values[self.clear_frames[index]]
        # clear_frame = Image.open(clear_frame)
        # clear_frame = self.transform_middle(clear_frame)

        return {'pixel_values': pixel_values, 'mid_frame': mid_frame, 'blur_bool': list(blur_degrees)}
        # return {'pixel_values': pixel_values, 'middle_frame': middle_frame, 'alpha': alpha}


    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train_dataset = AMCDataset(split="train", img_size=256)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=0, pin_memory=True, shuffle=True)

    val_dataset = AMCDataset(split="val", img_size=256)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, num_workers=0, pin_memory=True, shuffle=False)
    # #
    test_dataset = AMCDataset(split="test", img_size=256)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=0, pin_memory=True, shuffle=False)

    for step, batch in enumerate(train_loader):
        print(batch['pixel_values'].shape)
# Skip steps until we reach the resumed ste
