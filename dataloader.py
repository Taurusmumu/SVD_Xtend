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
            self, split, img_size=256, sample_frames=11, channels=3, blur_threshold=0.2,
            data_dir="/ssd2/AMC_zstack_2_patches_warp/pngs_mid",
            start_layer_path="/home/compu/jiamu/SVD_Xtend/image_process/blur_motion_data5.csv",
            split_file="/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt",
    ):
        self.blur_threshold = blur_threshold
        self.channels = channels
        self.sample_frames = sample_frames
        self.img_size = img_size
        self.split = split
        self.transform = transforms.Compose([
            transforms.CenterCrop((img_size-3, img_size-3)),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.Lambda(lambda img: img.convert("RGB")),  # _convert_to_rgb
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.48145466, 0.4578275, 0.40821073),
            #     std=(0.26862954, 0.26130258, 0.27577711)
            # )
            transforms.Normalize([0.5], [0.5]),
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
        blur_degree_dict = {}
        motion_degree_dict = {}
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
                    motion_degree_dict[slide_name] = {}
                start_layer_dict[slide_name][patch_name] = (start_layer, end_layer, clear_layer)
                motion_blur_score = [score for score in line_split[2:-3]]
                motion_degree_dict[slide_name][patch_name] = [score.split(';')[0] for score in motion_blur_score]
                blur_degree_dict[slide_name][patch_name] = [score.split(';')[1] for score in motion_blur_score]

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
        self.motion_degrees = []
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
                motions = []
                valid_clear_frame = False
                for i, (layer, patch_path) in enumerate(sorted(patch_data.items())):
                    while start_layer < 0:
                        patch_imgs.append(patch_path)
                        blurs.append(np.Inf)
                        motions.append(0)
                        start_layer += 1

                    if i >= start_layer:
                        patch_imgs.append(patch_path)
                        blurs.append(blur_degree_dict[slide_name][patch_name][i])
                        motions.append(motion_degree_dict[slide_name][patch_name][i])

                    # if i == clear_frame:
                    #     self.clear_frames.append(clear_frame - start_layer)
                    #     valid_clear_frame = True

                    if len(patch_imgs) == sample_frames:
                        break

                    if (i == len(patch_data.items()) - 1) and (len(patch_imgs) < sample_frames):
                        while len(patch_imgs) < sample_frames:
                            patch_imgs.append(patch_path)
                            blurs.append(np.Inf)
                            motions.append(0)

                assert len(patch_imgs) == sample_frames
                assert len(blurs) == sample_frames
                self.samples.append(patch_imgs)
                self.blur_degrees.append(blurs)
                self.motion_degrees.append(motions)

        print("{} samples loaded.".format(len(self.samples)))

    def __getitem__(self, index):
        selected_frames = self.samples[index]
        blur_degrees = [float(s) for s in self.blur_degrees[index]]
        motion_degrees = [float(s) for s in self.motion_degrees[index]]
        mask = np.array(blur_degrees) <= self.blur_threshold
        motion = np.sum(np.array(motion_degrees)[mask])

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
        return {'pixel_values': pixel_values, 'mid_frame': mid_frame, 'mask': list(mask), 'motion': motion}


    def __len__(self):
        return len(self.samples)


class AMC_spatial_Dataset(Dataset):
    def __init__(
            self, split, img_size=256, channels=3,blur_threshold=0.2,sample_frames=1,
            data_dir="/ssd1/AMC_zstack_2_patches_warp/pngs_mid",
            start_layer_path="/home/compu/jiamu/SVD_Xtend/image_process/blur_motion_data2.csv",
            split_file="/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt",
    ):
        self.sample_frames = sample_frames
        self.blur_threshold = blur_threshold
        self.channels = channels
        self.img_size = img_size
        self.split = split
        self.transform = transforms.Compose([
            # transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((img_size, img_size)),
            # transforms.Lambda(lambda img: img.convert("RGB")),  # _convert_to_rgb
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
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

        blur_valid_dict = {}
        target_layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09",
                         "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]
        target_layers = np.array(target_layers)
        with open(start_layer_path, "r") as rf:
            rf.readline()
            for line in rf:
            # num_lines_to_read = 2000
            # for i in range(num_lines_to_read):
            #     line = rf.readline()  # Read one line
            #     if not line:  # readline() returns an empty string at the end of the file
            #         break
                # print(line.strip())
                line_split = line.strip().split(",")
                slide_name = line_split[0]
                if split_dict[slide_name] != self.split:
                    continue

                patch_name = line_split[1]

                if slide_name not in blur_valid_dict:
                    blur_valid_dict[slide_name] = {}
                motion_blur_score = [score for score in line_split[2:-3]]
                blur_list = np.array([float(score.split(';')[1]) for score in motion_blur_score])
                blur_valid_dict[slide_name][patch_name] = target_layers[blur_list <= blur_threshold]

        print("Loading image paths from \"{}\".".format(data_dir))

        self.samples = []
        for slide_name, slide_data in blur_valid_dict.items():
            for patch_name, patch_data in slide_data.items():
                for layer in patch_data:
                    self.samples.append(os.path.join(data_dir, slide_name, layer, patch_name))

        print("{} samples loaded.".format(len(self.samples)))

    def __getitem__(self, index):
        file_path = self.samples[index]
        # Load and process each frame
        with Image.open(file_path) as img:
            # Resize the image and convert it to a tensor
            img_tensor = self.transform(img)

        return {'pixel_values': img_tensor.unsqueeze(0)}

    def __len__(self):
        return len(self.samples)


class AMC_TempAug_Dataset(Dataset):
    def __init__(
            self, split, img_size=256, sample_frames=11, channels=3, blur_threshold=0.2,
            data_dir="/ssd2/AMC_zstack_2_patches_warp/pngs_mid",
            start_layer_path="/home/compu/jiamu/SVD_Xtend/image_process/blur_motion_data4.csv",
            split_file="/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt",
            augment_dir="/ssd2/AMC_zstack_2_patches_flip/pngs_mid",
            augment_anno="/home/compu/jiamu/SVD_Xtend/image_process/motion_annotations.csv"
    ):
        self.blur_threshold = blur_threshold
        self.channels = channels
        self.sample_frames = sample_frames
        self.img_size = img_size
        self.split = split
        self.transform = transforms.Compose([
            transforms.CenterCrop((img_size-3, img_size-3)),
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
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
        blur_degree_dict = {}
        motion_degree_dict = {}
        with open(start_layer_path, "r") as rf:
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
                    blur_degree_dict[slide_name] = {}
                    motion_degree_dict[slide_name] = {}
                start_layer_dict[slide_name][patch_name] = (start_layer, end_layer, clear_layer)
                motion_blur_score = [score for score in line_split[2:-3]]
                motion_degree_dict[slide_name][patch_name] = [score.split(';')[0] for score in motion_blur_score]
                blur_degree_dict[slide_name][patch_name] = [score.split(';')[1] for score in motion_blur_score]

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
        self.motion_degrees = []
        self.is_aug = []
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
                motions = []
                valid_clear_frame = False
                for i, (layer, patch_path) in enumerate(sorted(patch_data.items())):
                    while start_layer < 0:
                        patch_imgs.append(patch_path)
                        blurs.append(np.Inf)
                        motions.append(0)
                        start_layer += 1

                    if i >= start_layer:
                        patch_imgs.append(patch_path)
                        blurs.append(blur_degree_dict[slide_name][patch_name][i])
                        motions.append(motion_degree_dict[slide_name][patch_name][i])

                    # if i == clear_frame:
                    #     self.clear_frames.append(clear_frame - start_layer)
                    #     valid_clear_frame = True

                    if len(patch_imgs) == sample_frames:
                        break

                    if (i == len(patch_data.items()) - 1) and (len(patch_imgs) < sample_frames):
                        while len(patch_imgs) < sample_frames:
                            patch_imgs.append(patch_path)
                            blurs.append(np.Inf)
                            motions.append(0)

                assert len(patch_imgs) == sample_frames
                assert len(blurs) == sample_frames
                self.samples.append(patch_imgs)
                self.blur_degrees.append(blurs)
                self.motion_degrees.append(motions)
                self.is_aug.append(0)

        print("{} samples loaded.".format(len(self.samples)))
        print("Loading augmented data from \"{}\".".format(augment_dir))
        with open(augment_anno, "r") as rf:
            rf.readline()
            for line in rf:
                line_split = line.strip().split(",")
                slide_name = line_split[0]
                if (self.split is not None and self.split != "all"
                        and split_dict[slide_name] != self.split):
                    continue
                patch_name = line_split[1]
                motion_scores = line_split[2:]

                patch_img_folder = os.path.join(augment_dir, slide_name, patch_name.split('.')[0])
                patch_imgs = []
                # if not os.path.exists(patch_img_folder):
                #     continue
                for path in sorted(os.listdir(patch_img_folder)):
                    patch_imgs.append(os.path.join(patch_img_folder, path))

                self.samples.append(patch_imgs)
                self.blur_degrees.append(-1)
                self.motion_degrees.append([float(score) for score in motion_scores])
                self.is_aug.append(1)

        print("{} Total samples loaded.".format(len(self.samples)))

    def __getitem__(self, index):
        selected_frames = self.samples[index]
        if self.is_aug[index] == 0:
            blur_degrees = [float(s) for s in self.blur_degrees[index]]
            motion_degrees = [float(s) for s in self.motion_degrees[index]]
            mask = np.array(blur_degrees) <= self.blur_threshold
            motion = np.sum(np.array(motion_degrees)[mask])
            pixel_values = torch.empty((self.sample_frames, self.channels, self.img_size, self.img_size))

            # Load and process each frame
            for i, frame_path in enumerate(selected_frames):
                with Image.open(frame_path) as img:
                    # Resize the image and convert it to a tensor
                    img_tensor = self.transform(img)
                    pixel_values[i] = img_tensor

            mid_frame = pixel_values[len(pixel_values)//2]
            spatial_mask = self.mask_for_aug(self.is_aug[index])
            return {'pixel_values': pixel_values, 'mid_frame': mid_frame, 'mask': list(mask), 'motion': motion,
                    'spatial_mask': spatial_mask}
        else:
            mask = [True for i in range(len(selected_frames))]
            motion = np.sum(self.motion_degrees[index])
            pixel_values = torch.empty((self.sample_frames, self.channels, self.img_size, self.img_size))
            spatial_mask = self.mask_for_aug(self.is_aug[index])

            # Load and process each frame
            for i, frame_path in enumerate(selected_frames):
                with Image.open(frame_path) as img:
                    # Resize the image and convert it to a tensor
                    img_tensor = self.transform(img)
                    pixel_values[i] = img_tensor

            # random rotation
            if random.random() > 0.5:
                angle = random.choice([90, 270])
                pixel_values = torch.rot90(pixel_values, k=angle//90, dims=[2,3])
                spatial_mask = torch.rot90(spatial_mask, k=angle//90, dims=[0,1])

            mid_frame = pixel_values[len(pixel_values)//2]
            return {'pixel_values': pixel_values, 'mid_frame': mid_frame, 'mask': list(mask), 'motion': motion,
                    'spatial_mask': spatial_mask}

    def mask_for_aug(self, is_aug):
        if is_aug == 0:
            mask = [True for _ in range(32)]
        else:
            slice_length, k = 32, 2
            mask = [False] * slice_length
            last_one_index = -random.randint(2, k + 1)
            while True:
                min_next_index = last_one_index + 2  # At least 1 zero in between
                max_next_index = last_one_index + k + 1  # At most k zeros in between
                next_one_index = random.randint(min_next_index, max_next_index)
                if next_one_index >= slice_length:
                    break
                mask[next_one_index] = True
                last_one_index = next_one_index

            # extend height
        mask = [mask for _ in range(32)]
        mask = torch.tensor(mask)

        assert (mask.shape[0] == 32 and mask.shape[1] == 32)
        return mask

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # train_dataset = AMC_TempAug_Dataset(split="train", img_size=256)

    train_dataset = AMCDataset(split="train", img_size=256)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=0, pin_memory=True, shuffle=True)
    #
    # val_dataset = AMCDataset(split="val", img_size=256)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, num_workers=0, pin_memory=True, shuffle=False)
    # # #
    # test_dataset = AMCDataset(split="test", img_size=256)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=0, pin_memory=True, shuffle=False)

    for step, batch in enumerate(train_loader):
        print(batch['pixel_values'].shape)
        loss_msk = torch.stack(batch["mask"], dim=0).permute(1, 0)
        pixel_values = batch['pixel_values']
        # batch['pixel_values'].unsqueeze(1).repeat(1,11,1,1)
        pixel_values[0][loss_msk[0]]

# Skip steps until we reach the resumed ste
