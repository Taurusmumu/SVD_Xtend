import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random


class GTSampleDataset(Dataset):
    def __init__(self, data_dir, sample_file_path, img_size=256, num_frames=11):
        self.data_dir = data_dir
        self.img_size = img_size
        self.sample_frames = num_frames
        sample_file = pd.read_csv(sample_file_path)
        self.image_root_path = "/ssd2/AMC_zstack_2_patches/pngs_mid"
        layers = list(sample_file.columns[2:-1])
        slides = list(sample_file["slide_name"])
        patches = list(sample_file["patch_name"])
        start_layers = list(sample_file["start_indices"])
        # clean_layers = list(sample_file["min_indices"])
        self.file_list = []
        self.start_list = []
        for i in range(len(slides)):
            frames = layers[start_layers[i]: start_layers[i] + self.sample_frames]
            # p = random.choice(['A', 'B', 'C'])
            # if p == 'A':
            #     selected_frames = [frames[0], frames[2], frames[4]]
            # elif p == 'B':
            #     selected_frames = [frames[2], frames[4], frames[6]]
            # else:
            #     selected_frames = [frames[4], frames[6], frames[8]]

            frame_list = []
            for frame in frames:
                patch_path = f"{slides[i]},{frame},{patches[i]}"
                frame_list.append(patch_path)
            self.file_list.append(frame_list)
            # self.clean_list.append(layers[clean_layers[i]])
            self.start_list.append(start_layers[i])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        output = {
            "source_paths": [],
            "save_paths": [],
            "frames": []
        }
        for frame in file_path:
            slide_name, layer, patch_name = frame.split(',')
            full_path = os.path.join(self.image_root_path, slide_name, layer, patch_name)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File {full_path} does not exist.")
            output["source_paths"].append(full_path)
            save_path = os.path.join(self.data_dir, slide_name, patch_name.split('.')[0], f"{layer}.png")
            output["save_paths"].append(save_path)
            output["frames"].append(layer)
        # output["clear_frames"] = os.path.join(self.image_root_path, slide_name, self.clean_list[idx], patch_name)
        output["slide_name"] = slide_name
        output["patch_name"] = patch_name.split('.')[0]
        output["start_index"] = self.start_list[idx]
        output["middle_frames"] = os.path.join(self.image_root_path, slide_name, output["frames"][len(output["frames"]) // 2], patch_name)

        return output

def load_videos_from_folder(dir_path, num_frames = 3):
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


if __name__ == "__main__":
    GT_dir = "/ssd2/AMC_zstack_2_patches/output_for_metrics/output_0604/GT"
    Pred_dir = "/ssd2/AMC_zstack_2_patches/output_for_metrics/output_0604/Pred/output0526_80000"
    # real_videos = load_videos_from_folder(GT_dir)
    # np.save("/ssd2/AMC_zstack_2_patches/output_for_metrics/real_videos.npy", real_videos.astype(np.float32))

    fake_videos = load_videos_from_folder(Pred_dir)
    np.save("/ssd2/AMC_zstack_2_patches/output_for_metrics/fake_videos_0526_80000.npy", fake_videos.astype(np.float32))


    sample_num = 5000

    sample_file_path = f"/home/compu/jiamu/SVD_Xtend/image_process/blur_data_sampled_{sample_num}.csv"
    output_dir = "/ssd2/AMC_zstack_2_patches/output_for_metrics/GT"
    dataset = GTSampleDataset(data_dir=output_dir, sample_file_path=sample_file_path)
    print(f"Dataset length: {len(dataset)}")
    sample_image = dataset[0]
    print(f"Sample image shape: {sample_image.shape}")