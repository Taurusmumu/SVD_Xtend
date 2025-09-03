import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from torchvision import transforms
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from tqdm import tqdm
from diffusers.optimization import get_scheduler
import lpips
from dataloader import AMCDataset
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from utils import save_orig_and_generated_images, count_num_params, save_orig_and_generated_gifs
from einops import rearrange

if __name__ == "__main__":
    device = "cuda:0"
    model_weight_path = "/ssd2/AMC_zstack_2_patches/vae_0809/VAETrainer/checkpoint_287500/pytorch_model.bin"
    model = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        subfolder="vae", revision=None, variant="fp16")

    train_dataset = AMCDataset(data_dir='/ssd2/AMC_zstack_2_patches/pngs_mid/',
                               split="train",
                               )
    dataloader = DataLoader(train_dataset,
                            batch_size=1,
                            num_workers=0,
                            shuffle=False)
    state_dict = torch.load(model_weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    for i, batch in enumerate(dataloader):
        # print(i)
        with torch.no_grad():
            pixel_values = batch["pixel_values"].to(device)
            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
            loss_msk = torch.stack(batch["mask"], dim=0)
            msk = loss_msk.squeeze()

            posterior = model.encode(pixel_values).latent_dist
            z = posterior.mode()
            reconstructions = model.decode(z, num_frames=11).sample

            save_orig_and_generated_gifs(original_images=pixel_values.detach()[msk],
                                         generated_image_tensors=reconstructions.detach()[msk],
                                         path_to_save_folder="/ssd2/AMC_zstack_2_patches/vae_0826/gen",
                                         step=i)
        if i == 5:
            break
