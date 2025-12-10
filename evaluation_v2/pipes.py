import torch
import torch.nn as nn
from src.pipeline_stable_video_diffusion_dpmsolver import StableVideoDiffusionPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from src.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

def load_pipe(config, device):
    """
    Loads the reconstruction model.

    Args:
        config (dict): The evaluation configuration.
        device (torch.device): The device to load the model onto.

    Returns:
        A loaded model ready for inference.
    """
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        config['pretrained_model_name'], subfolder="vae", variant="fp16")
    # Freeze vae and image_encoder
    vae_state_dict = torch.load(config['pretrained_vae_path'], map_location="cpu")
    vae.load_state_dict(vae_state_dict, strict=True)
    vae.requires_grad_(False)
    vae.to(device)
    generator = torch.Generator(device).manual_seed(8)
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        config['pretrained_model_name'],
        vae=vae,
        torch_dtype=torch.float32,
        generator=generator
    )
    if 'lora' in config['output_path']:
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            config['pretrained_model_name'],
            subfolder="unet",
            low_cpu_mem_usage=True,
            variant="fp16",
        )
        pipeline.unet = unet
        # pipeline.load_lora_weights(config['pretrained_unet_name'])
        pipeline.unet.load_attn_procs(config['pretrained_unet_path'])
    else:
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            config['pretrained_unet_path'],
            subfolder="unet",
            low_cpu_mem_usage=True,
        )
        pipeline.unet = unet
    pipeline.unet.requires_grad_(False)
    pipeline.unet.to(device)
    # --- 5. Move to GPU ---
    # pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)
    # pipeline.enable_model_cpu_offload()

    return pipeline
