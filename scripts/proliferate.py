import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
import numpy as np
import torch
from PIL import Image
from diffusers.utils import load_image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers import StableVideoDiffusionPipeline
from accelerate import PartialState
from PIL import ImageFont
from PIL import ImageDraw
distributed_state = PartialState()
device = distributed_state.device


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


if __name__ == "__main__":
    root_path = "/ssd2/AMC_zstack_2_patches/pngs_mid"
    pred_path = "/ssd2/AMC_zstack_2_patches/output_for_metrics/output0623/Pred_random/200000/"
    output_path = "/ssd2/AMC_zstack_2_patches/output0623/proliferate/"
    os.makedirs(output_path, exist_ok=True)
    num_frame = 11
    proliferate_time = (51 - num_frame) / (num_frame - 1)
    size = 256
    layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09", "z10", "z11", "z12", "z13", "z14",
              "z15", "z16", "z17", "z18"]
    pretrained_model_name = "stabilityai/stable-video-diffusion-img2vid-xt"

    version = "output0623"
    ckp_v = 200000
    pretrained_model_path = f"/ssd2/AMC_zstack_2_patches/{version}/checkpoint-{ckp_v}"

    info_sets = []
    info_sets.append({
        "patch_name": "patch_6114_17494_46381",
        "silde": '24S 048630;E;10;;FA0824;1_241226_161645',
        "start_layer": 0,
    })
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name, subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name, subfolder="image_encoder"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name, subfolder="vae", variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # # generator = torch.Generator("cuda:0").manual_seed(8)
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        pretrained_model_name,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=torch.float32
    )
    # pipeline.set_progress_bar_config(disable=True)

    pipeline.to(device)
    for info in info_sets:
        patch_name = info["patch_name"]
        slide = info["silde"]
        start_layer = info["start_layer"]

        final_image = []
        # step 1 intial generate
        frames = layers[start_layer: start_layer + num_frame]
        mid_frame = frames[(len(frames) - 1) // 2]
        source_path = f'{os.path.join(root_path, slide, mid_frame, patch_name)}.png'
        video_frames = pipeline(
            load_image(source_path).resize((size, size)),
            height=size,
            width=size,
            num_frames=num_frame,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02,
            # generator=generator,
        ).frames[0]
        for f in range(num_frame):
            img = video_frames[f]
            final_image.append(img)

        # step 2 top generate
        for j in range(2):
            mid_frame = final_image[0]
            video_frames = pipeline(
                load_image(mid_frame).resize((size, size)),
                height=size,
                width=size,
                num_frames=num_frame,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.02,
            ).frames[0]
            for f in range(5):
                img = video_frames[f]
                final_image.insert(f, img)

        # step 3 bottom generate
        for j in range(2):
            mid_frame = final_image[-1]
            video_frames = pipeline(
                load_image(mid_frame).resize((size, size)),
                height=size,
                width=size,
                num_frames=num_frame,
                decode_chunk_size=8,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.02,
            ).frames[0]
            for f in range(num_frame//2 + 1, num_frame):
                img = video_frames[f]
                final_image.append(img)

        os.makedirs(f"{output_path}/{slide}/{patch_name}", exist_ok=True)
        for e, img in enumerate(final_image):
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Layer {}".format(e + 1), (255, 255, 255), font=ImageFont.load_default())
            img.save(f"{output_path}/{slide}/{patch_name}/{e}.png")

        export_to_gif(final_image, output_path + f"{slide}/{patch_name}_proliferate.gif", fps=10)

        # pred_frames = []
        # pred_folder_path = os.path.join(pred_path, slide, patch_name)
        # for frame in range(start_layer, start_layer + num_frame, 1):
        #     frame_path = f"{pred_folder_path}/z{from_number_to_layer(frame)}.png"
        #     print(frame_path)
        #     frame = Image.open(frame_path)
        #     pred_frames.append(frame)
        # export_to_gif(pred_frames, output_gif_path + f"/{patch_name}_Pred.gif", fps=10)


