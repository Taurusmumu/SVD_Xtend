import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def count_num_params(model):
    total = 0
    for param in model.parameters():
        total += param.numel()

    suffixes = ['', 'K', 'M', 'B', 'T']

    # Find the magnitude of the number (i.e., how many powers of 1000 it has)
    magnitude = 0
    while abs(total) >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        total /= 1000.0

    # Format the number with 1 decimal place
    return f"{total:.1f}{suffixes[magnitude]}"

def save_orig_and_generated_images(original_images,
                                   generated_image_tensors,
                                   path_to_save_folder,
                                   step,
                                   accelerator):
    ### Create Folder if it doesnt Exist ###
    if not os.path.isdir(path_to_save_folder):
        accelerator.print(f"Creating Folder {path_to_save_folder} to save Reconstructions")
        os.makedirs(path_to_save_folder)

    ### Clamp Output Between [-1 to 1] and rescale back to [0 to 255] ###
    generated_image_tensors = generated_image_tensors.float()
    generated_image_tensors = torch.clamp(generated_image_tensors, -1., 1.)
    generated_image_tensors = (generated_image_tensors + 1) / 2
    generated_image_tensors = generated_image_tensors.cpu().permute(0, 2, 3, 1).numpy()
    generated_image_tensors = (255 * generated_image_tensors).astype(np.uint8)
    gen_imgs = [Image.fromarray(img).convert("RGB") for img in generated_image_tensors]

    ### Original Images have been scaled to [-1 to 1], rescale back to [0 to 255] ###
    original_images = original_images.float()
    original_images = (original_images + 1) / 2
    original_images = original_images.cpu().permute(0, 2, 3, 1).numpy()
    original_images = (255 * original_images).astype(np.uint8)
    orig_imgs = [Image.fromarray(img).convert("RGB") for img in original_images]

    ### Concat Images (so we can compare real vs reconstruction) ###
    img_width = orig_imgs[0].width
    img_height = orig_imgs[0].height
    combined_images = []
    for orig_img, gen_img in zip(orig_imgs, gen_imgs):
        combined_img = Image.new(mode="RGB", size=(img_width, 2 * img_height))
        combined_img.paste(orig_img, (0, 0))
        combined_img.paste(gen_img, (0, img_height))
        combined_images.append(combined_img)

    ### Concatenate All Samples Together ###
    final_image = Image.new(mode="RGB", size=(img_width * len(combined_images), 2 * img_height))
    x_offset = 0
    for img in combined_images:
        final_image.paste(img, (x_offset, 0))
        x_offset += img_width

    ### Save Output ###
    path_to_save = os.path.join(path_to_save_folder, f"iteration_{step}.png")
    final_image.save(path_to_save)


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


def save_orig_and_generated_gifs(original_images,
                                   generated_image_tensors,
                                   path_to_save_folder,
                                   step,
                                   accelerator=None):
    ### Create Folder if it doesnt Exist ###
    if not os.path.isdir(path_to_save_folder):
        os.makedirs(path_to_save_folder)
        if accelerator is not None:
            accelerator.print(f"Creating Folder {path_to_save_folder} to save Reconstructions")


    ### Clamp Output Between [-1 to 1] and rescale back to [0 to 255] ###
    generated_image_tensors = generated_image_tensors.float()
    generated_image_tensors = torch.clamp(generated_image_tensors, -1., 1.)
    generated_image_tensors = (generated_image_tensors + 1) / 2
    generated_image_tensors = generated_image_tensors.cpu().permute(0, 2, 3, 1).numpy()
    generated_image_tensors = (255 * generated_image_tensors).astype(np.uint8)
    gen_imgs = [Image.fromarray(img).convert("RGB") for img in generated_image_tensors]
    gen_gif = []
    for i in range(len(gen_imgs)):
        frame = gen_imgs[i]
        draw = ImageDraw.Draw(frame)
        draw.text((10, 10), str(i), (255, 255, 255), font=ImageFont.truetype("arial.ttf", 20))
        gen_gif.append(frame)

    export_to_gif(gen_gif, os.path.join(path_to_save_folder, f"iteration_{step}_Pred.gif"), fps=10)

    ### Original Images have been scaled to [-1 to 1], rescale back to [0 to 255] ###
    original_images = original_images.float()
    original_images = (original_images + 1) / 2
    original_images = original_images.cpu().permute(0, 2, 3, 1).numpy()
    original_images = (255 * original_images).astype(np.uint8)
    orig_imgs = [Image.fromarray(img).convert("RGB") for img in original_images]
    gen_gif = []
    for i in range(len(orig_imgs)):
        frame = orig_imgs[i]
        draw = ImageDraw.Draw(frame)
        draw.text((10, 10), str(i), (255, 255, 255), font=ImageFont.truetype("arial.ttf", 20))
        gen_gif.append(frame)

    export_to_gif(gen_gif, os.path.join(path_to_save_folder, f"iteration_{step}_GT.gif"), fps=10)


def save_generated_images(generated_image_tensors,
                          path_to_save_folder=None,
                          step=None,
                          path_to_save=None):
    """
    Quick helper function where:

    Args:
        - path_to_save_folder: Directory you want to save image
        - step: What iteration of training (expected when using path_to_save_folder)
        - path_to_save: Full path to .png if you want to save an individual image
    """

    ### Clamp Output Between [-1 to 1] and rescale back to [0 to 255] ###
    generated_image_tensors = generated_image_tensors.float()
    generated_image_tensors = torch.clamp(generated_image_tensors, -1., 1.)
    generated_image_tensors = (generated_image_tensors + 1) / 2
    generated_image_tensors = generated_image_tensors.cpu().permute(0, 2, 3, 1).numpy()
    generated_image_tensors = (255 * generated_image_tensors).astype(np.uint8)
    gen_imgs = [Image.fromarray(img).convert("RGB") for img in generated_image_tensors]

    if path_to_save_folder is not None:
        if step is not None:
            path_to_save = os.path.join(path_to_save_folder, f"iteration_{step}.png")

    ### Concat Images (so we can compare real vs reconstruction) ###
    img_width = gen_imgs[0].width
    img_height = gen_imgs[0].height

    ### Concatenate All Samples Together ###
    final_image = Image.new(mode="RGB", size=(img_width * len(gen_imgs), img_height))
    x_offset = 0
    for img in gen_imgs:
        final_image.paste(img, (x_offset, 0))
        x_offset += img_width

    final_image.save(path_to_save)


# if __name__ == "__main__":
#     load_testing_imagenet_encodings()