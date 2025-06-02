from PIL import Image
import os

def from_gif_to_frames(gif_path, output_dir):
    """
    Convert a GIF to individual frames and save them as images.

    Args:
    - gif_path (str): Path to the input GIF file.
    - output_dir (str): Directory where the frames will be saved.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with Image.open(gif_path) as img:
        for frame_number in range(img.n_frames):
            img.seek(frame_number)
            frame = img.convert("RGB")
            frame.save(os.path.join(output_dir, f"frame_{frame_number:04d}.png"))


if __name__ == "__main__":
    root_path = "/ssd2/AMC_zstack_2_patches/output0522/"  # Replace with your output directory

    for dir in ['GT', 'Pred']:
        gif_root_path = os.path.join(root_path, dir)  # Replace with your GIF path
        for gif_path in os.listdir(gif_root_path):
            if gif_path.endswith('.gif'):
                full_gif_path = os.path.join(gif_root_path, gif_path)
                basename = os.path.basename(gif_path)
                output_dir = os.path.join(gif_root_path, gif_path.replace('.gif', ''))
                from_gif_to_frames(full_gif_path, output_dir)
                print(f"Frames extracted to {output_dir}")
