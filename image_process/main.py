from tqdm import tqdm
import cv2
import random
import os
import skimage.io
from scipy.stats import norm
from skimage.restoration import richardson_lucy
from skimage import img_as_float
from skimage import img_as_ubyte
from scipy.signal import gaussian
from scipy.signal.windows import gaussian
import numpy as np
import matplotlib.pyplot as plt


from defocus_estimate import *
import multiprocessing as mp
import os

import cv2
import numpy as np
from joblib import Parallel, delayed


root_dir = "/ssd2/AMC_zstack_2_patches/pngs_mid"
text_file_path = "/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt"


def estimate_psf_size(blurred_image, iterations=30, psf_size=(0, 0)):
    """
    Estimate the blur level by applying Richardson-Lucy deconvolution
    and inspecting the PSF-like effect.
    """
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
    image = img_as_float(gray)

    # Assume a Gaussian-like PSF (initial guess)

    psf_1d = gaussian(psf_size[0], std=0.5).reshape(-1, 1)
    psf = psf_1d @ psf_1d.T
    psf /= psf.sum()

    # Perform Richardson-Lucy deconvolution
    deconvolved = richardson_lucy(image, psf)

    # Estimate blur: how much was "removed" (difference between input and deblurred image)
    difference = np.abs(image - deconvolved)
    blur_strength = difference.mean()

    # plt.figure(figsize=(10, 4))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(image.astype(np.uint8))
    # plt.title('Original Image')
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(deconvolved.astype(np.uint8))
    # plt.title('Gaussian Blurred Image')
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    return blur_strength, deconvolved


def gaussian_blur(image, layer):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

    # Apply Gaussian Blur
    # Parameters: (image, kernel size, sigmaX)
    blurred = cv2.GaussianBlur(image_rgb, ksize=(1, 1), sigmaX=0.3)

    # Display original vs. blurred
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title(f'{layer} Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blurred)
    plt.title('Gaussian Blurred Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def calc_score(layer_dir, patch_name):
    try:
        patch_path = os.path.join(layer_dir, patch_name)
        img = cv2.imread(patch_path)
        fblurmap = estimate_bmap_laplacian(img, sigma_c=1, std1=1, std2=1.5)
        score2 = np.mean(fblurmap)
    except Exception as e:
        print(f"Error processing {patch_name}: {e}")
        score2 = -1
    return [patch_name, score2]


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    # k = 100
    # layers = ["z01", "z03", "z05", "z07", "z09", "z11", "z13", "z15", "z17"]
    # layers = ["z00", "z03", "z06", "z09", "z12", "z15", "z18"]
    target_layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09",
              "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]
    # slices = ["24S 056115;F;9;;FA0824;1_241226_045830"]
    # patch_coor = "patch_39_5268_10000.png" # patch_43_5268_11024
    # patch_coor = "patch_43_5268_11024.png"  # patch_43_5268_11024

    # file = open(text_file_path, "r")
    # lines = file.read()
    # file.close()
    # lines = lines.split('\n')
    # slices = [line.split(',')[0] for line in lines]
    # clear_layers = [line.split(',')[1] for line in lines]
    # clear_layer_np = []
    # for layer in clear_layers:
    #     if layer == "00":
    #         clear_layer_np.append(0)
    #     elif layer == "09":
    #         clear_layer_np.append(9)
    #     else:
    #         clear_layer_np.append(18)
    # clear_layer_np = np.array(clear_layer_np)
    # train_val_splits = [line.split(',')[2] for line in lines]
    # train_val_splits = np.array([1 if s == "train" else 0 for s in train_val_splits])

    n_jobs = mp.cpu_count() * 80 // 100

    blur_dict = {}
    for slide_name in os.listdir(root_dir):
        if slide_name not in blur_dict:
            blur_dict[slide_name] = {}
        slide_dir = os.path.join(root_dir, slide_name)
        for layer in tqdm(os.listdir(slide_dir), desc="Processing slide {}".format(slide_name)):
            if layer not in target_layers:
                continue
            layer_dir = os.path.join(slide_dir, layer)
            outputs = Parallel(n_jobs=n_jobs)(delayed(calc_score)(
                layer_dir, patch_name) for patch_name in os.listdir(layer_dir))
            for patch_name, score2 in outputs:
                if patch_name not in blur_dict[slide_name]:
                    blur_dict[slide_name][patch_name] = {}
                blur_dict[slide_name][patch_name][layer] = score2
    with open("./blur_data.csv", "w") as wf:
        wf.write("slide_name,patch_name,{}\n".format(",".join(target_layers)))
        for slide_name, slide_data in blur_dict.items():
            for patch_name, patch_data in slide_data.items():
                calculated_layers = list(patch_data.keys())
                calculated_layers.sort()
                if len(calculated_layers) != len(target_layers):
                    print(slide_name, patch_name, "has {} layers only... skipped!".format(len(calculated_layers)))
                    continue
                scores = [str(s) for layer, s in sorted(patch_data.items())]
                scores = ",".join(scores)
                wf.write("{},{},{}\n".format(slide_name, patch_name, scores))

    # fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    # axs = axs.ravel()
    #
    # blur_data_9 = blur_data[np.where(clear_layer_np[np.where(train_val_splits == 0)[0]] == 9)[0]]
    # blur_data_9 = blur_data_9.reshape(-1, 9)
    # for layer in range(9):
    #     data = np.transpose(blur_data_9, (1,0))[layer]
    #     mu, sigma = norm.fit(data)
    #
    #     # Plot histogram of blur values
    #     axs[layer].hist(data, bins=20, density=True, alpha=0.6, color='b')
    #
    #     # Plot Gaussian fit
    #     x = np.linspace(min(data), max(data), 100)
    #     pdf = norm.pdf(x, mu, sigma)
    #     axs[layer].plot(x, pdf, 'r-', lw=2)
    #
    #     axs[layer].set_title(f'Layer {layers[layer]} - µ={mu:.2f}, σ={sigma:.2f}')
    #     axs[layer].set_xlabel('Blur Value')
    #     axs[layer].set_ylabel('Density')
    #
    # plt.tight_layout()
    # plt.show()
    #
