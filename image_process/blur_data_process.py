



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
import numpy as np
import pandas as pd

root_dir = "/ssd2/AMC_zstack_2_patches/pngs_mid"
image_info_path = "/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt"


def find_valid_patches(blur_data_df, blur_scores, threshold, consecutive_required):
    below_threshold = blur_scores < threshold
    kernel = np.ones(consecutive_required, dtype=int)

    # Convolve each row with kernel to find consecutive sequences quickly
    from scipy.signal import convolve2d
    consecutive_counts = convolve2d(below_threshold, kernel[None, :], mode='valid')

    # Check if any position in the patch satisfies the consecutive criteria
    valid_patches = np.any(consecutive_counts == consecutive_required, axis=1)
    valid_patches_indices = np.where(valid_patches)[0]

    blur_data_df = blur_data_df.iloc[valid_patches_indices]
    blur_scores = blur_scores[valid_patches_indices]
    min_counts = convolve2d(blur_scores, kernel[None, :], mode='valid')
    min_indices = np.argmin(min_counts, axis=1)

    # map_array = np.zeros_like(blur_scores)
    # for i in range(consecutive_required):
    #     map_array[:, min_indices + i] = 1


    return blur_data_df, min_indices


if __name__ == "__main__":

    # layers = ["z01", "z03", "z05", "z07", "z09", "z11", "z13", "z15", "z17"]
    # layers = ["z00", "z03", "z06", "z09", "z12", "z15", "z18"]
    layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]
    threshold = 0.5
    consecutive_layers_required = 9
    blur_data_df = pd.read_csv('./blur_data.csv')

    file = open(image_info_path, "r")
    lines = file.read()
    file.close()
    lines = lines.split('\n')
    slices = [line.split(',')[0] for line in lines]
    clear_layers = [line.split(',')[1] for line in lines]
    clear_layer_np = []
    for layer in clear_layers:
        if layer == "00":
            clear_layer_np.append(0)
        elif layer == "09":
            clear_layer_np.append(9)
        else:
            clear_layer_np.append(18)
    clear_layer_np = np.array(clear_layer_np)
    # train_val_splits = [line.split(',')[2] for line in lines]
    # train_val_splits = np.array([1 if s == "train" else 0 for s in train_val_splits])

    slices_0 = np.array(slices)[np.where(clear_layer_np == 0)[0]]
    slices_9 = np.array(slices)[np.where(clear_layer_np == 9)[0]]
    slices_18 = np.array(slices)[np.where(clear_layer_np == 18)[0]]

    # blur_data_df = blur_data_df.loc[~((blur_data_df["z02"] == -1) | (blur_data_df["z03"] == -1) | (blur_data_df["z04"] == -1))]
    # blur_data_df_9.loc[blur_data_df_9["patch_name"] == "patch_16799_32001_34820.png"]
    # blur_data_df_9 = blur_data_df.loc[blur_data_df["slide_name"].isin(slices_9)] # patch_3113_14444_27970
    blur_scores = blur_data_df.loc[:, layers]
    blur_scores = np.array(blur_scores)
    blur_data_df = blur_data_df.loc[np.where(~((np.min(blur_scores, axis=1) > 1) | (np.min(blur_scores, axis=1) < 0.7)))[0]]
    blur_scores = blur_scores[np.where(~((np.min(blur_scores, axis=1) > 1) | (np.min(blur_scores, axis=1) < 0.7)))[0]]

    blur_scores_min = np.min(blur_scores, axis=1)[:, None]
    blur_scores_norm = np.sqrt(np.square(blur_scores) - np.square(blur_scores_min))

    # Execute the function
    blur_data_df, min_indices = find_valid_patches(blur_data_df, blur_scores_norm, threshold, consecutive_layers_required)
    print(blur_data_df, min_indices)

    blur_data_df["min_indices"] = min_indices
    blur_data_df.to_csv("./blur_data1.csv", index=False)


    # blur_scores = blur_scores.reshape(-1, len(layers))
    #
    # fig, axs = plt.subplots(4, 4, figsize=(15, 12))
    # axs = axs.ravel()
    # for layer in range(len(layers)):
    #     data = np.transpose(blur_scores, (1,0))[layer]
    #     mu, sigma = norm.fit(data)
    #
    #     # Plot histogram of blur values
    #     axs[layer].hist(data, bins=100, density=True, alpha=0.6, color='b')
    #
    #     # Plot Gaussian fit
    #     x = np.linspace(0, 3, 100)
    #     pdf = norm.pdf(x, mu, sigma)
    #     axs[layer].plot(x, pdf, 'r-', lw=2)
    #
    #     axs[layer].set_title(f'Layer {layers[layer]} - µ={mu:.2f}, σ={sigma:.2f}')
    #     axs[layer].set_xlabel('Blur Value')
    #     axs[layer].set_ylabel('Density')
    #
    # plt.tight_layout()
    # plt.show()



