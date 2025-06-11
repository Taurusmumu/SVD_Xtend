from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root_dir = "/ssd2/AMC_zstack_2_patches/pngs_mid"
image_info_path = "/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt"


def find_valid_patches(blur_data_df, blur_scores, threshold, consecutive_required):
    kernel = np.ones(consecutive_required, dtype=int)
    below_threshold = blur_scores < threshold
    # Convolve each row with kernel to find consecutive sequences quickly
    consecutive_counts = convolve2d(below_threshold, kernel[None, :], mode='valid')
    # Check if any position in the patch satisfies the consecutive criteria
    valid_patches = np.any(consecutive_counts >= 10, axis=1)
    valid_patches_indices = np.where(valid_patches)[0]
    blur_data_df = blur_data_df.iloc[valid_patches_indices]
    blur_scores = blur_scores[valid_patches_indices]

    min_counts = convolve2d(blur_scores, kernel[None, :], mode='valid')
    start_indices = np.argmin(min_counts, axis=1)
    min_indices = np.argmin(blur_scores, axis=1)

    return blur_data_df, blur_scores, start_indices, min_indices


if __name__ == "__main__":

    # layers = ["z01", "z03", "z05", "z07", "z09", "z11", "z13", "z15", "z17"]
    # layers = ["z00", "z03", "z06", "z09", "z12", "z15", "z18"]
    layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]
    threshold = 0.4
    consecutive_layers_required = 14
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
    blur_scores = blur_data_df.loc[np.where(~((np.min(blur_scores, axis=1) > 1) | (np.min(blur_scores, axis=1) < 0.6)))[0]].loc[:, layers]
    blur_scores = np.array(blur_scores) # 0.7757876530616586 0.04739573415277596

    blur_scores_min = np.min(blur_scores, axis=1)
    plt.figure(figsize=(8, 5))
    sns.histplot(blur_scores_min, bins=50, kde=True, color='steelblue')
    plt.title("Distribution of Minimum Blur Values per Image")
    plt.xlabel("Minimum Blur Score")
    plt.ylabel("Number of 3D Images")
    plt.grid(True)
    plt.show()
    min_mean = np.mean(blur_scores_min)
    min_std = np.std(blur_scores_min)
    print(min_mean)
    print(min_std)

    blur_scores = blur_data_df.loc[:, layers]
    blur_scores = np.array(blur_scores)
    blur_data_df = blur_data_df.loc[np.where(~((np.min(blur_scores, axis=1) > (min_mean + 2*min_std)) | (np.min(blur_scores, axis=1) < (min_mean - 2*min_std))))[0]]
    blur_scores = blur_data_df.loc[:, layers]
    blur_scores = np.array(blur_scores)
    blur_scores_min = np.min(blur_scores, axis=1)[:, None]
    blur_scores_norm = np.sqrt(np.square(blur_scores) - np.square(blur_scores_min))

    # Execute the function
    blur_data_df, blur_scores_norm, start_indices, min_indices = find_valid_patches(blur_data_df, blur_scores_norm, threshold, consecutive_layers_required)
    for i, layer in enumerate(layers):
        blur_data_df[layer] = blur_scores_norm[:, i]

    blur_data_df["start_indices"] = start_indices
    blur_data_df["min_indices"] = min_indices
    blur_data_df.to_csv("./blur_data3.csv", index=False)


