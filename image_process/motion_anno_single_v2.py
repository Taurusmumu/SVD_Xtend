import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
import skimage.io
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def estimate_motion(clear_layer, slide_name, patch_name, target_layers, root_dir):
    try:
        frame1_path = os.path.join(root_dir, slide_name, clear_layer, patch_name)
        frame1 = cv2.imread(frame1_path)
        if frame1 is None:
            print(f"Error: Could not read reference frame: {frame1_path}. Skipping patch.")
            return None

        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        mag_list = []
        # print("Clear layer is: ", clear_layer)
        # print(f"Aligning all layers to reference: {clear_layer}/{patch_name}")

        for l_idx, next_layer in enumerate(target_layers):
            frame2_path = os.path.join(root_dir, slide_name, next_layer, patch_name)
            # try:
            frame2 = cv2.imread(frame2_path)
            if frame2 is None:
                print(f"  - Warning: Missing frame for layer {next_layer}. Appending score -1.")
                mag_list.append(-1)
                continue
            # Convert to grayscale
            # Create a grayscale version specifically forframe1 registration
            next_frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame_gray, None, 0.5, 3, 8, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag[np.where(mag == np.inf)] = 0
            mag_list.append(float(np.mean(mag)))

        return mag_list

    except Exception as e:
        print(f"Error processing {slide_name} {patch_name}: {e}")
        return None


if __name__ == '__main__':

    anno_path = "./blur_motion_data3.csv"
    root_dir = "/ssd2/AMC_zstack_2_patches_warp/pngs_mid"
    text_file_path = "/ssd2/AMC_zstack_2_patches/base_sudo_anno.txt"
    target_layers = ["z00", "z01", "z02", "z03", "z04", "z05", "z06", "z07", "z08", "z09",
                     "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18"]
    # import pandas as pd
    # data_hist = []
    # bool_data = []
    # # df = pd.read_csv("./blur_motion_data4.csv")
    # df = pd.read_csv("./motion_annotations.csv")
    # for index, row in df.iterrows():
    #     # patch_name = row['patch_name']
    #     # slide_name = row['slide_name']
    #     # (start_layer, end_layer, clear_layer) = (row['start_indices'], row['end_indices'], row['min_indices'])
    #     # target_row = df1.loc[df1["patch_name"] == patch_name]
    #     # if len(target_row) == 0:
    #     #     continue
    #     motion_scores = row.loc[target_layers].values
    #     max_sum, flag = 0,0
    #     for i in range(len(motion_scores) - 11):
    #         sum = np.sum(motion_scores[i:i + 11])
    #         if sum > max_sum:
    #             max_sum = sum
    #             flag = i
    #     data_hist.append(motion_scores[flag:flag + 11])
    #
    # data_hist = np.array(data_hist).squeeze(1)
    # result = []
    # for i in range(data_hist.shape[1]):
    #     result.append(np.mean(data_hist[:, i][bool_data[:, i]]))
    #
    # plt.figure(figsize=(8, 5))
    # plt.bar([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], result)
    # plt.title('before wapring')
    # plt.xlabel('Layer')
    # plt.ylabel('Avg Motion Score')
    # plt.show()


    # n_jobs = mp.cpu_count() * 80 //

    print()

    # df = pd.read_csv("./blur_motion_data4.csv")
    # data_hist = []
    # bool_data = []
    # for i in range(len(df)):
    #     row = df.iloc[i]
    #     start_layer, end_layer, clear_layer = row["start_indices"], row["end_indices"], row["min_indices"]
    #     motion_blur_scores = row[target_layers]
    #     motion_scores = [float(motion_blur_scores[j].split(';')[0]) for j in range(len(motion_blur_scores))]
    #
    #     l0 = []
    #     b = []
    #     for idx in range(start_layer, end_layer + 1):
    #         if idx < 0 or idx >= len(target_layers) or motion_scores[idx] > 10:
    #             b.append(False)
    #             l0.append(0)
    #         else:
    #             b.append(True)
    #             l0.append(motion_scores[idx])
    #     data_hist.append(l0)
    #     bool_data.append(b)
    #
    # data_hist = np.array(data_hist)
    # bool_data = np.array(bool_data)
    #
    # result = []
    # for i in range(data_hist.shape[1]):
    #     result.append(np.mean(data_hist[:, i][bool_data[:, i]]))
    #
    # # [0.22329782038026075, 0.18766250639497062, 0.15883637795235564, 0.13247836067883415, 0.11504107804637423, 0.0010112351694749474, 0.11614735584420642, 0.1353599893403316, 0.16571900283566848, 0.2004983182774774, 0.2449976051975214]
    # plt.figure(figsize=(8, 5))
    # plt.bar([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], result)
    # plt.title('before wapring')
    # plt.xlabel('Layer')
    # plt.ylabel('Avg Motion Score')
    # plt.show()

    blur_degree_dict = {}
    start_layer_dict = {}
    motion_dict = {}

    flag = 0
    with open(anno_path, "r") as rf:
        rf.readline()

        for line in tqdm(rf.readlines(), desc="Processing data"):
            # flag += 1
            # if flag == 100:
            #     break
            line_split = line.strip().split(",")
            slide_name = line_split[0]
            patch_name = line_split[1]
            start_layer = int(line_split[-3])
            end_layer = int(line_split[-2])
            clear_layer = int(line_split[-1])

            result = estimate_motion(target_layers[clear_layer], slide_name, patch_name, target_layers, root_dir)
            if result is not None:
                if slide_name not in motion_dict:
                    start_layer_dict[slide_name] = {}
                    blur_degree_dict[slide_name] = {}
                    motion_dict[slide_name] = {}

                start_layer_dict[slide_name][patch_name] = (start_layer, end_layer, clear_layer)
                motion_blur_list = [score for score in line_split[2:-3]]
                blur_degree_dict[slide_name][patch_name] = [float(score.split(';')[1]) for score in motion_blur_list]
                motion_dict[slide_name][patch_name] = result

    with open("./blur_motion_data4.csv", "w") as wf:
        wf.write("slide_name,patch_name,{},start_indices,end_indices,min_indices\n".format(",".join(target_layers)))
        for slide_name, slide_data in blur_degree_dict.items():
            for patch_name, blur_scores in slide_data.items():
                motion_scores = motion_dict[slide_name][patch_name]
                (start_layer, end_layer, clear_layer) = start_layer_dict[slide_name][patch_name]
                combined_scores = []
                for l_idx, layer in enumerate(target_layers):
                    combined_scores.append(f"{motion_scores[l_idx]};{blur_scores[l_idx]}")
                scores = ",".join(combined_scores)
                wf.write("{},{},{},{},{},{}\n".format(slide_name, patch_name, scores, start_layer, end_layer, clear_layer))
