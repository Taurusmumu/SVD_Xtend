
import json
import numpy as np

if __name__ == "__main__":

    file_path = '/ssd2/AMC_zstack_2_patches/output_for_metrics_prostate_harvard/unet_full_lora_v3_temp/50000_proj_v3/unet_full_lora_v3_temp_50000_proj_v3_eval_results.json'

    # file_path = '/ssd2/AMC_zstack_2_patches/output_for_metrics_aggc22/unet_full_lora_v3_temp/50000_proj/unet_full_lora_v3_temp_50000_proj_eval_results.json'

    clss = [0, 1, 2, 3]
    # clss = [1, 2, 3, 4, 5]
    with open(file_path, 'r') as file:
        data = json.load(file)

    keys = list(data.keys())

    for dimension in keys:
        if dimension == 'camera_motion':
            continue
        print(f"Dimension: {dimension}")
        dimension_data = data[dimension][1]
        if "image_path" in dimension_data[0]:
            cls_list = np.array([int(item["image_path"].split('.')[0][-1]) for item in dimension_data])
        else:
            cls_list = np.array([int(item["video_path"][-1]) for item in dimension_data])
        score_list = np.array([float(item["video_results"]) for item in dimension_data])
        scores = []
        for cls in clss:
            index = np.where(cls_list == cls)[0]
            scores.append(np.mean(score_list[index]))
        print(scores)
    print()
