import argparse
import json
import os
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_result_dir",
        type=str,
        required=True,
        help="Path to the folder containing evaluated results",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output 3-in-1 scores",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(f"{args.input_result_dir}/annotation_blip/vqa_result.json", "r") as f:  # read attribute score
        attribute_score = json.load(f)
    with open(f"{args.input_result_dir}/annotation_obj_detection_2d/vqa_result.json", "r") as f:
        spatial_score = json.load(f)

    with open(f"{args.input_result_dir}/annotation_clip/vqa_result.json", "r") as f:  # read action score
        action_score = json.load(f)

    # change json to list
    attribute_score = [float(i["answer"]) for i in attribute_score]
    spatial_score = [float(i["answer"]) for i in spatial_score]
    action_score = [float(i["answer"]) for i in action_score]

    # merge score with weight
    with open(DATA_ROOT / "complex_val_spatial.txt", "r") as f:
        spatial = f.readlines()
        spatial = [i.strip("\n").split(".")[0].lower() for i in spatial]
    with open(DATA_ROOT / "complex_val_action.txt", "r") as f:
        action = f.readlines()
        action = [i.strip("\n").split(".")[0].lower() for i in action]
    with open(DATA_ROOT / "complex_val.txt", "r") as f:
        data = f.readlines()
        data = [i.strip("\n").split(".")[0].lower() for i in data]

    num = 10  # number of images for each prompt
    dataset_num = len(data)
    total_score = np.zeros(num * dataset_num)
    spatial_score = np.array(spatial_score)
    action_score = np.array(action_score)
    attribute_score = np.array(attribute_score)

    for i in range(dataset_num):
        if data[i] in spatial:  # contain spatial relation and attribute
            total_score[i * num : (i + 1) * num] = (
                spatial_score[i * num : (i + 1) * num] + attribute_score[i * num : (i + 1) * num]
            ) * 0.5
        elif data[i] in action:  # contain action relation and attribute
            total_score[i * num : (i + 1) * num] = (
                action_score[i * num : (i + 1) * num] + attribute_score[i * num : (i + 1) * num]
            ) * 0.5
        else:  ##contain spatial, action relation and attribute
            total_score[i * num : (i + 1) * num] = (
                attribute_score[i * num : (i + 1) * num]
                + spatial_score[i * num : (i + 1) * num]
                + action_score[i * num : (i + 1) * num]
            ) / 3

    total_score = total_score.tolist()

    result = []
    for i in range(num * dataset_num):
        result.append({"question_id": i, "answer": total_score[i]})

    os.makedirs(f"{args.out_dir}/annotation_3_in_1", exist_ok=True)
    with open(f"{args.out_dir}/annotation_3_in_1/vqa_result.json", "w") as f:
        json.dump(result, f)
    # calculate avg
    print("avg score:", sum(total_score) / len(total_score))
    with open(f"{args.out_dir}/annotation_3_in_1/score.txt", "w") as f:
        f.write("score:" + str(sum(total_score) / len(total_score)))


if __name__ == "__main__":
    main()
