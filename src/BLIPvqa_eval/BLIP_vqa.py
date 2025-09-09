import argparse
import json
import os

import spacy
import torch
from tqdm import tqdm

from BLIPvqa_eval.BLIP.train_vqa_func import VQA_main


def Create_annotation_for_BLIP(image_folder, outpath, np_index=None):
    nlp = spacy.load("en_core_web_sm")

    annotations = []
    file_names = os.listdir(image_folder)
    # filename pattern: f"{idx}_{itr}_{level}_" + prompt.replace(" ", "_").jpg
    file_names.sort(key=lambda x: int(x.split("_")[0]))  # sort by idx (first part)

    cnt = 0

    # output annotation.json
    for file_name in file_names:
        image_dict = {}
        image_dict["image"] = image_folder + file_name
        image_dict["question_id"] = cnt
        # Extract prompt from filename: remove idx_itr_level_ prefix and .jpg suffix, then replace _ with spaces
        parts = file_name.split("_")
        if len(parts) >= 4:  # Should have at least idx_itr_level_prompt
            # Join all parts after the first 3 (idx, itr, level) and before the file extension
            prompt_part = "_".join(parts[3:])
            # Remove file extension
            prompt_part = prompt_part.rsplit(".", 1)[0]
            # Replace underscores with spaces to restore original prompt
            f = prompt_part.replace("_", " ")
        else:
            # Fallback: use entire filename without extension
            f = file_name.rsplit(".", 1)[0]
        doc = nlp(f)

        noun_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.text not in ["top", "the side", "the left", "the right"]:  # todo remove some phrases
                noun_phrases.append(chunk.text)
        if np_index is not None and len(noun_phrases) > np_index:
            q_tmp = noun_phrases[np_index]
            image_dict["question"] = f"{q_tmp}?"
        else:
            image_dict["question"] = ""

        image_dict["dataset"] = "color"
        cnt += 1

        annotations.append(image_dict)

    print("Number of Processed Images:", len(annotations))

    json_file = json.dumps(annotations)
    with open(f"{outpath}/vqa_test.json", "w") as f:
        f.write(json_file)


def parse_args():
    parser = argparse.ArgumentParser(description="BLIP vqa evaluation.")
    parser.add_argument(
        "--input_image_dir",
        type=str,
        required=True,
        help="Path to the folder containing images to be evaluated",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output BLIP vqa score",
    )
    parser.add_argument(
        "--np_num",
        type=int,
        default=8,
        help="Noun phrase number, can be greater or equal to the actual noun phrase number",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    np_index = args.np_num  # how many noun phrases
    out_dir = args.out_dir

    answer = []
    sample_num = len(os.listdir(args.input_image_dir))
    reward = torch.zeros((sample_num, np_index)).to(device="cuda")

    order = "_blip"  # rename file
    for i in tqdm(range(np_index)):
        print(f"start VQA{i + 1}/{np_index}!")
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}", exist_ok=True)
        os.makedirs(f"{out_dir}/annotation{i + 1}{order}/VQA/", exist_ok=True)
        Create_annotation_for_BLIP(
            args.input_image_dir,
            f"{out_dir}/annotation{i + 1}{order}",
            np_index=i,
        )
        answer_tmp = VQA_main(f"{out_dir}/annotation{i + 1}{order}/", f"{out_dir}/annotation{i + 1}{order}/VQA/")
        answer.append(answer_tmp)

        with open(f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
            r = json.load(file)
        with open(f"{out_dir}/annotation{i + 1}{order}/vqa_test.json", "r") as file:
            r_tmp = json.load(file)
        for k in range(len(r)):
            if r_tmp[k]["question"] != "":
                reward[k][i] = float(r[k]["answer"])
            else:
                reward[k][i] = 1
        print(f"end VQA{i + 1}/{np_index}!")
    reward_final = reward[:, 0]
    for i in range(1, np_index):
        reward_final *= reward[:, i]

    # output final json
    with open(f"{out_dir}/annotation{np_index}{order}/VQA/result/vqa_result.json", "r") as file:
        r = json.load(file)
    reward_after = 0
    for k in range(len(r)):
        r[k]["answer"] = "{:.4f}".format(reward_final[k].item())
        reward_after += float(r[k]["answer"])
    os.makedirs(f"{out_dir}/annotation{order}", exist_ok=True)
    with open(f"{out_dir}/annotation{order}/vqa_result.json", "w") as file:
        json.dump(r, file)

    # calculate avg of BLIP-VQA as BLIP-VQA score
    print("BLIP-VQA score:", reward_after / len(r), "!\n")
    with open(f"{out_dir}/annotation{order}/blip_vqa_score.txt", "w") as file:
        file.write("BLIP-VQA score:" + str(reward_after / len(r)))


if __name__ == "__main__":
    main()
