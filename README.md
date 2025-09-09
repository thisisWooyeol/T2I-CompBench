# T2I-CompBench(++): Benchmark for Compositonal Text-to-Image Generation

<a color='red'>**Warning: This repository is not the official code release. It is AIBL style re-implemented version. Please refer to the official repository for the original code.**</a>

<a href='https://karine-h.github.io/T2I-CompBench-new/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://ieeexplore.ieee.org/abstract/document/10847875'><img src='https://img.shields.io/badge/T2I--CompBench++-Paper-red'></a>
<a href='https://arxiv.org/pdf/2307.06350v2'><img src='https://img.shields.io/badge/T2I--CompBench-Arxiv-red'></a>
<a href='https://connecthkuhk-my.sharepoint.com/:f:/g/personal/huangky_connect_hku_hk/Er_BhrcMwGREht6gnKGIErMB4egvvKM5ouhmkc0u5ZIKPw'><img src='https://img.shields.io/badge/Dataset-T2I--CompBench++-blue'></a>
<a href='https://connecthkuhk-my.sharepoint.com/:u:/g/personal/huangky_connect_hku_hk/EXEFBTzE6khPlsx2qPMjF9EBQYkE4WC2Z_XQGIjRUevjRQ'><img src='https://img.shields.io/badge/Dataset-Human eval images-purple'></a>

This repository contains the following papers:

> **T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation**<br> > [Kaiyi Huang](https://scholar.google.com/citations?user=dB86D_cAAAAJ&hl=zh-CN&oi=sra)<sup>1</sup>, [Kaiyue Sun](https://scholar.google.com/citations?user=mieuBzUAAAAJ&hl=zh-CN&oi=sra)<sup>1</sup>, [Enze Xie](https://xieenze.github.io/)<sup>2</sup>, [Zhenguo Li](https://scholar.google.com.sg/citations?user=XboZC1AAAAAJ&hl=en)<sup>2</sup>, [Xihui Liu](https://xh-liu.github.io/)<sup>1+</sup><br> > <sup>1</sup>The University of Hong Kong, <sup>2</sup>Huawei Noah’s Ark Lab<br>
> Conference on Neural Information Processing Systems (**Neurips**), 2023

> **T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation**<br> > [Kaiyi Huang](https://scholar.google.com/citations?user=dB86D_cAAAAJ&hl=zh-CN&oi=sra)<sup>1</sup>, [Chengqi Duan](https://scholar.google.com/citations?user=r9qb4ZwAAAAJ&hl=zh-CN&oi=sra)<sup>1,3</sup>, [Kaiyue Sun](https://scholar.google.com/citations?user=mieuBzUAAAAJ&hl=zh-CN&oi=sra)<sup>1</sup>, [Enze Xie](https://xieenze.github.io/)<sup>2</sup>, [Zhenguo Li](https://scholar.google.com.sg/citations?user=XboZC1AAAAAJ&hl=en)<sup>2</sup>, [Xihui Liu](https://xh-liu.github.io/)<sup>1+</sup><br> > <sup>1</sup>The University of Hong Kong, <sup>2</sup>Huawei Noah’s Ark Lab, <sup>3</sup>Tsinghua University<br>
> IEEE Transactions on Pattern Analysis and Machine Intelligence (**TPAMI**), 2025

## Installing the dependencies

```bash
uv sync
source .venv/bin/activate
accelerate config
```

And if you want to evaluate 2D/3D-spatial relationships and numeracy, please download UniDet weights:

```bash
mkdir -p UniDet_eval/experts/expert_weights
cd UniDet_eval/experts/expert_weights
wget https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt
pip install gdown
gdown https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq
```

## Evaluation

For evaluation, the directory structure of your input image files should be like this:

```
/path/to/your/images
    /color
        {prompt_idx}_{iteration}_{level}_prompt1.jpg
        {prompt_idx}_{iteration}_{level}_prompt2.jpg
        ...
    /texture
        {prompt_idx}_{iteration}_{level}_prompt1.jpg
        {prompt_idx}_{iteration}_{level}_prompt2.jpg
        ...
```

The evaluation result directory should include a json file named "vqa_result.json", and the json file should be a dictionary that maps from
`{"question_id", "answer"}`, e.g.,

```json
[{"question_id": 0, "answer": "0.6900"},
 {"question_id": 1, "answer": "0.7110"},
 ...]
```

The question*id is the \_prompt_idx* of the image file name, for example, _0_0_0_a green bench and a blue bowl.jpg_ is with question*id \_0*, and the evaluation score _0.6900_.

### How to run

```bash

METHOD_NAME="SD15"
IMAGE_ROOT_DIR="/path/to/your/images"
OUT_ROOT_DIR="output/${METHOD_NAME}"

# 1. BLIP-VQA evaluation (for color and texture)
# Currently we do not support shape evaluation

# Run BLIP-VQA evaluation on color
python src/BLIPvqa_eval/BLIP_vqa.py \
    --input_image_dir ${IMAGE_ROOT_DIR}/color \
    --out_dir ${OUT_ROOT_DIR}/color

# Run BLIP-VQA evaluation on texture
python src/BLIPvqa_eval/BLIP_vqa.py \
    --input_image_dir ${IMAGE_ROOT_DIR}/texture \
    --out_dir ${OUT_ROOT_DIR}/texture

# 2. UniDet evaluation (for numeracy)
# ! Currently we do not support 2D/3D-spatial relationship evaluation

python src/UniDet_eval/numeracy_eval.py \
    --input_image_dir ${IMAGE_ROOT_DIR}/numeracy \
    --out_dir ${OUT_ROOT_DIR}/numeracy
```
