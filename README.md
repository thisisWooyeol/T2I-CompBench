# How to run

```bash

METHOD_NAME="SD15"
IMAGE_ROOT_DIR="/path/to/your/images"
OUT_ROOT_DIR="output/${METHOD_NAME}"

# Run BLIP-VQA evaluation on color
python src/BLIPvqa_eval/BLIP_vqa.py \
    --input_image_dir ${IMAGE_ROOT_DIR}/color \
    --out_dir ${OUT_ROOT_DIR}/color

# Run BLIP-VQA evaluation on texture
python src/BLIPvqa_eval/BLIP_vqa.py \
    --input_image_dir ${IMAGE_ROOT_DIR}/texture \
    --out_dir ${OUT_ROOT_DIR}/texture

# Currently we do not support shape evaluation
```
