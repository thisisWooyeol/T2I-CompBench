export project_dir="src/BLIPvqa_eval/"
cd $project_dir
out_dir="../examples/"
python BLIP_vqa.py --out_dir=$out_dir --input_image_dir="../input_images/"