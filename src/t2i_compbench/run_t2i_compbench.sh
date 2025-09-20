#!/bin/bash

# Help function
show_help() {
    echo "Usage: $0 IMAGE_ROOT_DIR [METHOD_NAME] [OUTPUT_ROOT_DIR]"
    echo ""
    echo "Run the T2I-CompBench evaluation for a given method and image directory."
    echo ""
    echo "Arguments:"
    echo "  IMAGE_ROOT_DIR   The root directory containing the generated images, organized by task subdirectories."
    echo "  METHOD_NAME      (Optional) The name of the method being evaluated. If not provided, uses the basename of IMAGE_ROOT_DIR."
    echo "  OUTPUT_ROOT_DIR  (Optional) The root directory for output results. Defaults to 'output'."
    echo ""
    echo "Expected directory structure for IMAGE_ROOT_DIR:"
    echo "  IMAGE_ROOT_DIR/"
    echo "    ├── color/"
    echo "    │   ├── {prompt_idx}_{iteration}_{level}_prompt1.jpg"
    echo "    │   └── ..."
    echo "    ├── texture/"
    echo "    │   ├── {prompt_idx}_{iteration}_{level}_prompt1.jpg"
    echo "    │   └── ..."
    echo "    ├── numeracy/"
    echo "    │   ├── {prompt_idx}_{iteration}_{level}_prompt1.jpg"
    echo "    │   └── ..."
    echo "    └── complex/"
    echo "        ├── {prompt_idx}_{iteration}_{level}_prompt1.jpg"
    echo "        └── ..."
    echo ""
    echo "Options:"
    echo "  --help, -h       Show this help message and exit."
    echo "  --skip-color     Skip color evaluation"
    echo "  --skip-texture   Skip texture evaluation"
    echo "  --skip-numeracy  Skip numeracy evaluation"
    echo "  --skip-complex   Skip complex evaluation"
}

# Initialize flags
SKIP_COLOR=false
SKIP_TEXTURE=false
SKIP_NUMERACY=false
SKIP_COMPLEX=false

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --skip-color)
            SKIP_COLOR=true
            shift
            ;;
        --skip-texture)
            SKIP_TEXTURE=true
            shift
            ;;
        --skip-numeracy)
            SKIP_NUMERACY=true
            shift
            ;;
        --skip-complex)
            SKIP_COMPLEX=true
            shift
            ;;
        -*)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Check for required arguments
if [ -z "$1" ]; then
    echo "Error: IMAGE_ROOT_DIR is a required argument."
    show_help
    exit 1
fi

IMAGE_ROOT_DIR=$1
METHOD_NAME=${2:-$(basename "$IMAGE_ROOT_DIR")}
OUTPUT_ROOT_DIR=${3:-"output"}

T2I_COMPBENCH_ROOT_DIR=$(dirname "$0")

echo "Running T2I-CompBench evaluation with method: $METHOD_NAME"
echo "Image root directory: $IMAGE_ROOT_DIR"
echo "Output root directory: $OUTPUT_ROOT_DIR"
echo "Method name extracted from: $(basename "$IMAGE_ROOT_DIR")"

# Create output directory for the method
OUT_ROOT_DIR="${OUTPUT_ROOT_DIR}/${METHOD_NAME}"
mkdir -p "$OUT_ROOT_DIR"

# Check which task directories exist
TASKS=("color" "texture" "numeracy" "complex")
AVAILABLE_TASKS=()

for TASK in "${TASKS[@]}"; do
    TASK_DIR="${IMAGE_ROOT_DIR}/${TASK}"
    if [ -d "$TASK_DIR" ]; then
        echo "Found task directory: $TASK_DIR"
        AVAILABLE_TASKS+=("$TASK")
    else
        echo "Task directory not found: $TASK_DIR"
    fi
done

if [ ${#AVAILABLE_TASKS[@]} -eq 0 ]; then
    echo "No task directories found. Exiting."
    exit 1
fi

echo ""
echo "Available tasks: ${AVAILABLE_TASKS[@]}"
echo ""

# Function to check if command was successful
check_command() {
    if [ $? -eq 0 ]; then
        echo "✓ $1 completed successfully"
        return 0
    else
        echo "✗ Error: $1 failed"
        return 1
    fi
}


# Download UniDet expert weights if not already present
if [ ! -d "${T2I_COMPBENCH_ROOT_DIR}/UniDet_eval/experts/expert_weights" ]; then
    echo "UniDet expert weights not found. Downloading..."
    mkdir -p "${T2I_COMPBENCH_ROOT_DIR}/UniDet_eval/experts/expert_weights"
    cd "${T2I_COMPBENCH_ROOT_DIR}/UniDet_eval/experts/expert_weights"
    wget https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth
    wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt
    gdown https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq
    cd -
    
    echo "UniDet expert weights downloaded."
else
    echo "UniDet expert weights already present."
fi

# Run evaluation for each available task
for TASK in "${AVAILABLE_TASKS[@]}"; do
    echo ""
    echo "=== Processing $TASK task ==="
    
    case $TASK in
        "color")
            if [ "$SKIP_COLOR" = true ]; then
                echo "Skipping color evaluation (--skip-color flag set)"
                continue
            fi
            
            echo "Running BLIP-VQA evaluation on color..."
            python "${T2I_COMPBENCH_ROOT_DIR}/BLIPvqa_eval/BLIP_vqa.py" \
                --input_image_dir "${IMAGE_ROOT_DIR}/color" \
                --out_dir "${OUT_ROOT_DIR}/color"
            check_command "Color evaluation"
            ;;
            
        "texture")
            if [ "$SKIP_TEXTURE" = true ]; then
                echo "Skipping texture evaluation (--skip-texture flag set)"
                continue
            fi
            
            echo "Running BLIP-VQA evaluation on texture..."
            python "${T2I_COMPBENCH_ROOT_DIR}/BLIPvqa_eval/BLIP_vqa.py" \
                --input_image_dir "${IMAGE_ROOT_DIR}/texture" \
                --out_dir "${OUT_ROOT_DIR}/texture"
            check_command "Texture evaluation"
            ;;
            
        "numeracy")
            if [ "$SKIP_NUMERACY" = true ]; then
                echo "Skipping numeracy evaluation (--skip-numeracy flag set)"
                continue
            fi
            
            echo "Running UniDet evaluation on numeracy..."
            python "${T2I_COMPBENCH_ROOT_DIR}/UniDet_eval/numeracy_eval.py" \
                --input_image_dir "${IMAGE_ROOT_DIR}/numeracy" \
                --out_dir "${OUT_ROOT_DIR}/numeracy"
            check_command "Numeracy evaluation"
            ;;
            
        "complex")
            if [ "$SKIP_COMPLEX" = true ]; then
                echo "Skipping complex evaluation (--skip-complex flag set)"
                continue
            fi
            
            echo "Running complex evaluation (3-in-1)..."
            
            # Step 1: BLIP-VQA evaluation with complex task images
            echo "  Step 1/4: Running BLIP-VQA evaluation..."
            python "${T2I_COMPBENCH_ROOT_DIR}/BLIPvqa_eval/BLIP_vqa.py" \
                --input_image_dir "${IMAGE_ROOT_DIR}/complex" \
                --out_dir "${OUT_ROOT_DIR}/complex"
            check_command "Complex BLIP-VQA evaluation" || continue
            
            # Step 2: CLIPScore evaluation with complex task images
            echo "  Step 2/4: Running CLIPScore evaluation..."
            python "${T2I_COMPBENCH_ROOT_DIR}/CLIPScore_eval/CLIP_similarity.py" \
                --input_image_dir "${IMAGE_ROOT_DIR}/complex" \
                --out_dir "${OUT_ROOT_DIR}/complex" \
                --complex
            check_command "Complex CLIPScore evaluation" || continue
            
            # Step 3: 2D-spatial relationship evaluation with complex task images
            echo "  Step 3/4: Running 2D-spatial relationship evaluation..."
            python "${T2I_COMPBENCH_ROOT_DIR}/UniDet_eval/2D_spatial_eval.py" \
                --input_image_dir "${IMAGE_ROOT_DIR}/complex" \
                --out_dir "${OUT_ROOT_DIR}/complex" \
                --complex
            check_command "Complex 2D-spatial evaluation" || continue
            
            # Step 4: 3-in-1 score calculation
            echo "  Step 4/4: Running 3-in-1 score calculation..."
            python "${T2I_COMPBENCH_ROOT_DIR}/3_in_1_eval/3_in_1.py" \
                --input_result_dir "${OUT_ROOT_DIR}/complex" \
                --out_dir "${OUT_ROOT_DIR}/complex"
            check_command "Complex 3-in-1 score calculation"
            ;;
            
        *)
            echo "Unknown task: $TASK"
            ;;
    esac
done

echo ""
echo "=== T2I-CompBench evaluation completed for method: $METHOD_NAME ==="
echo "Results saved in: $OUT_ROOT_DIR"

# Generate summary of results
echo ""
echo "=== Results Summary ==="
for TASK in "${AVAILABLE_TASKS[@]}"; do
    TASK_OUTPUT_DIR="${OUT_ROOT_DIR}/${TASK}"
    if [ -d "$TASK_OUTPUT_DIR" ]; then
        echo "Task: $TASK"
        
        # Look for common result files
        if [ -f "${TASK_OUTPUT_DIR}/annotation_blip/score.txt" ]; then
            echo "  BLIP score: $(cat ${TASK_OUTPUT_DIR}/annotation_blip/score.txt)"
        fi
        if [ -f "${TASK_OUTPUT_DIR}/annotation_num/score.txt" ]; then
            echo "  Numeracy score: $(cat ${TASK_OUTPUT_DIR}/annotation_num/score.txt)"
        fi
        if [ -f "${TASK_OUTPUT_DIR}/annotation_clip/score.txt" ]; then
            echo "  CLIP score: $(cat ${TASK_OUTPUT_DIR}/annotation_clip/score.txt)"
        fi
        if [ -f "${TASK_OUTPUT_DIR}/annotation_obj_detection_2d/score.txt" ]; then
            echo "  2D-spatial score: $(cat ${TASK_OUTPUT_DIR}/annotation_obj_detection_2d/score.txt)"
        fi
        if [ -f "${TASK_OUTPUT_DIR}/annotation_3_in_1/score.txt" ]; then
            echo "  3-in-1 score: $(cat ${TASK_OUTPUT_DIR}/annotation_3_in_1/score.txt)"
        fi
        echo ""
    fi
done

echo "Evaluation script completed."
