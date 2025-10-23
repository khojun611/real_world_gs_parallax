#!/bin/bash

# 평가할 모델들이 들어있는 최상위 폴더를 지정합니다.
OUTPUT_DIR="output"

# 최상위 폴더가 존재하는지 확인
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "오류: '$OUTPUT_DIR' 폴더를 찾을 수 없습니다. 스크립트를 종료합니다."
    exit 1
fi

echo "Starting evaluation for all models found in '$OUTPUT_DIR'..."
echo ""

# 'output' 폴더 안의 각 하위 폴더(예: output/white_car)에 대해 루프를 실행합니다.
for scene_dir in "$OUTPUT_DIR"/*; do
    # 해당 경로가 실제 디렉토리인지 확인합니다.
    if [ -d "$scene_dir" ]; then
        echo "===================================================="
        echo "Processing scene directory: ${scene_dir}"

        # 해당 씬 폴더 안에서 가장 최근에 수정된 모델 폴더(타임스탬프 폴더)를 찾습니다.
        # ls -td: 시간순(최신순)으로 폴더 정렬 | head -n 1: 첫 번째 줄(가장 최신 폴더) 선택
        latest_model_path=$(ls -td "${scene_dir}"/*/ | head -n 1)

        # 모델 폴더를 찾았는지 확인합니다.
        if [ -z "$latest_model_path" ]; then
            echo "Warning: No trained model subdirectory found in '${scene_dir}'. Skipping."
            continue
        fi
        
        # 경로 끝에 붙은 '/' 문자 제거
        latest_model_path=${latest_model_path%/}

        echo "Found latest model to evaluate: ${latest_model_path}"
        echo "Starting evaluation..."
        python -u eval.py --model_path "$latest_model_path"

        if [ $? -eq 0 ]; then
            echo "Evaluation for ${latest_model_path} completed successfully."
        else
            echo "ERROR: Evaluation failed for ${latest_model_path}."
        fi
        echo "----------------------------------------------------"
    fi
done

echo "===================================================="
echo "All evaluations are complete."