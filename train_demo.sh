#!/bin/bash

# ============================ 설정 (Configuration) ============================
# 사용자는 이 부분만 수정하면 됩니다.

# 1. 학습시킬 데이터셋 폴더들이 들어있는 '상위 폴더' 경로를 지정하세요.
#    예: /home/iris/datasets/realworld_scenes
DATA_PARENT_DIR="../3DGS-DR/data/GlossyReal"

# 2. 학습된 모델을 저장할 '출력 폴더' 경로를 지정하세요.
OUTPUT_PARENT_DIR="output"

# 3. 실행할 train.py 스크립트 경로
TRAIN_SCRIPT="train.py"

# ==============================================================================


# --- 스크립트 본문 (수정 필요 없음) ---

# 1. 데이터셋 상위 폴더가 존재하는지 확인
if [ ! -d "$DATA_PARENT_DIR" ]; then
    echo "오류: 데이터셋 폴더를 찾을 수 없습니다. 경로를 확인하세요: $DATA_PARENT_DIR"
    exit 1
fi

# 2. 출력 폴더가 없다면 생성
mkdir -p "$OUTPUT_PARENT_DIR"

# 3. 데이터셋 상위 폴더 안의 모든 하위 폴더에 대해 루프 실행
#    "*/" 패턴은 하위 디렉토리만 정확히 선택합니다.
for dataset_path in "$DATA_PARENT_DIR"/*/; do
    # 해당 경로가 실제 디렉토리인지 다시 한번 확인 (안전장치)
    if [ -d "$dataset_path" ]; then
        # 순수 데이터셋 폴더 이름 추출 (예: "white_car")
        dataset_name=$(basename "${dataset_path%/}")

        # 모델을 저장할 경로 생성 (예: "output/white_car_model")
        model_output_path="$OUTPUT_PARENT_DIR/${dataset_name}_model"

        echo "========================================================"
        echo "Starting training for: ${dataset_name}"
        echo " -> 데이터 경로: ${dataset_path}"
        echo " -> 모델 저장 경로: ${model_output_path}"
        echo "========================================================"

        # 4. 학습 스크립트 실행 (필요에 따라 옵션을 추가하거나 수정하세요)
        python "$TRAIN_SCRIPT" -s "$dataset_path" -m "$model_output_path" --eval --iterations 50000  --indirect_from_iter 10000 --lambda_normal_smooth 1.0

        # 직전 명령어의 성공 여부 확인 ($?는 0일 때 성공)
        if [ $? -eq 0 ]; then
            echo "Training for ${dataset_name} completed successfully."
        else
            echo "ERROR: Training for ${dataset_name} failed."
        fi
        echo "" # 가독성을 위한 줄바꿈
    fi
done

echo "모든 데이터셋에 대한 학습이 완료되었습니다."