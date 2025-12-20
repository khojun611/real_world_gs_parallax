import cv2
import numpy as np
import os

# ================= 설정 =================
# 경로 설정 (사용자 경로 유지)
GT_PATH = '/home/iris/mount1/marigold/pcc_monkey/images'
INF_PATH = '/home/iris/mount1/ref14/real_world_gs/output/pcc_monkey_wopcc/test/renders_infinite/rgb'
PARA_PATH = '/home/iris/mount1/ref14/real_world_gs/output/pcc_monkey3/test/renders_parallax/rgb'
OUTPUT_DIR = './edge_monkey'

# Canny Edge Detection 파라미터
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 150

# ================= 함수 정의 =================
def create_edge_overlay(img_gt_path, img_pred_path, save_path):
    # 이미지 로드 (경로에 한글이 없다고 가정, cv2.imread 사용)
    gt_img = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(img_pred_path, cv2.IMREAD_GRAYSCALE)

    # 이미지가 제대로 로드되었는지 확인
    if gt_img is None:
        print(f"[Error] GT 이미지를 읽을 수 없습니다: {img_gt_path}")
        return
    if pred_img is None:
        print(f"[Error] 예측 이미지를 읽을 수 없습니다: {img_pred_path}")
        return

    # 크기가 다를 경우를 대비해 리사이즈 (선택 사항, 보통 3DGS 렌더링은 크기가 같음)
    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    # Canny Edge 추출
    edges_gt = cv2.Canny(gt_img, LOW_THRESHOLD, HIGH_THRESHOLD)
    edges_pred = cv2.Canny(pred_img, LOW_THRESHOLD, HIGH_THRESHOLD)

    # 컬러 채널 생성 (검은 배경)
    height, width = gt_img.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # GT 에지를 초록색(Green) 채널에 할당 (BGR: 1번 인덱스)
    overlay[:, :, 1] = edges_gt

    # 예측 에지를 자홍색(Magenta = Blue + Red) 채널에 할당 (BGR: 0, 2번 인덱스)
    overlay[:, :, 0] = edges_pred # Blue
    overlay[:, :, 2] = edges_pred # Red

    # 저장
    cv2.imwrite(save_path, overlay)
    print(f"[Saved] {os.path.basename(save_path)}")

# ================= 메인 실행 로직 =================
def main():
    # 결과 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. GT 폴더의 파일 리스트 가져오기 (이미지 파일만 필터링)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(GT_PATH) if f.lower().endswith(valid_extensions)]
    files.sort()  # 순서대로 처리

    print(f"총 처리할 파일 수: {len(files)}개")

    for filename in files:
        # 2. 각 폴더의 전체 경로(Full Path) 생성
        gt_full_path = os.path.join(GT_PATH, filename)
        inf_full_path = os.path.join(INF_PATH, filename)
        para_full_path = os.path.join(PARA_PATH, filename)

        # 파일명만 추출 (확장자 제외) - 저장 파일명 생성용
        name_only = os.path.splitext(filename)[0]

        # 3. Infinite 폴더에 해당 파일이 존재하는지 확인 후 처리
        if os.path.exists(inf_full_path):
            save_name = f"edge_overlap_infinite_{name_only}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            create_edge_overlay(gt_full_path, inf_full_path, save_path)
        else:
            print(f"[Skip] Infinite 폴더에 파일 없음: {filename}")

        # 4. Parallax 폴더에 해당 파일이 존재하는지 확인 후 처리
        if os.path.exists(para_full_path):
            save_name = f"edge_overlap_parallax_{name_only}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            create_edge_overlay(gt_full_path, para_full_path, save_path)
        else:
            print(f"[Skip] Parallax 폴더에 파일 없음: {filename}")

    print("\n모든 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()