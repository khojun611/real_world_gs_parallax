import cv2
import numpy as np
import os

# ================= 설정 =================
# 경로 설정 (사용자 경로)
GT_PATH = '/home/iris/mount1/marigold/pcc_monkey/images'
INF_PATH = '/home/iris/mount1/ref14/real_world_gs/output/pcc_monkey_wopcc512/test/renders_infinite/rgb'
PARA_PATH = '/home/iris/mount1/ref14/real_world_gs/output/pcc_monkey3/test/renders_parallax/rgb'
OUTPUT_DIR = './heatmap_monkey'

# Canny Edge 파라미터
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 150

# 히트맵 민감도 설정 (최대 오차 거리 픽셀)
# 이 값보다 멀리 떨어지면 가장 빨간색으로 표시됨
MAX_DIST_CAP = 20.0 

# ================= 함수 정의 =================
def create_edge_distance_heatmap(img_gt_path, img_pred_path, save_path):
    # 이미지 로드
    gt_img = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(img_pred_path, cv2.IMREAD_GRAYSCALE)

    if gt_img is None or pred_img is None:
        return

    # 크기 맞추기
    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

    # 1. Canny Edge 추출
    edges_gt = cv2.Canny(gt_img, LOW_THRESHOLD, HIGH_THRESHOLD)
    edges_pred = cv2.Canny(pred_img, LOW_THRESHOLD, HIGH_THRESHOLD)

    # 2. GT 에지에 대한 Distance Transform 계산
    # (에지는 255(흰색), 배경은 0(검은색)이므로 반전시켜야 함)
    # distanceTransform은 '0'인 픽셀까지의 거리를 계산함. 따라서 에지를 '0'으로 만듦.
    edges_gt_inv = cv2.bitwise_not(edges_gt)
    
    # dist_map: 모든 픽셀에서 '가장 가까운 GT 에지'까지의 거리 (실수형)
    dist_map = cv2.distanceTransform(edges_gt_inv, cv2.DIST_L2, 5)

    # 3. 예측된 에지가 있는 위치의 거리값만 가져오기 (마스킹)
    # 예측 에지가 없는 곳(배경)은 볼 필요 없으므로 0으로 만듦
    # edges_pred가 255인 곳만 dist_map 값을 유지
    error_map = np.where(edges_pred > 0, dist_map, 0)

    # 4. 시각화를 위한 정규화 (0~255)
    # MAX_DIST_CAP(예: 20픽셀) 이상 틀린 건 그냥 다 똑같이 빨간색으로 처리
    error_map_clipped = np.clip(error_map, 0, MAX_DIST_CAP)
    
    # 0(정확) -> 255(틀림)으로 정규화
    heatmap_norm = (error_map_clipped / MAX_DIST_CAP * 255).astype(np.uint8)

    # 5. 컬러맵 적용 (JET: 파랑=낮음, 빨강=높음)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # 배경을 검은색으로 만들기 (에지가 없는 부분은 검은색 유지)
    # applyColorMap은 0인 부분도 파란색으로 칠해버리므로, 에지가 없는 부분은 다시 검게 칠함
    final_output = np.where(edges_pred[:, :, None] > 0, heatmap_color, 0)

    # 저장
    cv2.imwrite(save_path, final_output)
    
    # 평균 오차 거리 계산 (참고용)
    # 예측된 에지 픽셀들의 평균 거리 오차
    num_pred_pixels = np.count_nonzero(edges_pred)
    if num_pred_pixels > 0:
        mean_error = np.sum(error_map) / num_pred_pixels
        print(f"[Saved] {os.path.basename(save_path)} | Avg Edge Error: {mean_error:.2f} px")
    else:
        print(f"[Saved] {os.path.basename(save_path)} | No edges detected")

# ================= 메인 실행 =================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(GT_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    files.sort()

    print(f"Processing {len(files)} images...")

    for filename in files:
        gt_full_path = os.path.join(GT_PATH, filename)
        inf_full_path = os.path.join(INF_PATH, filename)
        para_full_path = os.path.join(PARA_PATH, filename)
        name_only = os.path.splitext(filename)[0]

        # Infinite 처리
        if os.path.exists(inf_full_path):
            save_path = os.path.join(OUTPUT_DIR, f"heatmap_infinite_{name_only}.jpg")
            create_edge_distance_heatmap(gt_full_path, inf_full_path, save_path)

        # Parallax 처리
        if os.path.exists(para_full_path):
            save_path = os.path.join(OUTPUT_DIR, f"heatmap_parallax_{name_only}.jpg")
            create_edge_distance_heatmap(gt_full_path, para_full_path, save_path)

    print("Done.")

if __name__ == "__main__":
    main()