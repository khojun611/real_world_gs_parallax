import os
import torch
import lpips
import cv2
import numpy as np
import csv  # CSV 저장을 위한 모듈 추가
from torchvision import transforms
from PIL import Image

# ================= 설정 =================
# 각 폴더의 경로를 입력하세요.
GT_DIR = '/home/iris/mount1/marigold/pcc_monkey/images'            # GT 이미지 폴더
INFINITE_DIR = '/home/iris/mount1/ref14/real_world_gs/output/pcc_monkey_wopcc512/test/renders_infinite/rgb' # Infinite 결과 폴더
PARALLAX_DIR = '/home/iris/mount1/ref14/real_world_gs/output/pcc_monkey3/test/renders_parallax/rgb' # Parallax 결과 폴더
OUTPUT_DIR = './lpips_monkey'     # 결과물이 저장될 폴더
CSV_FILENAME = 'lpips_monkey/lpips_scores.csv' # 저장될 CSV 파일 이름

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= 함수 정의 =================

def load_image_as_tensor(path):
    """이미지를 불러와 LPIPS 입력 형식([-1, 1])으로 변환"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [0,1] -> [-1,1]
    ])
    return transform(img).unsqueeze(0).to(device)

def save_heatmap(lpips_map_tensor, save_path):
    """LPIPS 맵 텐서를 히트맵 이미지로 변환하여 저장"""
    # 텐서를 numpy로 변환 (1, 1, H, W) -> (H, W)
    heatmap = lpips_map_tensor.squeeze().detach().cpu().numpy()
    
    # 시각화를 위해 정규화 (0 ~ 255)
    # 상대적 차이를 극대화해서 보여주기 위해 개별 이미지의 min/max 사용
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    
    # 히트맵 컬러 적용 (JET: 파랑-낮음, 빨강-높음)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 저장
    cv2.imwrite(save_path, heatmap_color)

# ================= 메인 실행 =================

def main():
    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # LPIPS 모델 로드 (spatial=True를 위해 VGG 권장)
    loss_fn = lpips.LPIPS(net='vgg', spatial=True).to(device)
    loss_fn.eval()

    # GT 폴더 내의 파일 리스트 가져오기 및 정렬
    files = [f for f in os.listdir(GT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort() # 파일 이름 순서대로 정렬

    print(f"Total images to process: {len(files)}")

    # CSV 데이터를 담을 리스트 초기화 (헤더 추가)
    csv_data = [['Filename', 'LPIPS_Infinite', 'LPIPS_Parallax', 'Winner']]

    for filename in files:
        # 파일 이름에서 확장자 제외한 이름(imagenum) 추출
        name_only = os.path.splitext(filename)[0]
        
        # 각 경로 설정
        gt_path = os.path.join(GT_DIR, filename)
        inf_path = os.path.join(INFINITE_DIR, filename)
        para_path = os.path.join(PARALLAX_DIR, filename)

        # 파일 존재 여부 확인
        if not os.path.exists(inf_path) or not os.path.exists(para_path):
            print(f"Skipping {filename}: Missing in Infinite or Parallax folder.")
            continue

        # 이미지 로드
        t_gt = load_image_as_tensor(gt_path)
        t_inf = load_image_as_tensor(inf_path)
        t_para = load_image_as_tensor(para_path)

        # LPIPS Map 계산 (spatial=True)
        # 결과는 (1, 1, H, W) 형태의 텐서입니다.
        with torch.no_grad():
            map_inf = loss_fn.forward(t_gt, t_inf)
            map_para = loss_fn.forward(t_gt, t_para)

        # ---------------------------------------------------------
        # [추가됨] 스칼라 점수 계산
        # Spatial Map의 평균(mean)이 곧 해당 이미지의 LPIPS 점수입니다.
        score_inf = map_inf.mean().item()
        score_para = map_para.mean().item()

        # 누가 더 낮은지(좋은지) 판단
        winner = "Parallax" if score_para < score_inf else "Infinite"
        if score_para == score_inf: winner = "Draw"

        # CSV 리스트에 추가
        csv_data.append([filename, f"{score_inf:.6f}", f"{score_para:.6f}", winner])
        # ---------------------------------------------------------

        # 결과 저장 경로 설정
        out_name_inf = f"lpipsmap_infinite_{name_only}.jpg"
        out_name_para = f"lpipsmap_parallax_{name_only}.jpg"
        
        save_path_inf = os.path.join(OUTPUT_DIR, out_name_inf)
        save_path_para = os.path.join(OUTPUT_DIR, out_name_para)

        # 히트맵 저장
        save_heatmap(map_inf, save_path_inf)
        save_heatmap(map_para, save_path_para)

        print(f"Processed: {filename} | Inf: {score_inf:.4f} | Para: {score_para:.4f} -> Best: {winner}")

    # CSV 파일 쓰기
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
        print(f"\n[Success] CSV file saved at: {csv_path}")
    except Exception as e:
        print(f"\n[Error] Failed to save CSV file: {e}")

    print("모든 처리가 완료되었습니다.")

if __name__ == "__main__":
    main()