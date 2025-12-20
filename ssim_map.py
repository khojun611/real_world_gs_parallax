import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import lpips
import os
import glob
import pandas as pd  # 결과 저장을 위해 pandas 사용 (없으면 pip install pandas)

# ============================================================
# 1) PSNR / SSIM utility (기존 코드 유지)
# ============================================================

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D @ _1D.t()
    window = _2D.unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean(), ssim_map
    else:
        return ssim_map.mean(1).mean(1).mean(1), ssim_map

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    window = window.to(img1.device).type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))


# ============================================================
# 2) Core Logic: Process Single Pair
# ============================================================

def process_single_pair(render_path, gt_path, lpips_model, save_prefix):
    """
    이미지 한 쌍을 비교하고 Metrics를 반환하며, Map을 저장합니다.
    save_prefix: 예) "./results/image001_method1"
    """
    # Load images
    render = cv2.imread(render_path)
    gt     = cv2.imread(gt_path)

    if render is None:
        print(f"[Warning] Cannot load render: {render_path}")
        return None
    if gt is None:
        print(f"[Warning] Cannot load GT: {gt_path}")
        return None

    render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
    gt     = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    H, W = render.shape[:2]
    # GT 크기가 다르면 리사이즈 (보통 GT가 더 클 수 있음)
    if gt.shape != render.shape:
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_AREA)

    # Tensor conversion
    render_t = torch.from_numpy(render).permute(2,0,1).unsqueeze(0).cuda().float() / 255.
    gt_t     = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).cuda().float() / 255.

    # --- Metrics ---
    psnr_val = psnr(render_t, gt_t).item()
    ssim_val, ssim_map_tensor = ssim(render_t, gt_t)
    ssim_val = ssim_val.item()
    lpips_val = lpips_model(render_t, gt_t).item()

    # --- Error Map Saving ---
    error_map = torch.abs(render_t - gt_t).mean(dim=1)[0].detach().cpu().numpy()
    err_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    
    cmap = plt.get_cmap("turbo")
    err_color = (cmap(err_norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(err_color).save(f"{save_prefix}_error.png")

    # --- SSIM Map Saving (1 - SSIM) ---
    ssim_map_gray = ssim_map_tensor.mean(dim=1, keepdim=True)
    ssim_map_np = (1 - ssim_map_gray[0,0]).detach().cpu().numpy()
    ssim_norm = (ssim_map_np - ssim_map_np.min()) / (ssim_map_np.max() - ssim_map_np.min() + 1e-8)
    
    ssim_color = (cmap(ssim_norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(ssim_color).save(f"{save_prefix}_ssim.png")

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val
    }

# ============================================================
# 3) Batch Processing Logic
# ============================================================

def get_file_map(directory):
    """
    디렉토리 내의 모든 이미지 파일을 스캔하여 
    { '파일명_확장자제외': '파일전체경로' } 형태의 딕셔너리를 반환합니다.
    예: '0001.png' -> {'0001': '/path/to/0001.png'}
    """
    file_map = {}
    # 모든 파일 검색
    files = glob.glob(os.path.join(directory, "*"))
    
    for f_path in files:
        # 이미지 확장자 체크 (대소문자 무시)
        if f_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            filename = os.path.basename(f_path)
            basename, _ = os.path.splitext(filename)
            file_map[basename] = f_path
            
    return file_map

def compare_folders(gt_dir, render1_dir, render2_dir, output_dir, method1_name="method1", method2_name="method2"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("[INFO] Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').cuda()

    # 1. 각 폴더의 파일을 미리 스캔하여 Map 생성 (확장자 무시 매칭을 위해)
    print(f"[INFO] Scanning directories...")
    gt_map = get_file_map(gt_dir)
    r1_map = get_file_map(render1_dir)
    r2_map = get_file_map(render2_dir)
    
    print(f" - Found {len(gt_map)} GT images")
    print(f" - Found {len(r1_map)} Render1 images")
    print(f" - Found {len(r2_map)} Render2 images")

    results_list = []
    
    # GT 파일 이름을 기준으로 순회
    # 정렬하여 순서대로 처리
    sorted_basenames = sorted(gt_map.keys())

    for base_name in sorted_basenames:
        gt_path = gt_map[base_name]
        
        # Render 폴더에도 해당 파일이름이 있는지 확인 (확장자는 달라도 됨)
        if base_name in r1_map and base_name in r2_map:
            r1_path = r1_map[base_name]
            r2_path = r2_map[base_name]
            
            print(f"Processing: {base_name} ...")
            # (디버깅용: 실제 매칭된 파일 경로 확인)
            # print(f"   GT: {os.path.basename(gt_path)} / R1: {os.path.basename(r1_path)} / R2: {os.path.basename(r2_path)}")

            # 저장 경로 접두어
            save_prefix_m1 = os.path.join(output_dir, f"{base_name}_{method1_name}")
            save_prefix_m2 = os.path.join(output_dir, f"{base_name}_{method2_name}")

            # Method 1 처리
            m1_res = process_single_pair(r1_path, gt_path, lpips_fn, save_prefix_m1)
            
            # Method 2 처리
            m2_res = process_single_pair(r2_path, gt_path, lpips_fn, save_prefix_m2)

            if m1_res and m2_res:
                row = {
                    "filename": base_name,
                    f"{method1_name}_psnr": m1_res['psnr'],
                    f"{method1_name}_ssim": m1_res['ssim'],
                    f"{method1_name}_lpips": m1_res['lpips'],
                    f"{method2_name}_psnr": m2_res['psnr'],
                    f"{method2_name}_ssim": m2_res['ssim'],
                    f"{method2_name}_lpips": m2_res['lpips'],
                }
                results_list.append(row)
        else:
            # 어느 한쪽이라도 없으면 Skip
            missing = []
            if base_name not in r1_map: missing.append(f"{method1_name}")
            if base_name not in r2_map: missing.append(f"{method2_name}")
            print(f"[Skip] Missing render file for '{base_name}' in {missing}")

    # 결과 저장
    if results_list:
        df = pd.DataFrame(results_list)
        
        mean_row = df.mean(numeric_only=True).to_dict()
        mean_row["filename"] = "AVERAGE"
        df_final = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        
        csv_path = os.path.join(output_dir, "metrics_summary.csv")
        df_final.to_csv(csv_path, index=False)
        print(f"\n[Done] Summary saved to {csv_path}")
        print(f"Method1 Avg PSNR: {mean_row[f'{method1_name}_psnr']:.4f}")
        print(f"Method2 Avg PSNR: {mean_row[f'{method2_name}_psnr']:.4f}")
    else:
        print("[Error] No matched images found.")
# ============================================================
# 4) 실행 설정
# ============================================================

if __name__ == "__main__":
    
    # 예시 경로 설정
    gt_folder_path      = "/home/iris/mount1/marigold/pcc_monkey/images"
    
    render1_folder_path = "/home/iris/mount1/ref14/real_world_gs/output/pcc_monkey_wopcc512/test/renders_infinite/rgb"
    render2_folder_path = "/home/iris/mount1/ref14/real_world_gs/output/pcc_monkey3/test/renders_parallax/rgb"
    
    output_save_path    = "./ssim_monkey"

    # method_name은 파일명 뒤에 붙을 식별자입니다 (예: 219_baseline_error.png)
    compare_folders(
        gt_dir=gt_folder_path,
        render1_dir=render1_folder_path,
        render2_dir=render2_folder_path,
        output_dir=output_save_path,
        method1_name="baseline",
        method2_name="parallax"
    )