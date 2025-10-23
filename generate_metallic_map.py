import torch
import argparse
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from skimage.measure import label
import os

# --- 1. DINO 모델 로딩 및 이미지 전처리 함수 ---

def load_dinov2_model(model_name='dinov2_vitl14'):
    """사전 학습된 DINOv2 모델을 로드합니다."""
    print(f"Loading DINOv2 model: {model_name}...")
    try:
        dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name).cuda().eval()
        print("DINOv2 model loaded successfully.")
        return dinov2_model
    except Exception as e:
        print(f"Error loading DINOv2 model: {e}")
        print("Please ensure you have an internet connection and pytorch hub is working.")
        exit()

def prepare_image_for_dino(pil_image):
    """PIL 이미지를 DINO 입력 형식에 맞는 텐서로 변환합니다."""
    img_tensor = TF.to_tensor(pil_image).unsqueeze(0).cuda()
    return img_tensor

@torch.no_grad()
def create_metallic_map(args):
    """이미지 파일들을 입력받아 Metallic 맵을 생성합니다."""
    
    # --- 2. 입력 이미지 로드 ---
    print(f"Loading RGB image from: {args.rgb_image_path}")
    rgb_image_pil = Image.open(args.rgb_image_path).convert("RGB")
    
    print(f"Loading uncertainty map from: {args.uncertainty_map_path}")
    # .npy 파일 로드
    uncertainty_array = np.load(args.uncertainty_map_path)

    # NumPy 배열을 PyTorch 텐서로 변환
    uncertainty_tensor = torch.from_numpy(uncertainty_array).float().cuda()

    # 텐서 모양이 [H, W] 형태일 수 있으므로, [1, H, W] 형태로 채널 차원 추가
    if uncertainty_tensor.ndim == 2:
        uncertainty_tensor = uncertainty_tensor.unsqueeze(0)
    H, W = rgb_tensor.shape[1], rgb_tensor.shape[2]
    
    # DINO 모델 로드
    dino_model = load_dinov2_model()
    
    # --- 3. DINO 특징 추출 ---
    print("Extracting DINO features...")
    dino_input_tensor = prepare_image_for_dino(rgb_image_pil)
    
    patch_size = dino_model.patch_embed.patch_size[0] # 모델의 패치 사이즈 (보통 14)
    features_dict = dino_model.forward_features(dino_input_tensor)
    feature_map = features_dict['x_norm_patchtokens'].reshape(
        1, H // patch_size, W // patch_size, -1
    ).permute(0, 3, 1, 2)
    
    feature_map_upsampled = torch.nn.functional.interpolate(
        feature_map, size=(H, W), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    # --- 4. 씨앗 마스크 생성 및 정제 ---
    print("Creating and refining seed mask...")
    threshold = torch.quantile(uncertainty_tensor, args.seed_quantile)
    seed_mask = (uncertainty_tensor > threshold).squeeze(0)

    if args.use_largest_component:
        labels = label(seed_mask.cpu().numpy())
        if labels.max() > 0:
            largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
            seed_mask = torch.from_numpy(labels == largest_label).cuda()
            print("Filtered seed mask to the largest component.")

    # --- 5. 대표 특징 계산 및 확장 ---
    seed_features = feature_map_upsampled[:, seed_mask]
    if seed_features.shape[1] == 0:
        print("Error: No seed pixels found. Try lowering --seed_quantile value.")
        return
        
    target_feature = torch.mean(seed_features, dim=1, keepdim=True)
    
    print("Calculating similarity across the image...")
    similarity_map = torch.nn.functional.cosine_similarity(
        feature_map_upsampled, target_feature.unsqueeze(-1), dim=0
    )
    
    # --- 6. 최종 Metallic 맵 생성 및 저장 ---
    metallic_map = (similarity_map > args.similarity_thresh).float()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    TF.to_pil_image(metallic_map.cpu()).save(args.output_path)
    print(f"✅ Success! Metallic map saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a metallic map from an RGB image and an Uncertainty map using DINO.")
    
    parser.add_argument("--rgb_image_path", type=str, required=True, help="Path to the input RGB image file (e.g., render.png).")
    parser.add_argument("--uncertainty_map_path", type=str, required=True, help="Path to the input uncertainty map image file (e.g., uncertainty.png).")
    parser.add_argument("--output_path", type=str, default="output/metallic_map.png", help="Path to save the final metallic map image.")
    
    parser.add_argument("--seed_quantile", type=float, default=0.95, help="Quantile (0.0 to 1.0) to select seeds from the uncertainty map. Default is 0.95 (top 5%).")
    parser.add_argument("--similarity_thresh", type=float, default=0.8, help="Cosine similarity threshold (0.0 to 1.0) to generate the final map.")
    parser.add_argument("--use_largest_component", action='store_true', help="Use only the largest connected region of seeds to reduce noise.")
    
    args = parser.parse_args()
    create_metallic_map(args)