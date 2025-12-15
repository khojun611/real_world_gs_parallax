import numpy as np
import torch
from plyfile import PlyData, PlyElement

def save_box_as_ply(path, box_min, box_max, rotation=None, center=None):
    """
    box_min, box_max: (3,) tensor or numpy array (Local Extent)
    rotation: (3, 3) tensor or numpy array (Rotation Matrix) [Optional]
    center: (3,) tensor or numpy array (World Center) [Optional]
    """
    # 텐서를 넘파이로 변환
    if torch.is_tensor(box_min): box_min = box_min.detach().cpu().numpy()
    if torch.is_tensor(box_max): box_max = box_max.detach().cpu().numpy()
    if rotation is not None and torch.is_tensor(rotation): 
        rotation = rotation.detach().cpu().numpy()
    if center is not None and torch.is_tensor(center):
        center = center.detach().cpu().numpy()

    # 1. 로컬 좌표계에서 8개의 꼭짓점 생성
    # (min_x, min_y, min_z) ... (max_x, max_y, max_z) 조합
    corners = np.array([
        [box_min[0], box_min[1], box_min[2]],
        [box_min[0], box_min[1], box_max[2]],
        [box_min[0], box_max[1], box_min[2]],
        [box_min[0], box_max[1], box_max[2]],
        [box_max[0], box_min[1], box_min[2]],
        [box_max[0], box_min[1], box_max[2]],
        [box_max[0], box_max[1], box_min[2]],
        [box_max[0], box_max[1], box_max[2]]
    ])

    # 2. 회전 적용 (Local -> World Rotation)
    if rotation is not None:
        # corners shape: (8, 3), rotation shape: (3, 3)
        # Transpose 주의: 보통 점 @ R.T 형태로 곱하거나 R @ 점.T
        corners = np.dot(corners, rotation.T)

    # 3. 중심점 이동 (Translation)
    if center is not None:
        corners += center

    # 4. PLY 파일로 저장 (Vertex & Face 정의)
    vertices = np.array([tuple(v) for v in corners], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    # 육면체의 6개 면을 구성하는 12개의 삼각형 인덱스 (0~7번 버텍스 연결)
    faces = np.array([
        ([0, 1, 3],), ([0, 3, 2],), # Left
        ([4, 6, 7],), ([4, 7, 5],), # Right
        ([0, 2, 6],), ([0, 6, 4],), # Bottom
        ([1, 5, 7],), ([1, 7, 3],), # Top
        ([0, 4, 5],), ([0, 5, 1],), # Front
        ([2, 3, 7],), ([2, 7, 6],)  # Back
    ], dtype=[('vertex_indices', 'i4', (3,))])

    el_verts = PlyElement.describe(vertices, 'vertex')
    el_faces = PlyElement.describe(faces, 'face')
    
    PlyData([el_verts, el_faces]).write(path)