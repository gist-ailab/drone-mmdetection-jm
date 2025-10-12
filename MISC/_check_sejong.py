import os
import cv2
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
import argparse

def get_category_names(coco_instance):
    """
    COCO 객체에서 카테고리 ID와 이름의 매핑을 가져옵니다.
    
    Returns:
        dict: {category_id: category_name}
    """
    category_ids = coco_instance.getCatIds()
    categories = coco_instance.loadCats(category_ids)
    return {cat['id']: cat['name'] for cat in categories}

def draw_bbox(image, bbox, label, color, category_names):
    """
    단일 이미지에 바운딩 박스와 라벨을 그립니다.
    """
    x, y, w, h = [int(c) for c in bbox]
    
    # Bbox 그리기
    cv2.rectangle(image, (x, y), (x + w, y + h), color.tolist(), 2)
    
    # 텍스트 라벨
    label_text = category_names.get(label, 'Unknown')
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # 텍스트 배경 및 텍스트 그리기
    cv2.rectangle(image, (x, y - text_h - 5), (x + text_w, y), color.tolist(), -1)
    cv2.putText(image, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image

def visualize(coco_path: str, data_root: str):
    """
    Sejong Multimodal 데이터셋의 GT를 시각화하고 상호작용합니다.

    Args:
        coco_path (str): COCO annotation json 파일 경로.
        data_root (str): 'images' 폴더가 있는 데이터셋 루트 경로.
    """
    if not os.path.exists(coco_path):
        print(f"Error: Annotation file not found at {coco_path}")
        return
        
    coco = COCO(coco_path)
    category_names = get_category_names(coco)
    
    # 클래스 개수에 맞춰 동적으로 색상 팔레트 생성
    num_classes = len(category_names)
    palette = (np.random.rand(num_classes, 3) * 255).astype(int)
    
    print("Found Categories:", list(category_names.values()))
    
    img_ids = sorted(coco.getImgIds())
    if not img_ids:
        print("No images found in the annotation file.")
        return
        
    current_idx = 0
    
    while True:
        # 1. 현재 이미지 정보 로드
        img_info = coco.loadImgs(img_ids[current_idx])[0]
        
        # 'file_name'은 일반적으로 'group_rgb/group_XX/frame_YYY.png'와 같은 형태
        # 이 경로를 기준으로 다른 모달리티 경로를 생성
        relative_rgb_path = img_info['file_name']
        rgb_path = os.path.join(data_root, '', relative_rgb_path)
        
        # 2. 모든 모달리티 이미지 경로 구성
        modality_paths = {
            'RGB': rgb_path,
            'Depth': rgb_path.replace('group_rgb', 'group_depth'),
            'Event': rgb_path.replace('group_rgb', 'group_ir'), # IR이 Event로 사용됨
            'LiDAR': rgb_path.replace('group_rgb', 'group_intensity') # Intensity가 LiDAR로 사용됨
        }
        
        images = {}
        for modality, path in modality_paths.items():
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is None:
                    # 이미지를 로드할 수 없는 경우, 검은색 이미지로 대체
                    print(f"Warning: Could not load {modality} image at {path}")
                    images[modality] = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    # Depth나 다른 모달리티가 그레이스케일일 경우 컬러로 변환
                    if len(img.shape) == 2 or img.shape[2] == 1:
                        img = cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET)
                    images[modality] = img
            else:
                # 파일이 존재하지 않는 경우, 검은색 이미지로 대체
                print(f"Warning: {modality} file not found at {path}")
                images[modality] = np.zeros((480, 640, 3), dtype=np.uint8)

        # 3. Annotation 정보 가져오기
        ann_ids = coco.getAnnIds(imgIds=img_info['id'])
        anns = coco.loadAnns(ann_ids)

        # 4. 각 모달리티 이미지에 Bbox 그리기
        for ann in anns:
            bbox = ann['bbox']
            category_id = ann['category_id']
            color = palette[list(category_names.keys()).index(category_id)]
            
            for modality in images:
                images[modality] = draw_bbox(images[modality], bbox, category_id, color, category_names)

        # 5. 2x2 그리드로 이미지 합치기
        top_row = np.hstack((images.get('RGB'), images.get('Depth')))
        bottom_row = np.hstack((images.get('Event'), images.get('LiDAR')))
        combined_img = np.vstack((top_row, bottom_row))
        
        # 이미지 정보 텍스트 추가
        info_text = f"Index: {current_idx}/{len(img_ids)-1} | File: {relative_rgb_path}"
        cv2.putText(combined_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Sejong Dataset GT Check (q: quit, a/d: prev/next)', combined_img)

        # 6. 키보드 입력 처리
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): # 종료
            break
        elif key == ord('d') or key == 83: # 'd' 또는 오른쪽 화살표
            current_idx = (current_idx + 1) % len(img_ids)
        elif key == ord('a') or key == 81: # 'a' 또는 왼쪽 화살표
            current_idx = (current_idx - 1 + len(img_ids)) % len(img_ids)
            
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visualize Sejong Multimodal Dataset Ground Truth")
    parser.add_argument('--coco_path', 
                        default='/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_V2/labels/test_filtered_10p.json',
                        help='Path to the COCO format annotation file (.json)')
    parser.add_argument('--data_root', 
                        default='/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_V2/images_heuristic',
                        help='Root directory of the dataset containing the "images" folder')
    args = parser.parse_args()
    
    visualize(args.coco_path, args.data_root)

if __name__ == "__main__":
    main()