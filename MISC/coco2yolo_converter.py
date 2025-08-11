import json
import os
import shutil
from tqdm import tqdm
import yaml

def convert_coco_to_yolo(coco_json_path, image_root_path, output_yolo_root, split_name, class_names):
    """
    COCO JSON 파일을 YOLO 형식으로 변환하고 이미지를 복사합니다.

    Args:
        coco_json_path (str): COCO 형식의 JSON 파일 경로 (예: train.json 또는 test.json).
        image_root_path (str): 원본 이미지 파일들이 있는 기본 경로 (예: /ailab_mat2/dataset/drone/drone_250610/images).
        output_yolo_root (str): YOLO 데이터셋이 저장될 기본 경로.
        split_name (str): 'train' 또는 'test'와 같이 데이터 분할 이름.
        class_names (list): COCO JSON에서 추출한 클래스 이름 리스트 (인덱스 순서대로).
    """
    print(f"\n--- {split_name} 데이터셋 변환 시작 ---")

    # 출력 디렉토리 경로 설정
    output_images_dir = os.path.join(output_yolo_root, 'images', split_name)
    output_labels_dir = os.path.join(output_yolo_root, 'labels', split_name)

    # 디렉토리 생성
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"오류: COCO JSON 파일을 찾을 수 없습니다: {coco_json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"오류: JSON 파일을 파싱하는 중 오류 발생: {e}")
        return

    # 이미지 ID를 이미지 정보 (파일명, 너비, 높이)에 매핑
    images = {img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']}
              for img in coco_data['images']}

    # 각 이미지의 어노테이션을 저장할 딕셔너리 초기화
    image_annotations = {img_id: [] for img_id in images.keys()}

    # 어노테이션 정보 파싱 및 이미지별로 그룹화
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox'] # [x_min, y_min, width, height]

        # COCO category_id를 YOLO class_id로 변환
        # COCO category_id는 1부터 시작할 수 있고, 중간에 비어있는 ID가 있을 수 있으므로
        # 실제 class_names 리스트의 인덱스에 매핑하는 작업이 필요합니다.
        # 여기서는 COCO JSON의 'categories' 섹션에서 얻은 순서대로 0부터 시작하는 인덱스를 사용한다고 가정합니다.
        # 즉, coco_data['categories']의 id와 class_names 리스트의 인덱스가 매칭되어야 합니다.
        # 만약 그렇지 않다면, 별도의 매핑 로직이 필요합니다.
        # 일단은 coco_data['categories']에서 추출한 class_id를 그대로 사용합니다.
        
        # COCO category_id와 class_names 인덱스 매핑 (가장 일반적인 경우)
        # COCO category_id는 1부터 시작하는 경우가 많으므로, 실제 클래스 이름 리스트 인덱스로 변환해야 합니다.
        # 예시: coco_data['categories']가 [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'car'}] 이고
        # class_names가 ['person', 'car']라면 category_id 1은 index 0으로, 2는 index 1로 매핑해야 합니다.
        
        # 좀 더 안전한 매핑을 위해 COCO category ID -> YOLO class index 매핑 딕셔너리 생성
        if not hasattr(convert_coco_to_yolo, 'category_id_to_yolo_idx'):
            convert_coco_to_yolo.category_id_to_yolo_idx = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

        yolo_class_idx = convert_coco_to_yolo.category_id_to_yolo_idx.get(category_id)
        if yolo_class_idx is None:
            print(f"경고: COCO category_id {category_id}에 해당하는 YOLO 클래스 인덱스를 찾을 수 없습니다. 어노테이션을 건너뜜.")
            continue


        image_annotations[image_id].append({'class_id': yolo_class_idx, 'bbox': bbox})

    # 이미지 처리 및 라벨 파일 생성
    print(f"이미지 복사 및 라벨 파일 생성 중...")
    for img_id, anns in tqdm(image_annotations.items(), desc=f"처리 중 {split_name}"):
        img_info = images[img_id]
        img_file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']

        original_img_path = os.path.join(image_root_path, img_file_name)
        output_img_path = os.path.join(output_images_dir, img_file_name)

        # 이미지 복사
        if os.path.exists(original_img_path):
            shutil.copy(original_img_path, output_img_path)
        else:
            print(f"경고: 원본 이미지를 찾을 수 없습니다. 건너뜜: {original_img_path}")
            continue

        # 라벨 파일 생성 (YOLO 형식)
        label_file_name = os.path.splitext(img_file_name)[0] + '.txt'
        label_file_path = os.path.join(output_labels_dir, label_file_name)

        with open(label_file_path, 'w') as f:
            for ann in anns:
                class_id = ann['class_id']
                x_min, y_min, bbox_width, bbox_height = ann['bbox']

                # COCO (x_min, y_min, width, height) -> YOLO (x_center_norm, y_center_norm, width_norm, height_norm)
                x_center = (x_min + bbox_width / 2) / img_width
                y_center = (y_min + bbox_height / 2) / img_height
                width_norm = bbox_width / img_width
                height_norm = bbox_height / img_height

                # 좌표 유효성 검사 (0.0 ~ 1.0 범위 확인)
                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and
                        0.0 <= width_norm <= 1.0 and 0.0 <= height_norm <= 1.0):
                    print(f"경고: {img_file_name}의 바운딩 박스 좌표가 범위를 벗어났습니다. ({x_center:.2f}, {y_center:.2f}, {width_norm:.2f}, {height_norm:.2f}). 라벨링 오류일 수 있습니다.")
                    # 이 어노테이션은 건너뛰거나 다른 방식으로 처리할 수 있습니다.
                    continue

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    
    print(f"--- {split_name} 데이터셋 변환 완료 ---")


def create_data_yaml(output_yolo_root, class_names, num_classes):
    """
    YOLOv5 학습을 위한 data.yaml 파일을 생성합니다.
    """
    yaml_content = {
        'path': output_yolo_root,
        'train': 'images/train',
        'val': 'images/test', # 일반적으로 test.json은 val로 사용되거나 별도 test 폴더로 사용
        'nc': num_classes,
        'names': class_names
    }

    yaml_file_path = os.path.join(output_yolo_root, 'data.yaml')
    with open(yaml_file_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"\ndata.yaml 파일이 생성되었습니다: {yaml_file_path}")
    print(f"내용:\n{yaml.dump(yaml_content, default_flow_style=False)}")


# --- 설정 변수 ---
ORIGINAL_DATASET_ROOT = '/ailab_mat2/dataset/drone/drone_250610'
OUTPUT_YOLO_ROOT = '/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_dataset_0723'

ORIGINAL_IMAGES_PATH = os.path.join(ORIGINAL_DATASET_ROOT, 'images')
ORIGINAL_LABELS_PATH = os.path.join(ORIGINAL_DATASET_ROOT, 'labels')

TRAIN_JSON = os.path.join(ORIGINAL_LABELS_PATH, 'train.json')
TEST_JSON = os.path.join(ORIGINAL_LABELS_PATH, 'test.json')

# COCO JSON에서 클래스 정보 로드
# train.json 또는 test.json 중 하나에서 클래스 정보를 추출합니다.
# 두 파일의 클래스 정보가 동일하다고 가정합니다.
try:
    with open(TRAIN_JSON, 'r') as f:
        train_coco_data = json.load(f)
    categories = train_coco_data['categories']
    # 클래스 이름 리스트 (COCO category_id 순서에 관계없이 0부터 시작하는 YOLO 인덱스로 매핑)
    # COCO JSON의 'categories' 섹션에 있는 순서대로 YOLO의 0, 1, 2... 인덱스에 매핑합니다.
    # 즉, `names` 리스트의 n번째 요소는 YOLO 클래스 ID n에 해당합니다.
    CLASS_NAMES = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    NUM_CLASSES = len(CLASS_NAMES)
except FileNotFoundError:
    print(f"오류: '{TRAIN_JSON}'을 찾을 수 없습니다. 클래스 정보를 로드할 수 없습니다.")
    exit()
except json.JSONDecodeError as e:
    print(f"오류: '{TRAIN_JSON}' 파싱 중 오류 발생: {e}. 클래스 정보를 로드할 수 없습니다.")
    exit()
except Exception as e:
    print(f"클래스 정보 로드 중 예상치 못한 오류 발생: {e}")
    exit()


# --- 변환 실행 ---
if __name__ == "__main__":
    print(f"--- COCO to YOLO 변환 스크립트 시작 ---")
    print(f"원본 데이터셋 경로: {ORIGINAL_DATASET_ROOT}")
    print(f"새로운 YOLO 데이터셋 저장 경로: {OUTPUT_YOLO_ROOT}")
    print(f"클래스 개수: {NUM_CLASSES}, 클래스 이름: {CLASS_NAMES}")

    # 기존 출력 디렉토리 삭제 후 재생성 (선택 사항: 기존에 있다면 삭제하고 새로 시작)
    if os.path.exists(OUTPUT_YOLO_ROOT):
        print(f"기존 출력 디렉토리 '{OUTPUT_YOLO_ROOT}'를 삭제합니다...")
        shutil.rmtree(OUTPUT_YOLO_ROOT)
        print("삭제 완료.")
    os.makedirs(OUTPUT_YOLO_ROOT, exist_ok=True)
    
    # 학습 데이터 변환
    convert_coco_to_yolo(TRAIN_JSON, ORIGINAL_IMAGES_PATH, OUTPUT_YOLO_ROOT, 'train', CLASS_NAMES)

    # 테스트 데이터 변환 (test.json을 val로 사용)
    convert_coco_to_yolo(TEST_JSON, ORIGINAL_IMAGES_PATH, OUTPUT_YOLO_ROOT, 'test', CLASS_NAMES) # 'test' 폴더로 생성

    # data.yaml 파일 생성
    create_data_yaml(OUTPUT_YOLO_ROOT, CLASS_NAMES, NUM_CLASSES)

    print("\n--- COCO to YOLO 변환 스크립트 완료 ---")
    print(f"YOLO 데이터셋이 다음 경로에 생성되었습니다: {OUTPUT_YOLO_ROOT}")
    print("이제 이 데이터셋으로 YOLOv5 모델을 학습할 수 있습니다.")
    print("생성된 data.yaml 파일을 확인하여 경로 및 클래스 정보가 올바른지 확인해주세요.")