import json
from pycocotools.coco import COCO

# 검증할 annotation 파일 경로
ann_file = '/ailab_mat2/dataset/drone/250312_sejong/drone_250312_sejong_multimodal_coco/labels/train.json'

# COCO API 로드
coco = COCO(ann_file)

print(f"총 Annotation 개수: {len(coco.anns)}")

problematic_anns = []
for ann_id, ann in coco.anns.items():
    # bbox format: [x, y, width, height]
    bbox = ann['bbox']
    width = bbox[2]
    height = bbox[3]
    
    # 너비 또는 높이가 1 픽셀 이하인 경우 (0 포함)
    if width <= 3 or height <= 3:
        problematic_anns.append(ann)
        
if not problematic_anns:
    print("✅ 데이터셋에서 유효하지 않은 Bounding Box를 찾지 못했습니다.")
else:
    print(f"🚨 총 {len(problematic_anns)}개의 유효하지 않은 Bounding Box를 찾았습니다.")
    for ann in problematic_anns:
        image_id = ann['image_id']
        image_info = coco.loadImgs(image_id)[0]
        print(f"  - Image ID: {image_id} (파일: {image_info['file_name']}), Anno ID: {ann['id']}, BBox: {ann['bbox']}")

# 수정이 필요하다면 아래와 같이 새로운 annotation 파일을 생성할 수 있습니다.
# (이 코드는 예시이며, 실제 수정 시에는 백업 후 신중하게 진행해야 합니다.)
#
# with open(ann_file, 'r') as f:
#     data = json.load(f)
#
# valid_annotations = [ann for ann in data['annotations'] if ann['bbox'][2] > 1 and ann['bbox'][3] > 1]
# data['annotations'] = valid_annotations
#
# new_ann_file = ann_file.replace('.json', '_filtered.json')
# with open(new_ann_file, 'w') as f:
#     json.dump(data, f)
#
# print(f"\n수정된 Annotation 파일이 '{new_ann_file}'에 저장되었습니다.")
# print(f"수정 후 Annotation 개수: {len(valid_annotations)}")