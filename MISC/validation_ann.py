import json

# 검사할 annotation 파일 경로
ann_file_path = '/home/jovyan/SSDc/jemo_maeng/dset/drone_250312_sejong_multimodal_coco_cropped/labels/train_cropped3.json'

print(f"'{ann_file_path}' 파일을 검사합니다...")

with open(ann_file_path, 'r') as f:
    coco_data = json.load(f)

# 모든 카테고리 ID를 set으로 만들어 유효성 검사에 사용
valid_category_ids = {cat['id'] for cat in coco_data['categories']}
print(f"유효한 Category IDs: {valid_category_ids}")

problematic_anns = []
# 'annotations' 키가 있는지 확인
if 'annotations' not in coco_data:
    raise KeyError("'annotations' 키를 찾을 수 없습니다. 파일 형식을 확인해주세요.")

for ann in coco_data['annotations']:
    # category_id 키가 없거나, 값이 유효하지 않은 경우
    if 'category_id' not in ann or ann['category_id'] not in valid_category_ids:
        problematic_anns.append(ann)

if problematic_anns:
    print(f"\n[오류] 총 {len(problematic_anns)}개의 문제가 있는 annotation을 찾았습니다.")
    # 문제가 있는 annotation 중 최대 5개만 출력
    for i, ann in enumerate(problematic_anns[:5]):
        print(f"  - Annotation ID: {ann.get('id', 'N/A')}, Image ID: {ann.get('image_id', 'N/A')}, Category ID: {ann.get('category_id', 'Missing or Invalid')}")
    if len(problematic_anns) > 5:
        print("  ...")
    print("\nJSON 파일을 열어 해당 annotation들을 수정하거나 제거한 후 다시 시도하십시오.")
else:
    print("\n[성공] 모든 annotation에 유효한 'category_id'가 존재합니다.")