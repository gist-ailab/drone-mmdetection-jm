import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    """
    COCO 형식의 Ground Truth와 Prediction JSON 파일을 비교하여 mAP를 계산합니다.
    """
    parser = argparse.ArgumentParser(description='COCO mAP Evaluation Script')
    
    # 첫 번째 인자: Ground Truth (GT) Annotation 파일 경로
    parser.add_argument(
        'gt_file', 
        help='Path to the ground truth COCO annotation JSON file (cropped version).'
    )
    # 두 번째 인자: 예측 결과(Prediction) 파일 경로
    parser.add_argument(
        'prediction_file', 
        help='Path to the COCO prediction results JSON file.'
    )
    args = parser.parse_args()

    print("평가를 시작합니다...")
    print(f"  - Ground Truth 파일: {args.gt_file}")
    print(f"  - Prediction 파일: {args.prediction_file}")

    # 1. Ground Truth Annotation 파일을 로드합니다.
    coco_gt = COCO(args.gt_file)

    # 2. Prediction 결과 파일을 로드합니다.
    #    loadRes()는 GT 객체를 사용하여 Prediction을 로드하며,
    #    두 파일 간의 이미지 ID가 일치하는지 확인하는 역할도 합니다.
    coco_dt = coco_gt.loadRes(args.prediction_file)

    # 3. COCOeval 객체를 생성합니다. 평가 유형은 'bbox' (바운딩 박스) 입니다.
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    # 4. 평가를 실행합니다.
    print("\nRunning evaluation...")
    coco_eval.evaluate()

    # 5. 결과를 종합합니다.
    coco_eval.accumulate()

    # 6. 최종 결과를 요약하여 출력합니다.
    print("\n---*--- COCO mAP 평가 결과 ---*---")
    coco_eval.summarize()
    print("---*--------------------------*---")


if __name__ == '__main__':
    main()