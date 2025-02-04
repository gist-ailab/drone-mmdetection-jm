import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
# from ..functional import eval_recalls
from mmdet.evaluation.functional import eval_recalls
import numpy as np
import json
from mmdet.evaluation import CocoMetric
from collections import defaultdict



@METRICS.register_module()
class RGBTEvaluator:
    def __init__(self, ann_file):
        """
        Initialize the evaluation object.
        :param annotation_path: Path to the ground truth annotations in COCO format.
        """
        self.coco_gt = json.load(open(ann_file, 'r'))
        # self.image_id_map = {img['id']: img['im_name'] for img in self.coco_gt['images']}
        # self.image_id_map = {img['id']: img['im_name'] for img in self.coco_gt['images']}
        self.image_ids = [img['id'] for img in self.coco_gt['images']]
        self.annotations = defaultdict(list)
        
        # Store ground truth annotations per image
        for ann in self.coco_gt['annotations']:
            self.annotations[ann['image_id']].append(ann)

    def evaluate(self, results):
        """
        Evaluate the results using both COCO mAP and Miss Rate.
        :param results: List of detection results in format:
                        [
                            {"image_id": int, "bbox": [x, y, w, h], "score": float},
                            ...
                        ]
        :return: Evaluation metrics dictionary
        """
        # Compute COCO mAP
        coco_metric = CocoMetric(ann_file=self.coco_gt)
        coco_results = coco_metric.evaluate(results)

        # Compute Miss Rate
        miss_rate_metrics = self.compute_miss_rate(results)

        # Combine results
        evaluation_metrics = {
            "COCO_mAP": coco_results,
            "Miss_Rate": miss_rate_metrics
        }
        return evaluation_metrics

    def compute_miss_rate(self, results, iou_threshold=0.5):
        """
        Compute the Miss Rate (MR) at different False Positives Per Image (FPPI).
        :param results: List of detection results in COCO format.
        :param iou_threshold: IoU threshold for matching.
        :return: Dictionary containing FPPI and Miss Rate values.
        """
        # img_ids = list(self.image_id_map.keys())

        total_images = len(self.img_ids)
        tp_list = []
        fp_list = []
        num_gt = 0

        for img_id in self.img_ids:
            gt_boxes = [ann["bbox"] for ann in self.annotations[img_id]]
            det_boxes = [det["bbox"] for det in results if det["image_id"] == img_id]
            det_scores = [det["score"] for det in results if det["image_id"] == img_id]

            num_gt += len(gt_boxes)

            if not det_boxes:
                fp_list.append(len(gt_boxes))  # All ground truths are missed
                tp_list.append(0)
                continue

            # Sort detections by confidence score
            sorted_indices = np.argsort(det_scores)[::-1]
            det_boxes = [det_boxes[i] for i in sorted_indices]

            # Compute IoUs
            matches, false_positives = self.match_detections(gt_boxes, det_boxes, iou_threshold)
            tp_list.append(matches)
            fp_list.append(false_positives)

        # Compute cumulative TP and FP
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        fppi = fp_cumsum / total_images
        miss_rate = 1 - (tp_cumsum / num_gt)

        return {"FPPI": fppi.tolist(), "Miss_Rate": miss_rate.tolist()}

    def match_detections(self, gt_boxes, det_boxes, iou_threshold):
        """
        Match detections to ground truth using IoU.
        :param gt_boxes: List of ground truth bounding boxes.
        :param det_boxes: List of detected bounding boxes.
        :param iou_threshold: IoU threshold for a valid match.
        :return: Number of true positives and false positives.
        """
        if len(gt_boxes) == 0:
            return 0, len(det_boxes)  # No ground truth, all detections are false positives

        ious = self.compute_iou_matrix(gt_boxes, det_boxes)
        matches = 0
        used_gt = set()

        for det_idx, iou_values in enumerate(ious.T):
            best_match = np.argmax(iou_values)
            if iou_values[best_match] >= iou_threshold and best_match not in used_gt:
                matches += 1
                used_gt.add(best_match)

        false_positives = len(det_boxes) - matches
        return matches, false_positives

    def compute_iou_matrix(self, gt_boxes, det_boxes):
        """
        Compute IoU matrix between ground truth and detected boxes.
        :param gt_boxes: List of ground truth bounding boxes.
        :param det_boxes: List of detected bounding boxes.
        :return: IoU matrix.
        """
        gt_boxes = np.array(gt_boxes)
        det_boxes = np.array(det_boxes)

        gt_x1, gt_y1, gt_w, gt_h = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
        gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h

        det_x1, det_y1, det_w, det_h = det_boxes[:, 0], det_boxes[:, 1], det_boxes[:, 2], det_boxes[:, 3]
        det_x2, det_y2 = det_x1 + det_w, det_y1 + det_h

        # Compute intersection
        inter_x1 = np.maximum(det_x1[:, None], gt_x1)
        inter_y1 = np.maximum(det_y1[:, None], gt_y1)
        inter_x2 = np.minimum(det_x2[:, None], gt_x2)
        inter_y2 = np.minimum(det_y2[:, None], gt_y2)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Compute union
        det_area = det_w * det_h
        gt_area = gt_w * gt_h
        union_area = det_area[:, None] + gt_area - inter_area

        # Compute IoU
        iou_matrix = inter_area / np.maximum(union_area, 1e-6)
        return iou_matrix
