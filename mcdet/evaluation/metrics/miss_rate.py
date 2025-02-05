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
class RGBTEvaluator(BaseMetric):
    def __init__(self, ann_file, iou_threshold=0.5):
        """
        RGBT Evaluator for COCO mAP and Miss Rate (MR) computation.
        :param ann_file: Path to the ground truth annotations in COCO format.
        :param iou_threshold: IoU threshold for a valid match.
        """
        super().__init__()
        self.ann_file = ann_file
        self.iou_threshold = iou_threshold

        # Load ground truth annotations
        with open(self.ann_file, 'r') as f:
            self.coco_gt = json.load(f)

        self.image_ids = [img['id'] for img in self.coco_gt['images']]
        self.annotations = defaultdict(list)

        for ann in self.coco_gt['annotations']:
            self.annotations[ann['image_id']].append(ann)

        # Store collected results
        self.results = []

    def process(self, data_batch, data_samples):
        """
        Process batch of data samples and store predictions.
        :param data_batch: The input batch (not needed for evaluation).
        :param data_samples: List of detection results.
        """
        for sample in data_samples:
            image_id = sample["img_id"]
            bboxes = sample["pred_instances"]["bboxes"].tolist()
            scores = sample["pred_instances"]["scores"].tolist()
            category_ids = sample["pred_instances"]["labels"].tolist()  # Ensure category_id is included

            for bbox, score, category_id in zip(bboxes, scores, category_ids):
                self.results.append({
                    "image_id": image_id,
                    "bbox": bbox,
                    "score": score,
                    "category_id": category_id  # Required for COCO evaluation
                })

    def compute_metrics(self, results):
        """
        Compute the final evaluation metrics (COCO mAP + Miss Rate).
        :return: Dictionary with evaluation scores.
        """
        if len(self.results) == 0:
            raise ValueError("No detection results were collected during evaluation!")

        # Compute COCO mAP using MMDetection's built-in metric
        coco_metric = CocoMetric(ann_file=self.ann_file)
        coco_results = coco_metric.compute_metrics(self.results)  # Use collected results

        # Compute Miss Rate
        miss_rate_metrics = self.compute_miss_rate(self.results)

        # Combine results
        evaluation_metrics = {
            "COCO_mAP": coco_results,
            "Miss_Rate": miss_rate_metrics
        }
        
        # Reset results after evaluation to avoid memory issues
        self.results = []
        
        return evaluation_metrics

    def compute_miss_rate(self, results):
        """
        Compute Log-Average Miss Rate (LAMR) at different False Positives Per Image (FPPI).
        :param results: List of detection results.
        :return: Dictionary containing FPPI, Miss Rate, and LAMR.
        """
        total_images = len(self.image_ids)
        tp_list = []
        fp_list = []
        num_gt = sum(len(self.annotations[img_id]) for img_id in self.image_ids)

        for img_id in self.image_ids:
            gt_boxes = [ann["bbox"] for ann in self.annotations[img_id]]
            det_boxes = [det["bbox"] for det in results if det["image_id"] == img_id]
            det_scores = [det["score"] for det in results if det["image_id"] == img_id]

            if not det_boxes:
                fp_list.append(len(gt_boxes))  # All ground truths are missed
                tp_list.append(0)
                continue

            # Sort detections by confidence score
            sorted_indices = np.argsort(det_scores)[::-1]
            det_boxes = [det_boxes[i] for i in sorted_indices]

            # Compute IoUs
            matches, false_positives = self.match_detections(gt_boxes, det_boxes)
            tp_list.append(matches)
            fp_list.append(false_positives)

        # Compute cumulative TP and FP
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        fppi = fp_cumsum / total_images
        miss_rate = 1 - (tp_cumsum / num_gt)

        # Define standard FPPI thresholds used for LAMR
        fppi_thresholds = np.array([0.01, 0.0178, 0.0316, 0.0562, 0.1, 0.1778, 0.3162, 0.5623, 1.0])

        # Interpolate Miss Rate at given FPPI thresholds
        interpolated_mr = np.interp(fppi_thresholds, fppi, miss_rate, left=1, right=miss_rate[-1])

        # Avoid log(0) by clipping values
        interpolated_mr = np.clip(interpolated_mr, 1e-6, 1.0)

        # Compute Log-Average Miss Rate (LAMR)
        lamr = np.exp(np.mean(np.log(interpolated_mr)))

        return {
            "FPPI": fppi.tolist(),
            "Miss_Rate": miss_rate.tolist(),
            "LAMR": lamr
        }

    def match_detections(self, gt_boxes, det_boxes):
        """
        Match detections to ground truth using IoU.
        :param gt_boxes: List of ground truth bounding boxes.
        :param det_boxes: List of detected bounding boxes.
        :return: Number of true positives and false positives.
        """
        if len(gt_boxes) == 0:
            return 0, len(det_boxes)  # No ground truth, all detections are false positives

        ious = self.compute_iou_matrix(gt_boxes, det_boxes)
        matches = 0
        used_gt = set()

        for det_idx, iou_values in enumerate(ious.T):
            best_match = np.argmax(iou_values)
            if iou_values[best_match] >= self.iou_threshold and best_match not in used_gt:
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

        inter_x1 = np.maximum(det_x1[:, None], gt_x1)
        inter_y1 = np.maximum(det_y1[:, None], gt_y1)
        inter_x2 = np.minimum(det_x2[:, None], gt_x2)
        inter_y2 = np.minimum(det_y2[:, None], gt_y2)

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union_area = (det_w * det_h)[:, None] + (gt_w * gt_h) - inter_area
        return inter_area / np.maximum(union_area, 1e-6)