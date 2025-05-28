# 디버깅을 위한 커스텀 훅 추가
import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class BboxLossDebugHook(Hook):
    """Debug hook for bbox loss analysis"""
    
    def __init__(self, log_interval=50):
        self.log_interval = log_interval
        
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (batch_idx + 1) % self.log_interval == 0:
            # 현재 losses 분석
            losses = outputs.get('loss', {}) if outputs else {}
            
            # GT 정보 분석
            if data_batch and 'data_samples' in data_batch:
                data_samples = data_batch['data_samples']
                
                total_gt_boxes = 0
                bbox_sizes = []
                
                for sample in data_samples:
                    if hasattr(sample, 'gt_instances') and hasattr(sample.gt_instances, 'bboxes'):
                        bboxes = sample.gt_instances.bboxes
                        total_gt_boxes += len(bboxes)
                        
                        for bbox in bboxes:
                            w, h = bbox[2].item(), bbox[3].item()
                            bbox_sizes.append((w, h))
                
                if bbox_sizes:
                    widths = [w for w, h in bbox_sizes]
                    heights = [h for w, h in bbox_sizes]
                    areas = [w * h for w, h in bbox_sizes]
                    
                    runner.logger.info(
                        f"GT Analysis - Total boxes: {total_gt_boxes}, "
                        f"Avg width: {sum(widths)/len(widths):.2f}, "
                        f"Avg height: {sum(heights)/len(heights):.2f}, "
                        f"Avg area: {sum(areas)/len(areas):.2f}, "
                        f"Min area: {min(areas):.2f}, Max area: {max(areas):.2f}"
                    )
            
            # Loss 분석
            rpn_cls_loss = outputs.get('loss_rpn_cls', 0)
            rpn_bbox_loss = outputs.get('rpn_loss_bbox', 0)
            roi_cls_loss = outputs.get('loss_cls', 0)
            roi_bbox_loss = outputs.get('loss_bbox', 0)
            
            runner.logger.info(
                f"Loss Analysis - RPN cls: {rpn_cls_loss:.6f}, "
                f"RPN bbox: {rpn_bbox_loss:.6f}, "
                f"ROI cls: {roi_cls_loss:.6f}, "
                f"ROI bbox: {roi_bbox_loss:.6f}"
            )

