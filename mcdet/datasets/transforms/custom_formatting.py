from typing import Optional, Sequence

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, ReIDDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes
from mmdet.datasets.transforms.formatting import PackDetInputs
import torch


# @TRANSFORMS.register_module()
# class FLIR_CATPackDetInputs(PackDetInputs):
#     """Pack the inputs data for the detection / semantic segmentation /
#     panoptic segmentation.

#     The ``img_meta`` item is always populated.  The contents of the
#     ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

#         - ``img_id``: id of the image

#         - ``img_path``: path to the image file

#         - ``ori_shape``: original shape of the image as a tuple (h, w)

#         - ``img_shape``: shape of the image input to the network as a tuple \
#             (h, w).  Note that images may be zero padded on the \
#             bottom/right if the batch tensor is larger than this shape.

#         - ``scale_factor``: a float indicating the preprocessing scale

#         - ``flip``: a boolean indicating if image flip transform was used

#         - ``flip_direction``: the flipping direction

#     Args:
#         meta_keys (Sequence[str], optional): Meta keys to be converted to
#             ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
#             Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
#             'scale_factor', 'flip', 'flip_direction')``
#     """
#     mapping_table = {
#         'gt_bboxes': 'bboxes',
#         'gt_bboxes_labels': 'labels',
#         'gt_masks': 'masks'
#     }

#     def __init__(self,
#                  meta_keys=('img_id', 'img_path','thermal_img_path' 'ori_shape', 'img_shape',
#                             'scale_factor', 'flip', 'flip_direction')):
#         self.meta_keys = meta_keys

#     def transform(self, results: dict) -> dict:
#         """Method to pack the input data.

#         Args:
#             results (dict): Result dict from the data pipeline.

#         Returns:
#             dict:

#             - 'inputs' (obj:`torch.Tensor`): The forward data of models.
#             - 'data_sample' (obj:`DetDataSample`): The annotation info of the
#                 sample.
#         """
#         packed_results = dict()
#         if 'img' in results:
#             img = results['img']
#             if len(img.shape) < 3:
#                 img = np.expand_dims(img, -1)
#             # To improve the computational speed by by 3-5 times, apply:
#             # If image is not contiguous, use
#             # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
#             # If image is already contiguous, use
#             # `torch.permute()` followed by `torch.contiguous()`
#             # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
#             # for more details
#             if not img.flags.c_contiguous:
#                 img = np.ascontiguousarray(img.transpose(2, 0, 1))
#                 img = to_tensor(img)
#             else:
#                 img = to_tensor(img).permute(2, 0, 1).contiguous()

#             packed_results['inputs'] = img
        
#         if 'thermal_img' in results:
#             thermal_img = results['thermal_img']
#             if len(thermal_img.shape) < 3:
#                 thermal_img = np.expand_dims(thermal_img, -1)
#             if not thermal_img.flags.c_contiguous:
#                 thermal_img = np.ascontiguousarray(thermal_img.transpose(2, 0, 1))
#                 thermal_img = to_tensor(thermal_img)
#             else:
#                 thermal_img = to_tensor(thermal_img).permute(2, 0, 1).contiguous()

#             packed_results['thermal_inputs'] = thermal_img

#         if 'gt_ignore_flags' in results:
#             valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
#             ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

#         data_sample = DetDataSample()
#         instance_data = InstanceData()
#         ignore_instance_data = InstanceData()

#         for key in self.mapping_table.keys():
#             if key not in results:
#                 continue
#             if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
#                 if 'gt_ignore_flags' in results:
#                     instance_data[
#                         self.mapping_table[key]] = results[key][valid_idx]
#                     ignore_instance_data[
#                         self.mapping_table[key]] = results[key][ignore_idx]
#                 else:
#                     instance_data[self.mapping_table[key]] = results[key]
#             else:
#                 if 'gt_ignore_flags' in results:
#                     instance_data[self.mapping_table[key]] = to_tensor(
#                         results[key][valid_idx])
#                     ignore_instance_data[self.mapping_table[key]] = to_tensor(
#                         results[key][ignore_idx])
#                 else:
#                     instance_data[self.mapping_table[key]] = to_tensor(
#                         results[key])
#         data_sample.gt_instances = instance_data
#         data_sample.ignored_instances = ignore_instance_data

#         if 'proposals' in results:
#             proposals = InstanceData(
#                 bboxes=to_tensor(results['proposals']),
#                 scores=to_tensor(results['proposals_scores']))
#             data_sample.proposals = proposals

#         if 'gt_seg_map' in results:
#             gt_sem_seg_data = dict(
#                 sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
#             gt_sem_seg_data = PixelData(**gt_sem_seg_data)
#             if 'ignore_index' in results:
#                 metainfo = dict(ignore_index=results['ignore_index'])
#                 gt_sem_seg_data.set_metainfo(metainfo)
#             data_sample.gt_sem_seg = gt_sem_seg_data

#         img_meta = {}
#         for key in self.meta_keys:
#             if key in results:
#                 img_meta[key] = results[key]
#         data_sample.set_metainfo(img_meta)
#         packed_results['data_samples'] = data_sample

#         concat = torch.cat([packed_results['inputs'], packed_results['thermal_inputs']], dim=0)
#         packed_results['inputs'] = concat

#         return packed_results

@TRANSFORMS.register_module()
class FLIR_CATPackDetInputs(PackDetInputs):
    mapping_table={
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }
    def __init__(self, meta_keys=('img_id', 'img_path', 'thermal_img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        
        # Handle RGB image
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()
            packed_results['inputs'] = img
        
        # Handle thermal image
        if 'thermal_img' in results:
            thermal_img = results['thermal_img']
            if len(thermal_img.shape) < 3:
                thermal_img = np.expand_dims(thermal_img, -1)
            if not thermal_img.flags.c_contiguous:
                thermal_img = np.ascontiguousarray(thermal_img.transpose(2, 0, 1))
                thermal_img = to_tensor(thermal_img)
            else:
                thermal_img = to_tensor(thermal_img).permute(2, 0, 1).contiguous()
            packed_results['thermal_inputs'] = thermal_img
        
        # Handle ground truth ignore flags if present
        valid_idx = None
        ignore_idx = None
        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        # Initialize data samples
        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if valid_idx is not None and ignore_idx is not None:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if valid_idx is not None and ignore_idx is not None:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key])
        
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        # Handle proposals if present
        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        # Handle ground truth segmentation map if present
        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        # Pack image metadata
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        # Concatenate RGB and thermal images along the channel dimension
        inputs = packed_results['inputs']
        thermal_inputs = packed_results['thermal_inputs']
        concat = torch.cat([inputs, thermal_inputs, thermal_inputs], dim=0)  # Use dim=1 for channels
        packed_results['inputs'] = concat

        return packed_results


@TRANSFORMS.register_module()
class PackMultiModalDetInputs(BaseTransform):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs_rgb'] = img
        
        if 'ir' in results:
            ir = results['ir']
            if len(ir.shape) < 3:
                ir = np.expand_dims(ir, -1)
            if not ir.flags.c_contiguous:
                ir = np.ascontiguousarray(ir.transpose(2, 0, 1))
                ir = to_tensor(ir)
            else:
                ir = to_tensor(ir).permute(2, 0, 1).contiguous()

            packed_results['inputs_ir'] = ir

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results