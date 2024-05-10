# This is a demo for using the mmdetection library to perform object detection using the DETR model

from mmdet.apis import DetInferencer

inferencer = DetInferencer(model ='detr_r50_8xb2-150e_coco', device='cpu')
models = DetInferencer.list_models('mmdet')
print(models)
# inferencer('demo/demo.jpg', out_dir='demo/demo_out')
pred = inferencer('demo/demo.jpg')
print(pred)