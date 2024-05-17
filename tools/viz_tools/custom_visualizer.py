import cv2
import numpy as np
import matplotlib.pyplot as plt
# color pallete for coco dataset
COLOR_PALETTE = [
    (220, 20, 60), (11, 129, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
    (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243),
    (45, 89, 255), (134, 134, 103), (145, 148, 174), (255, 208, 186), (197, 226, 255),
    (171, 134, 1), (109, 63, 54), (207, 138, 255), (151, 0, 95), (9, 80, 61),
    (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
    (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
    (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0), (119, 0, 170),
    (0, 182, 199), (0, 165, 120), (183, 130, 88), (95, 32, 0), (130, 114, 135),
    (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
    (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
    (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122), (191, 162, 208)
]

GT_COLOR_PALETTE = [
    (255, 50, 80), (21, 149, 52), (20, 20, 162), (20, 20, 250), (126, 20, 248),
    (20, 80, 120), (20, 100, 120), (20, 20, 90), (20, 20, 212), (255, 190, 50),
    (120, 190, 50), (240, 240, 20), (195, 136, 195), (255, 20, 50), (185, 62, 62),
    (255, 97, 255), (20, 246, 272), (202, 202, 255), (20, 102, 20), (140, 186, 177),
    (130, 96, 20), (194, 77, 255), (219, 120, 20), (92, 20, 138), (255, 199, 250),
    (20, 145, 112), (229, 20, 171), (208, 228, 202), (20, 240, 196), (255, 119, 184),
    (112, 20, 93), (153, 149, 255), (98, 200, 255), (20, 248, 20), (194, 255, 255),
    (65, 109, 255), (154, 154, 123), (165, 168, 194), (255, 228, 206), (217, 246, 255),
    (191, 154, 21), (129, 83, 74), (227, 158, 255), (171, 20, 115), (29, 100, 81),
    (104, 125, 71), (94, 85, 125), (186, 216, 122), (228, 215, 230), (255, 129, 85),
    (20, 163, 169), (199, 20, 214), (229, 119, 126), (25, 141, 20), (247, 255, 225),
    (167, 206, 228), (173, 89, 21), (23, 115, 181), (183, 255, 20), (139, 20, 190),
    (20, 202, 219), (20, 185, 140), (203, 150, 108), (115, 52, 20), (150, 134, 155),
    (130, 149, 153), (186, 94, 138), (239, 162, 205), (99, 230, 134), (198, 110, 82),
    (85, 90, 35), (147, 187, 135), (79, 125, 126), (162, 128, 65), (216, 192, 20),
    (115, 74, 100), (148, 96, 255), (221, 77, 21), (266, 20, 142), (211, 182, 228)
]

def vis_pred(img, result, classes, score_thr=0.5):
        # Get bbox from result
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()

    img_result = img.copy()

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        score = scores[i]
        label = labels[i]
        color = COLOR_PALETTE[label]
        if score < score_thr:
            cv2.rectangle(img_result,  (int(bbox[0]), int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)
            cv2.putText(img_result, f'{classes[label]}: {score:.2f}', (int(bbox[0])-2, int(bbox[1])-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_result


def vis_pred_w_gt(img, result, gt, classes, gt_classes, score_thr=0.5):
    
    p_bboxes = result.pred_instances.bboxes.cpu().numpy()
    p_scores = result.pred_instances.scores.cpu().numpy()
    p_labels = result.pred_instances.labels.cpu().numpy()

    g_bboxes = [ann['bbox'] for ann in gt]
    g_labels = [ann['category_id'] for ann in gt]

    img_result = img.copy()

    for i in range(len(p_bboxes)):
        bbox = p_bboxes[i]
        score = p_scores[i]
        label = p_labels[i]
        color = COLOR_PALETTE[label]
        if score < score_thr:
            cv2.rectangle(img_result,  (int(bbox[0]), int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)
            cv2.putText(img_result, f'Pred: {classes[label]}: {score:.2f}', (int(bbox[0])-2, int(bbox[1])-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for i in range(len(g_bboxes)):
        bbox = g_bboxes[i]
        label = g_labels[i]
        color = GT_COLOR_PALETTE[label]
        cv2.rectangle(img_result,  (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), color, 1)
        cv2.putText(img_result, f'GT: {gt_classes[label]}', (int(bbox[0])-2, int(bbox[1])-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img_result