import cv2
import numpy as np
import os
import glob
import natsort
from tqdm import tqdm
import argparse

class  Folder2Video():
    def __init__(self, folder_pth, frame):
         self.folder_pth = folder_pth
         self.frame = frame
         #parent folder path of folder pth
         self.save_pth = os.path.join(os.path.dirname(self.folder_pth), os.path.basename(os.path.dirname(self.folder_pth))+'_'+os.path.basename(folder_pth) + '.mp4')

    def make_video(self):
        img_list = glob.glob(os.path.join(self.folder_pth, '*.png'))
        img_list = natsort.natsorted(img_list)
        frame_list = []
        for img_pth in img_list:
            img = cv2.imread(img_pth)
            frame_list.append(img)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(self.save_pth, fourcc, self.frame, (frame_list[0].shape[1], frame_list[0].shape[0]))
        for frame in tqdm(frame_list):
            out.write(frame)
        out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_pth', type = str, default = '/ailab_mat/personal/maeng_jemo/Project/24-Drone/Detection/mmdetection-drone-jemo/work_dirs/faster-rcnn_r50_fpn_2x_flair-adas-rgb-v3/inference_val_w_gt')
    parser.add_argument('--frame', type = int, default = 5)
    args = parser.parse_args()

    folder2video = Folder2Video(args.folder_pth, args.frame)
    folder2video.make_video()


if __name__ == '__main__':
    main()