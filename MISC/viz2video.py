import os
import glob
import cv2
import natsort


def folder_cluster(img_files):
    '''
    Cluster foler by name
        file name : drone_2024-11-05-16-13-04_0_000186.jpg
        folder  : drone_2024-11-05-16-13-04_0
        return : list of folders
    '''
    folders = []

    for img_file in img_files:
        folder_parent = os.path.dirname(img_file)
        folder_name_list = os.path.basename(img_file).split('_')
        folder_name = '_'.join(folder_name_list[:-1])
        # folder = os.path.join(folder_parent, folder_name)
        if folder_name not in folders:
            folders.append(folder_name)
    return folders

def folder2video(img_folder, folder_name, output_dir, fps):
    # img_files = natsort.natsorted(glob.glob(os.path.join(folder, '*.')))
    img_files = natsort.natsorted(glob.glob(os.path.join(img_folder, f'{folder_name}_*.jpg')))
    img = cv2.imread(img_files[0])
    h, w, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(f'{output_dir}/{folder_name}.mp4', fourcc, fps, (w, h))
    for img_file in img_files:
        img = cv2.imread(img_file)
        video.write(img)
    video.release()
    print(f'Video saved to {folder_name}.mp4')



def viz2video(img_folder, fps=10):
    img_files = sorted(glob.glob(os.path.join(img_folder, '*')))
    output_dir = os.path.join(os.path.dirname(img_folder), 'viz_video')
    os.makedirs(output_dir, exist_ok=True)
    img_folders = folder_cluster(img_files)
    for folder_name in img_folders:
        folder2video(img_folder, folder_name, output_dir, fps)


if __name__ == '__main__':
    pth = '/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/faster-rcnn_r101_fpn_2x_GISTindoor_splitRandom_lr0.001/20241122_122643/imgs_epoch4'
    viz2video(pth)