import glob
import os
import cv2
import numpy as np
import natsort
import open3d as o3d
import matplotlib.pyplot as plt


def addPadding2FitRatio(img, ratio = (4,3)):
    h, w = img.shape[:2]
    if h == 576 and w == 56:
        img = img[100:-100]
        img = cv2.resize(img, (50,300))
        #min max norm
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)
        h, w = img.shape[:2]
    if h == 192 and w == 56:
        img = img[30:-30]
        h,w = img.shape[:2]
    new_h = h
    new_w = w
    if w/h > ratio[1]/ratio[0]:
        new_h = h  # Keep original height
        new_w = h * ratio[1] / ratio[0]  # Adjust width based on height
    else:
        new_w = w  # Keep original width
        new_h = w * ratio[0] / ratio[1]  # Adjust height based on width
    pad_h = max(0, int((new_h - h) / 2))  # Ensure padding is non-negative
    pad_w = max(0, int((new_w - w) / 2))  # Ensure padding is non-negative
    img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0,0,0])
    return img

def process_pcd(pcd_path):
    """
    Process PCD file and save visualization without displaying.
    Returns the RGB image as a numpy array.
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    # Create a visualizer object in headless mode
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Create invisible window
    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)
    # Get render options and set background color
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # Black background
    
    # Set view control for desired view
    ctr = vis.get_view_control()
    ctr.set_front([-5, 0, -1])  # Looking down along z-axis
    ctr.set_up([0, 0.5, 0])     # Up direction is y-axis
    ctr.set_lookat([0, 0, 0])   # Look at center
    ctr.set_zoom(0.1)           # Adjust zoom level
    # Update visualization
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    # Capture the screen as a numpy array
    image_float_buffer = vis.capture_screen_float_buffer(do_render=True)
    rgb_image = (np.asarray(image_float_buffer) * 255).astype(np.uint8)
    rgb_image = rgb_image[:,480:-480,:]
    rgb_image = cv2.resize(rgb_image, (120, 160))
    vis.destroy_window()
    
    return rgb_image

def read_data(folder_pth, folder_list):
    modal_images = []
    for folder in folder_list:
        images = []
        modal_folder = os.path.join(folder_pth, folder)
        if folder =='ml_pointcloud':
            pcd_list = natsort.natsorted(glob.glob(os.path.join(modal_folder, '*.pcd')))
            for pcd_pth in pcd_list:
                pcd_img = process_pcd(pcd_pth)
                images.append(pcd_img)
        else:
            img_list = natsort.natsorted(glob.glob(os.path.join(modal_folder, '*.png')))
            for img_pth in img_list:
                img = cv2.imread(img_pth)
                img = addPadding2FitRatio(img)
                img = cv2.resize(img, (120, 160))  # Ensure all images are resized to the same dimensions
                images.append(img)

        modal_images.append(images)
    modal_images = np.array(modal_images, dtype=object)  # Use dtype=object to handle varying shapes
    return modal_images

        

def main():
    folder_pth = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/241108-indoor-gist/source_data/drone_2024-11-05-16-14-27_0'
    # folder_list = ['ir_image_raw', 'ml_ambient_color', 'aligned_rgb', 'ml_depth_color', 'ml_intensity_color']
    folder_list = ['ir_image_raw', 'ml_ambient_color', 'aligned_rgb', 'ml_depth_color', 'ml_intensity_color', 'ml_pointcloud']
    save_pth = '/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/MISC/data_collection2.mp4'

    img_lists = read_data(folder_pth, folder_list)
    len_img = len(img_lists[0])
    frames = []
    for i in range(len_img):
        ir = img_lists[0][i].astype(np.uint8)
        ml_ambient = img_lists[1][i].astype(np.uint8)
        aligned_rgb = img_lists[2][i].astype(np.uint8)
        ml_depth = img_lists[3][i].astype(np.uint8)
        ml_intensity = img_lists[4][i].astype(np.uint8)
        # black = np.zeros((160, 120, 3), dtype=np.uint8)

        ml_pointcloud = img_lists[5][i].astype(np.uint8)
        ml_pointcloud = cv2.resize(ml_pointcloud, (120, 160))  # Ensure pointcloud image has the same dimensions as others

        '''concat images to 3 x 2'''
        img = np.concatenate((ir, aligned_rgb, ml_ambient), axis=1)
        img2 = np.concatenate((ml_depth, ml_intensity, ml_pointcloud), axis=1)
        out = np.concatenate((img, img2), axis=0)
        frames.append(out)
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = cv2.VideoWriter(save_pth, fourcc, 5.0, (out.shape[1], out.shape[0]))  # Create VideoWriter object
    for frame in frames:
        video_writer.write(frame)  # Write each frame to the video

    video_writer.release()  # Release the VideoWriter object
    print(f"Video saved to {save_pth}")

main()