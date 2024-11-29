import cv2
import numpy as np

def apply_motion_blur(image_path, save_path, kernel_size=50):
    # Create the motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size

    # Read the image
    image = cv2.imread(image_path)

    # Apply the motion blur
    blurred_image = cv2.filter2D(image, -1, kernel)

    # Save the blurred image
    cv2.imwrite(save_path, blurred_image)

# Example usage

if __name__ == "__main__":
    img_pth = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/241108-indoor-gist/source_data/drone_2024-11-05-16-21-24_0/aligned_rgb/0013.png'
    save_pth = '/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/drone-mmdetection-jm/MISC/blur.png'
    apply_motion_blur(img_pth, save_pth)


