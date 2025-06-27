import cv2
import os

def save_image(image, filename):
    save_path = os.path.join(os.getcwd(), filename)  # 현재 디렉토리에 저장
    success = cv2.imwrite(save_path, image)
    if success:
        print(f"Image saved to: {save_path}")
    else:
        print("Failed to save image.")

if __name__ == "__main__":
    image_path = '/SSDb/jemo_maeng/dset/DELIVER/img/cloud/train/MAP_1_point102/110100_rgb_front.png'

    if os.path.isfile(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            filename = 'saved_image.png'  # 저장할 파일 이름
            save_image(image, filename)
        else:
            print(f"Failed to load image: {image_path}")
    else:
        print(f"File does not exist: {image_path}")
