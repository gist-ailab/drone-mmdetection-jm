import cv2
import json
import os

palette = {
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 0, 0)
}

def visualize_image_with_bbox(image_folder, annotation_path):
    # Load the annotation
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    # Get image and annotation IDs
    image_ids = [img['id'] for img in annotations['images']]
    annotation_ids = [ann['id'] for ann in annotations['annotations']]

    # Initialize current image index
    current_image_index = 0

    while True:
        # Get current image ID
        current_image_id = image_ids[current_image_index]

        # Get current image path
        current_image = next(img for img in annotations['images'] if img['id'] == current_image_id)
        current_image_path = os.path.join(image_folder, current_image['file_name_RGB'])
        current_image_path_IR = os.path.join(image_folder, current_image['file_name_IR'])

        # Load the image
        img = cv2.imread(current_image_path)
        img_IR = cv2.imread(current_image_path_IR)
        # Draw bounding boxes
        for ann in annotations['annotations']:
            if ann['image_id'] == current_image_id:
                x1 = int(ann['bbox'][0])
                y1 = int(ann['bbox'][1])
                w = int(ann['bbox'][2])
                h = int(ann['bbox'][3])
                category = ann['category_id']
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(img, (x1, y1), (x2, y2), palette[category], 2)
                cv2.rectangle(img_IR, (x1, y1), (x2, y2), palette[category], 2)
        # Display the image
        cv2.imshow('Image', cv2.hconcat([img, img_IR]))

        # Wait for key press
        key = cv2.waitKey(0)

        # Move to next image
        if key == ord('n'):
            current_image_index = (current_image_index + 1) % len(image_ids)
        # Move to previous image
        elif key == ord('p'):
            current_image_index = (current_image_index - 1) % len(image_ids)
        # Quit
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    image_folder = '/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/SOTA/CAFF-DETR/CAFF-DINO/data/kaist_coco'
    annotation_path = '/media/ailab/HDD1/Workspace/src/Project/Drone24/detection/SOTA/CAFF-DETR/CAFF-DINO/data/kaist_coco/annotations/test-day-20.txt.json'

    visualize_image_with_bbox(image_folder, annotation_path)

if __name__ == "__main__":
    main()

