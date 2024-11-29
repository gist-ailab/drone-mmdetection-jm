import cv2
import numpy as np
import os

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

# Function to generate a box with the desired IoU
def generate_box_with_iou(reference_box, target_iou, image_size):
    ref_x1, ref_y1, ref_x2, ref_y2 = reference_box
    box_width = ref_x2 - ref_x1
    box_height = ref_y2 - ref_y1

    for _ in range(1000):  # Try multiple times to find a matching box
        shift_x = np.random.uniform(-box_width / 2, box_width / 2)
        shift_y = np.random.uniform(-box_height / 2, box_height / 2)

        candidate_box = [
            ref_x1 + shift_x,
            ref_y1 + shift_y,
            ref_x2 + shift_x,
            ref_y2 + shift_y,
        ]
        candidate_box = [
            max(0, candidate_box[0]),
            max(0, candidate_box[1]),
            min(image_size[1], candidate_box[2]),
            min(image_size[0], candidate_box[3]),
        ]
        iou = calculate_iou(reference_box, candidate_box)
        if abs(iou - target_iou) < 0.01:  # Allowable error
            return candidate_box
    return None

# Create visualization
def visualize_bboxes(image_size, reference_box, pairs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for iou, pair_box in pairs:
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        # Draw reference box in green
        cv2.rectangle(image, 
                      (int(reference_box[0]), int(reference_box[1])), 
                      (int(reference_box[2]), int(reference_box[3])),
                      (0, 255, 0), 2)
        # Draw pair box in red
        cv2.rectangle(image, 
                      (int(pair_box[0]), int(pair_box[1])), 
                      (int(pair_box[2]), int(pair_box[3])),
                      (0, 0, 255), 2)

        # Add IoU text
        text = f"IoU: {iou:.1f}"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save image
        cv2.imwrite(os.path.join(output_dir, f"iou_{iou:.1f}.png"), image)

# Main process
if __name__ == "__main__":
    image_size = (480, 640)  # Height, Width
    reference_box = [
        image_size[1] // 4,  # x1
        image_size[0] // 4,  # y1
        image_size[1] * 3 // 4,  # x2
        image_size[0] * 3 // 4,  # y2
    ]

    output_dir = "./bbox_visualizations"
    pairs = []

    for iou_target in np.arange(0.3, 1.0, 0.1):
        box = generate_box_with_iou(reference_box, iou_target, image_size)
        if box is not None:
            pairs.append((iou_target, box))

    visualize_bboxes(image_size, reference_box, pairs, output_dir)
