#%%
import os
import json
import cv2
from pathlib import Path
from datetime import datetime

# Paths
video_path = "/media/ailab/HDD1/Workspace/Download/Drone-detection/241108-indoor-gist/video/drone_2024-11-05-15-52-11_0.mp4"
json_annotation_path = "/media/ailab/HDD1/Workspace/Download/Drone-detection/241108-indoor-gist/ann/drone_2024-11-05-15-52-11_0.mp4.json"
output_dir = Path("/media/ailab/HDD1/Workspace/Download/Drone-detection/241108-indoor-gist/output")
#%%
mapped_category = {
    'human': 1,
    'Door': 2,
    'fire extinguisher': 3,
    'exit pannel': 4,
    'window': 5,
    # Add other categories as needed
}


def supervisely_video2coco(video_path, json_annotation_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = video_path.stem
# Load annotations
    with open(json_annotation_path) as f:
        annotations_data = json.load(f)

    total_cfame = annotations_data['framesCount']

    # COCO format dictionary
    coco_data = {
        "info": {
            "description": "Converted dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    objects = annotations_data["objects"]
    supervisely_categories = {obj["key"]: obj["classTitle"] for obj in objects}  # value of key and classTitle
    
    coco_data["categories"] = [{"id": idx, "name": name} for name, idx in mapped_category.items()]

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    annotation_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame_filename = f"{video_name}/{frame_index:06d}.jpg"
        frame_filename = f"rgb_images/{frame_index:06d}.jpg"
        # frame_path = output_dir / video_name / frame_filename
        frame_path = output_dir / frame_filename
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_path), frame)
        
        # Add image information to COCO
        coco_data["images"].append({
            "id": frame_index,
            "file_name": frame_filename,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "date_captured": datetime.now().isoformat()
        })
        

        # Extract annotations for this frame
        frame_annotations = next((frame for frame in annotations_data["frames"] if frame["index"] == frame_index), None)
        if frame_annotations:
            for figure in frame_annotations["figures"]:
                class_key = figure['objectKey']
                category_name = supervisely_categories[class_key]
                class_id = mapped_category[category_name]
                x1, y1 = figure["geometry"]["points"]["exterior"][0]
                x2, y2 = figure["geometry"]["points"]["exterior"][1]
                width, height = x2 - x1, y2 - y1
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": frame_index,
                    "category_id": class_id,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1

        frame_index += 1

    cap.release()

    # Save the COCO annotation file
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"COCO dataset saved to {output_dir}")
# %%
def main():
    root = Path("/media/ailab/HDD1/Workspace/dset/Drone-Detection-Custom/241108-indoor-gist/")
    videos = list(root.glob("video/*.mp4"))
    for video_path in videos:
        json_annotation_path = root / "ann" / (video_path.stem+'.mp4' + ".json")
        output_dir = root / "output" / video_path.stem
        supervisely_video2coco(video_path, json_annotation_path, output_dir)

if __name__ == "__main__":
    main()

# %%
