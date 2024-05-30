#! Code form https://github.com/gist-ailab/sj-mmd/blob/main/tools/analysis_tools/statistic_dataset.py

import json
from collections import defaultdict

def load_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def get_coco_statistics(coco_data):
    # Initialize statistics
    stats = {
        'num_images': 0,
        'num_annotations': 0,
        'num_objects_per_category': defaultdict(int)
    }
    
    # Get number of images
    stats['num_images'] = len(coco_data['images'])
    
    # Get number of annotations
    stats['num_annotations'] = len(coco_data['annotations'])
    
    # Count objects per category
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        stats['num_objects_per_category'][category_id] += 1
    
    return stats

def print_statistics(stats, coco_data):
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Number of images: {stats['num_images']}")
    print(f"Number of annotations: {stats['num_annotations']}")
    # Number of valid categories
    print(f"Number of valid categories: {len(stats['num_objects_per_category'])}")
    print("Number of objects per category:")
    category_list = []
    for category_id, count in stats['num_objects_per_category'].items():
        category_name = categories.get(category_id, "Unknown")
        category_list.append((category_name))
        print(f"  {category_name} {category_id}: {count}")
    print(f"{category_list}")


if __name__ == "__main__":
    annotation_file = '/ailab_mat/dataset/FLIR_ADAS_v2/video_rgb_test/coco.json'
    # annotation_file = '/SSDc/Workspaces/seongju_lee/dset/FLIR_ADAS_v2/video_rgb_test/coco.json'
    coco_data = load_coco_annotations(annotation_file)
    stats = get_coco_statistics(coco_data)
    print_statistics(stats, coco_data)