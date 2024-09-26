#%%
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Load COCO dataset
save_ir_path = '/ailab_mat/dataset/FLIR_ADAS_v2/video_thermal_test'
with open(f'{save_ir_path}/coco.json', 'r') as file:
    coco = json.load(file)

save_rgb_pth = '/ailab_mat/dataset/FLIR_ADAS_v2/video_rgb_test'
with open(f'{save_rgb_pth}/coco.json', 'r') as file:
    coco_rgb = json.load(file)

mapping_pth = '/ailab_mat/dataset/FLIR_ADAS_v2/rgb_to_thermal_vid_map.json'
with open(mapping_pth, 'r') as file:
    mapping = json.load(file)

# Create DataFrame from RGB images
rgb_images_df = pd.DataFrame(coco_rgb['images'])
train_rgb_images, val_rgb_images = train_test_split(rgb_images_df, test_size=0.2, random_state=42)


thermal_images_df = pd.DataFrame(coco['images'])
def get_paired_thermal_images(rgb_images, mapping):
    thermal_images = []
    for rgb_image in rgb_images:
        file_name = rgb_image['file_name'].split('/')[-1]
        thermal_file = mapping.get(file_name)
        if thermal_file:
            thermal_images.append('data/' + thermal_file)
        else:
            print(f"No mapping found for {file_name}")
    return thermal_images

train_thermal_images = get_paired_thermal_images(train_rgb_images.to_dict(orient='records'), mapping)
val_thermal_images = get_paired_thermal_images(val_rgb_images.to_dict(orient='records'), mapping)

file_name_to_id = {image['file_name']: image['id'] for image in coco['images']}

def get_image_ids_from_filenames(file_names):
    image_ids = []
    for file_name in file_names:
        if file_name in file_name_to_id:
            image_ids.append(file_name_to_id[file_name])
        else:
            print(f"File name {file_name} not found in dataset")
    return image_ids

train_thermal_ids = get_image_ids_from_filenames(train_thermal_images)
val_thermal_ids = get_image_ids_from_filenames(val_thermal_images)
# %%
def filter_annotations(images, annotations):
    image_ids = set(images)
    return [anno for anno in annotations if anno['image_id'] in image_ids]

train_thermal_annotations = filter_annotations(train_thermal_ids, coco['annotations'])
val_thermal_annotations = filter_annotations(val_thermal_ids, coco['annotations'])

#%%
def new_coco(selected_ids):
    selected_images = [img for img in coco['images'] if img['id'] in selected_ids]
    return {
        'images': selected_images,
        'annotations': [anno for anno in coco['annotations'] if anno['image_id'] in selected_ids],
        'categories': coco['categories'],
        'info': coco['info']
    }

train_thermal_coco = new_coco(train_thermal_ids)
val_thermal_coco = new_coco(val_thermal_ids)


# %%
def re_id_images_annotations(images, annotations):
    # Create a mapping from old IDs to new IDs
    id_map = {image['id']: i + 1 for i, image in enumerate(images)}
    
    # Update the image IDs
    for image in images:
        image['id'] = id_map[image['id']]
    
    # Re-assign annotation IDs and update their image references
    for i, anno in enumerate(annotations):
        anno['id'] = i + 1
        if anno['image_id'] in id_map:
            anno['image_id'] = id_map[anno['image_id']]
    
    return images, annotations

#%%

train_images, train_annotations = re_id_images_annotations(train_thermal_coco['images'], train_thermal_coco['annotations'])
val_images, val_annotations = re_id_images_annotations(val_thermal_coco['images'],val_thermal_coco['annotations'])
# %%
def create_coco_json(images, annotations, categories, file_name):
    with open(file_name, 'w') as f:
        json.dump({'images': images, 'annotations': annotations, 'categories': categories, 'info': coco['info']}, f)

# Save new COCO files
save_thermal_path = '/ailab_mat/dataset/FLIR_ADAS_v2/video_thermal_test'
create_coco_json(train_images, train_annotations, coco['categories'], f'{save_thermal_path}/train_coco_thermal_v4.json')
create_coco_json(val_images, val_annotations, coco['categories'], f'{save_thermal_path}/test_coco_thermal_v4.json')
# %%
