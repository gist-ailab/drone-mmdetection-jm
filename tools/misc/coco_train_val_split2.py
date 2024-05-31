#%%
import json
import pandas as pd
from sklearn.model_selection import train_test_split


#%%
# Load your COCO dataset
save_path = '/ailab_mat/dataset/FLIR_ADAS_v2/video_rgb_test'
with open(save_path+'/coco.json', 'r') as file:
    coco = json.load(file)

#%%

# Create a DataFrame for images
images_df = pd.DataFrame(coco['images'])

# Split the images into train and validation sets
train_images, val_images = train_test_split(images_df, test_size=0.2, random_state=42)
#%%
# Function to filter annotations for selected images
def filter_annotations(images, annotations):
    image_ids = set(images['id'])
    return [anno for anno in annotations if anno['image_id'] in image_ids]

# Filter annotations for each dataset
train_annotations = filter_annotations(train_images, coco['annotations'])
val_annotations = filter_annotations(val_images, coco['annotations'])
#%%

# Optionally re-id images and annotations
def re_id_images_annotations(images, annotations):
    # Re-assign image IDs
    id_map = {old_id: i + 1 for i, old_id in enumerate(images['id'])}
    images['id'] = images['id'].map(id_map)

    # Re-assign annotation IDs and update their image references
    for i, anno in enumerate(annotations):
        anno['id'] = i + 1
        anno['image_id'] = id_map[anno['image_id']]
    return images, annotations

train_images, train_annotations = re_id_images_annotations(train_images, train_annotations)
val_images, val_annotations = re_id_images_annotations(val_images, val_annotations)
#%%
# Create new COCO JSON files
def create_coco_json(images, annotations, categories, file_name):
    with open(file_name, 'w') as f:
        json.dump({'images': images, 'annotations': annotations, 'categories': coco['categories'], 'info': coco['info']}, f)

# Save new COCO files
create_coco_json(train_images.to_dict(orient='records'), train_annotations, coco['categories'], save_path+'/train_coco_tmp.json')
create_coco_json(val_images.to_dict(orient='records'), val_annotations, coco['categories'],  save_path+'/test_coco_tmp.json')
# %%
