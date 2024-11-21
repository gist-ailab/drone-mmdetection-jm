#%%
import random
import shutil
import glob
import os

root = '/media/ailab/HDD1/Workspace/Download/Drone-detection/241108-indoor-gist/data_tmp'
save_root = '/media/ailab/HDD1/Workspace/Download/Drone-detection/241108-indoor-gist/output/thermal_images'
os.makedirs(save_root, exist_ok=True)

folders = glob.glob(os.path.join(root, '*', 'ir_image_raw'), recursive=True)

# %%
for folder in folders:
    for file in glob.glob(os.path.join(folder, '*.png')):
        file_name = os.path.basename(file)
        file_name = f"{int(file_name.split('.')[0]):06}.png"
        save_name = os.path.basename(os.path.dirname(os.path.dirname(file))) + '_' + file_name
        shutil.copy(file, os.path.join(save_root, save_name))
# %%
 