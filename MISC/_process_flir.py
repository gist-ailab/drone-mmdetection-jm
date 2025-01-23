
from pycocotools.coco import COCO
# val_coco = COCO('/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/Annotations/FLIR_coco/val.json')
# num_images = len(coco_annotation.getImgIds())
# print("Number of images:", num_images)
# # %%
# train_coco = COCO('/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/Annotations/FLIR_coco/train.json')
# num_imgs = len(train_coco.getImgIds())
# print("Number of images:", num_imgs)
# # %%
# import glob

# search_root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/FLIR-align'
# id = train_coco.getImgIds()[0]
# img = train_coco.loadImgs(id)[0]

# # %%
# rgb_name = img['file_name_RGB']
# thermal_name = img['file_name_IR']
# rgb_files = glob.glob(search_root + '/*/' + rgb_name)
# thermal_files = glob.glob(search_root + '/*/' + thermal_name)

# %%

# %%
import shutil
import glob
def copy_files(coco_path, isTrain):
    search_root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/FLIR-align'
    coco = COCO(coco_path)
    if isTrain:
        rgb_root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/FLIR-align/train_RGB'
        thermal_root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/FLIR-align/train_thermal'
    else:
        rgb_root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/FLIR-align/val_RGB'
        thermal_root = '/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/FLIR-align/val_thermal'
    
    for id in coco.getImgIds():
        img = coco.loadImgs(id)[0]
        rgb_name = img['file_name_RGB']
        thermal_name = img['file_name_IR']
        rgb_file = glob.glob(search_root + '/JPEGImages/' + rgb_name)[0]
        thermal_file = glob.glob(search_root + '/JPEGImages/' + thermal_name)[0]
        rgb_filename = rgb_file.split('/')[-1]
        thermal_filename = thermal_file.split('/')[-1]

        shutil.copy2(rgb_file, rgb_root + '/' + rgb_filename)
        shutil.copy2(thermal_file, thermal_root + '/' + thermal_filename)
        print(f"Copy {rgb_file} to {rgb_root + '/' + rgb_filename}  |   {thermal_file} to {thermal_root + '/' + thermal_filename}")



# copy_files('/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/Annotations/FLIR_coco/val.json', False)
copy_files('/media/ailab/HDD1/Workspace/dset/Drone-Detection-Benchmark/Source/Annotations/FLIR_coco/train.json', True)