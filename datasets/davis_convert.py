import os
from davis import DavisHDF5Maker
import cv2
from torchvision import transforms
import numpy as np

def center_crop(image):
    h, w, c = image.shape
    new_h, new_w = h if h < w else w, w if w < h else h
    r_min, r_max = h//2 - new_h//2, h//2 + new_h//2
    c_min, c_max = w//2 - new_w//2, w//2 + new_w//2
    return image[r_min:r_max, c_min:c_max, :]

def make_h5_from_davis(davis_dir, split="train", out_dir="./h5_ds", vids_per_shard=100000, force_h5=False, image_size=64):
    
    h5_maker = DavisHDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    # get filename list    
    if split=="train":
        path = os.path.join(davis_dir, 'ImageSets/480p/train.txt')
    elif split in ["test", "valid"]:
        path = os.path.join(davis_dir, "ImageSets/480p/val.txt")
        
    with open(path, "r") as f:
        filename_list = f.readlines()

    # load frames and save as hdf5
    pre_video_title = None
    frames = []
    frames_ann = []
    for filename_pair in filename_list:
        image, annotation = filename_pair.split()
        video_title = os.path.dirname(image).split('/')[-1]
        image = cv2.cvtColor(cv2.imread(davis_dir + image), cv2.COLOR_BGR2RGB)
        annotation = cv2.cvtColor(cv2.imread(davis_dir + annotation), cv2.COLOR_BGR2RGB)
        
        # resize (by resize, value range will be change: {0 or 1} -> [0~1])
        image, annotation = transforms.Resize(image_size)(transforms.ToTensor()(image)), transforms.Resize(image_size)(transforms.ToTensor()(annotation))
        image, annotation = transforms.CenterCrop(image_size)(image), transforms.CenterCrop(image_size)(annotation)
        image, annotation = np.moveaxis(image.numpy()*255, 0, -1), np.moveaxis(annotation.numpy()*255, 0, -1)
        
        if video_title==pre_video_title or pre_video_title is None:
            frames.append(image)
            frames_ann.append(annotation)
            if pre_video_title is None:
                pre_video_title = video_title
        else:
            h5_maker.add_data((frames, frames_ann), dtype='uint8')
            pre_video_title = video_title
            frames = [image]
            frames_ann = [annotation]
            

import matplotlib.pyplot as plt
if __name__=='__main__':
    davis_dir = 'datasets/DAVIS'
    make_h5_from_davis(davis_dir, split='train', out_dir='datasets/DAVIS64_h5/train', force_h5=True)
    make_h5_from_davis(davis_dir, split='test', out_dir='datasets/DAVIS64_h5/test', force_h5=True)