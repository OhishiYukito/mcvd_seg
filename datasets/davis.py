from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

from .h5 import HDF5Maker, HDF5Dataset


class DavisHDF5Maker(HDF5Maker):
        
    def create_video_groups(self):
        self.writer.create_group('len')
        self.writer.create_group('videos')
        self.writer.create_group('annotations')
        
    def add_video_data(self, data, dtype=None):
        frames, frames_ann = data
        self.writer['len'].create_dataset(str(self.count), data=len(frames))
        self.writer['videos'].create_group(str(self.count))
        for i, frame in enumerate(frames):
            self.writer['videos'][str(self.count)].create_dataset(str(i), data=frame, dtype=dtype, compression="lzf")
        self.writer['annotations'].create_group(str(self.count))
        for i, frame in enumerate(frames_ann):
            self.writer['annotations'][str(self.count)].create_dataset(str(i), data=frame, dtype=dtype, compression="lzf")
    
            
class DavisHDF5Dataset(Dataset):
    
    def __init__(self, data_path, batch_size, frames_per_sample=5, image_size=64, random_time=True, random_horizontal_flip=True, grayscale=False, color_jitter=0,
                 total_videos=-1):

        self.data_path = data_path
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample
        self.image_size = image_size
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.grayscale = grayscale
        self.color_jitter = color_jitter
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)

            
        self.jitter = transforms.ColorJitter(hue=color_jitter)

        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

        print(f"Dataset length: {self.__len__()}")

    def __len__(self):
        return self.total_videos if self.total_videos > 0 else max(len(self.videos_ds), self.batch_size)
    
    def max_index(self):
        return len(self.videos_ds)
    
    def __getitem__(self, index, time_idx=0):
        video_index = round(index / (self.__len__() - 1) * (self.max_index() - 1))
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

        prefinals = []
        prefinals_ann = []
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
            if self.random_time and video_len > self.frames_per_sample:
                time_idx = np.random.choice(video_len - self.frames_per_sample)
            for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
                img = f['videos'][str(idx_in_shard)][str(i)][()]
                img_ann = f['annotations'][str(idx_in_shard)][str(i)][()]
                
                img, img_ann = transforms.ToTensor()(img), transforms.ToTensor()(img_ann)
                # Images has been resized in 'davis_convert.py'
                ## resize (by resize, value range will be change: {0 or 1} -> [0~1])
                #img, img_ann = transforms.Resize(self.image_size)(transforms.ToTensor()(img)), transforms.Resize(self.image_size)(transforms.ToTensor()(img_ann))
                ##plt.imshow(img.permute(1,2,0))
                ##plt.imshow(img_ann.permute(1,2,0))
                #img, img_ann = transforms.CenterCrop(self.image_size)(img), transforms.CenterCrop(self.image_size)(img_ann)
                
                # random flip
                arr = transforms.RandomHorizontalFlip(flip_p)(img)
                arr_ann = transforms.RandomHorizontalFlip(flip_p)(img_ann)

                # grayscale (to set 'channels = 1')
                if self.grayscale:
                    arr = transforms.Grayscale(num_output_channels=1)(img)
                    arr_ann = transforms.Grayscale(num_output_channels=1)(img_ann)

                prefinals.append(arr)
                prefinals_ann.append(arr_ann)
                
        data = torch.stack(prefinals)
        data_ann = torch.stack(prefinals_ann)
        data = self.jitter(data)

        return data, data_ann

        
    
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '/home/ohishiyukito/Documents/GraduationResearch/DiffModel_with_Seg/datasets/DAVIS64_h5/train'
    dataset = DavisHDF5Dataset(path, 100)
    for sample, sample_ann in dataset:
        sample = sample[0].permute(1,2,0)
        sample_ann = sample_ann[0].permute(1,2,0)
        plt.imshow(sample)
        plt.figure()
        plt.imshow(sample_ann)
        plt.show()