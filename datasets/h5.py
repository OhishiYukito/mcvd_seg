# https://github.com/voletiv/mcvd-pytorch/blob/master/datasets/h5.py

import h5py
import numpy as np
import os
import glob
import sys
from torch.utils.data import Dataset

default_opener = lambda p: h5py.File(p, 'r')

class HDF5Dataset(Dataset):

    @staticmethod
    def _get_num_in_shard(shard_p, opener=default_opener):
        print(f'\rh5: Opening {shard_p}... ', end='')
        try:
            with opener(shard_p) as f:
                num_per_shard = len(f['len'].keys())
        except:
            print(f"h5: Could not open {shard_p}!")
            num_per_shard = -1
        return num_per_shard

    @staticmethod
    def check_shard_lengths(file_paths, opener=default_opener):
        """
        Filter away the last shard, which is assumed to be smaller. this double checks that all other shards have the
        same number of entries.
        :param file_paths: list of .hdf5 files
        :param opener:
        :return: tuple (ps, num_per_shard) where
            ps = filtered file paths,
            num_per_shard = number of entries in all of the shards in `ps`
        """
        shard_lengths = []
        print("Checking shard_lengths in", file_paths)
        for i, p in enumerate(file_paths):
            shard_lengths.append(HDF5Dataset._get_num_in_shard(p, opener))
        return shard_lengths

    def __init__(self, data_path,   # hdf5 file, or directory of hdf5s
                 shuffle_shards=False,
                 opener=default_opener,
                 seed=29):
        self.data_path = data_path
        self.shuffle_shards = shuffle_shards
        self.opener = opener
        self.seed = seed

        # If `data_path` is an hdf5 file
        if os.path.splitext(self.data_path)[-1] == '.hdf5' or os.path.splitext(self.data_path)[-1] == '.h5':
            self.data_dir = os.path.dirname(self.data_path)
            self.shard_paths = [self.data_path]
        # Else, if `data_path` is a directory of hdf5s
        else:
            self.data_dir = self.data_path
            self.shard_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.hdf5')) + glob.glob(os.path.join(self.data_dir, '*.h5')))

        assert len(self.shard_paths) > 0, "h5: Directory does not have any .hdf5 files! Dir: " + self.data_dir

        self.shard_lengths = HDF5Dataset.check_shard_lengths(self.shard_paths, self.opener)
        self.num_per_shard = self.shard_lengths[0]
        self.total_num = sum(self.shard_lengths)

        assert len(self.shard_paths) > 0, "h5: Could not find .hdf5 files! Dir: " + self.data_dir + " ; len(self.shard_paths) = " + str(len(self.shard_paths))

        self.num_of_shards = len(self.shard_paths)

        print("h5: paths", len(self.shard_paths), "; shard_lengths", self.shard_lengths, "; total", self.total_num)

        # Shuffle shards
        if self.shuffle_shards:
            np.random.seed(seed)
            np.random.shuffle(self.shard_paths)

    def __len__(self):
        return self.total_num

    def get_indices(self, idx):
        shard_idx = np.digitize(idx, np.cumsum(self.shard_lengths))
        idx_in_shard = str(idx - sum(self.shard_lengths[:shard_idx]))
        return shard_idx, idx_in_shard

    def __getitem__(self, index):
        idx = index % self.total_num
        shard_idx, idx_in_shard = self.get_indices(idx)
        # Read from shard
        with self.opener(self.shard_paths[shard_idx]) as f:
            data = f[idx_in_shard][()]
        return data
    
    
#if __name__=='__main__':
    
    # command is "python h5.py file_path"
#    path = sys.argv[1]
    
    # Read from HDF5 file
#    h5_ds = HDF5Dataset(path)
#    data = h5_ds[0]
#    print(data.shape)