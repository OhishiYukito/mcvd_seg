import torch
import datasets
import sys

class Dataloader:
    def __init__(self, dataset_name, mode='train', batch_size=64):
        if dataset_name=="bair64" and mode=='train':
            self.dataset = datasets.BAIRDataset('datasets/BAIR_h5/train')
        elif dataset_name == "cityscapes64":
            self.dataset = datasets.CityscapesDataset('datasets/Cityscapes64_h5/train')
        elif dataset_name == "cityscapes128":
            self.dataset = datasets.CityscapesDataset('datasets/Cityscapes128_h5/train')
        elif dataset_name == "kth64":
            self.dataset = datasets.KTHDataset('datasets/KTH64_h5')
        else:
            print("dataset type is not correct! (input: {})".format(dataset_name))
            sys.exit()
            
        self.batch_size = batch_size
        self.max_len = len(self.dataset)
        self.num_outputed = 0
        
        
    def __itr__(self):
        return self
    
    def __next__(self):
        if self.num_outputed >= self.max_len:
            return StopIteration
        
        # make mini-batch
        batch = []
        for i, data in enumerate(self.dataset):
            if i >= self.batch_size:
                break
            batch.append(data.unsqueeze(0))
        
        batch = torch.cat(batch, dim=0)
        self.num_outputed += self.batch_size
        
        return batch