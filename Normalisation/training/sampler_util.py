import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, RandomSampler

"""
class TmpDataset(Dataset):
    def __init__(self, m=10):
        self.len = m

    def __getitem__(self, index):
        return (list(range(10)) * index, [0] * index)

    def __len__(self):
        return self.len
"""

class FixedLengthBatchSampler(Sampler):
    def __init__(self, sampler, fixed_length, drop_last):
        self.sampler = sampler
        self.fixed_length = fixed_length
        self.drop_last = drop_last
        self.rel_sampler_count = 0

    def __iter__(self):
        batch = []
        now_length = 0
        for idx in self.sampler:
            #print(batch, now_length)
            sample_length = len(self.sampler.data_source[idx][-1]) * 3
            if now_length + sample_length > self.fixed_length:
                #print(batch, now_length)
                yield batch
                batch = []
                now_length = 0
            batch.append(idx)
            now_length += sample_length
            self.rel_sampler_count += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

def my_collate_fn(batch):
    type_count = len(batch[0])
    batch_size = sum([len(item[-1]) for item in batch])
    output = ()
    for i in range(type_count):
        tmp = []
        for item in batch:
            tmp.extend(item[i])
        if len(tmp) <= batch_size:
            output += (torch.LongTensor(tmp),)
        else:
            output += (torch.LongTensor(tmp).reshape(batch_size, -1),)
    return output
