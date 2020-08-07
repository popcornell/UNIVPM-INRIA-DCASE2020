import numpy as np
from torch.utils.data import Dataset

class MultiDataset(Dataset):

    def __init__(self, dataset_list, sample_probs, length=None):
        self.dataset_list = dataset_list
        self.sample_probs = sample_probs
        self.change_prob(sample_probs)
        self.length = length

    def __len__(self):

        if self.length:
            return self.length
        else:
            return min([len(x) for x in self.dataset_list])

    def __getitem__(self, item):

        dataset_indx = np.random.choice([x for x in range(len(self.dataset_list))], 1, replace=True, p=self.sample_probs)[0]
        return self.dataset_list[dataset_indx][item]

    def change_prob(self, new_prob):
        if sum(new_prob) != 1:
            print('Warning: sample proba did not sum up to one, normalizing')
            new_prob = [s/sum(new_prob) for s in new_prob]
        self.sample_probs = new_prob
