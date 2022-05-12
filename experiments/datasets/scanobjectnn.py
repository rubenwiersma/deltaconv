import os
import os.path as osp
import shutil
import glob

import torch
from torch_geometric.data import InMemoryDataset, Data

import h5py


class ScanObjectNN(InMemoryDataset):
    r"""The pre-processed ScanObjectNN dataset from the paper
    'Revisiting Point Cloud Classification: A New Benchmark Dataset and Classification Model on Real-World Data'
    https://arxiv.org/pdf/1908.04616.pdf

    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, background=False, augmentation=None, train=True, transform=None,
                 pre_transform=None, pre_filter=None):

        assert augmentation in self.augmentation_variants
        self.augmentation = augmentation
        self.background = background
        self.bg_path = 'main_split' if background else 'main_split_nobg'

        super(ScanObjectNN, self).__init__(root, transform, pre_transform,
                                       pre_filter)

        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def class_names(self):
        return [
            'bag', 'bed', 'bin', 'box', 'cabinets', 'chair', 'desk', 'display',
            'door', 'pillow', 'shelves', 'sink', 'sofa', 'table', 'toilet'
        ]
    
    @property
    def augmentation_variants(self):
        return [None, 'PB_T25', 'PB_T25_R', 'PB_T50_R', 'PB_T50_RS']

    @property
    def raw_file_dict(self): 
        return {
            None: ['training_objectdataset.h5', 'test_objectdataset.h5'],
            'PB_T25': ['training_objectdataset_augmented25_norot.h5', 'test_objectdataset_augmented25_norot.h5'],
            'PB_T25_R': ['training_objectdataset_augmented25rot.h5', 'test_objectdataset_augmented25rot.h5'],
            'PB_T50_R': ['training_objectdataset_augmentedrot.h5', 'test_objectdataset_augmentedrot.h5'],
            'PB_T50_RS': ['training_objectdataset_augmentedrot_scale75.h5', 'test_objectdataset_augmentedrot_scale75.h5'],
        }

    @property
    def raw_file_names(self):
        return [os.path.join(self.bg_path, filename) for filename in self.raw_file_dict[self.augmentation]]

    @property
    def processed_file_names(self):
        bg_string = 'bg' if self.background else 'nobg'
        augmentation_string = self.augmentation if self.augmentation is not None else 'vanilla'
        folder = bg_string + '_' + augmentation_string
        return [os.path.join(folder, 'training.pt'), os.path.join(folder, 'test.pt')]

    def download(self):
        if (not os.path.exists(os.path.join(self.raw_dir, self.raw_file_names[0]))):
            raise RuntimeError(
                'Dataset not found, please place the processed h5 files in {}.'.format(self.raw_dir)
            )
        return

    def process(self):
        for raw_path, path in zip(self.raw_paths, self.processed_paths):
            training = h5py.File(os.path.join(self.raw_dir, raw_path), 'r')
            data_list = []
            for i, pos in enumerate(training['data']):
                y = training['label'][i]
                data_list.append(Data(pos=torch.from_numpy(pos), y=torch.Tensor([y]).long()))

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            torch.save(self.collate(data_list), path)
            

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
