import os
import os.path as osp
from os import listdir as osls
import shutil
import numpy as np
import progressbar

import torch
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
import openmesh

class SHREC(InMemoryDataset):
    r"""The shrec classification dataset.

    This is the remeshed version from MeshCNN.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

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

    url = 'https://dl.dropboxusercontent.com/s/biiwlkkky7bp5ya/shrec_16.zip'
    class_names = [
        'alien',
        'ants',
        'armadillo',
        'bird1',
        'bird2',
        'camel',
        'cat',
        'centaur',
        'dinosaur',
        'dino_ske',
        'dog1',
        'dog2',
        'flamingo',
        'glasses',
        'gorilla',
        'hand',
        'horse',
        'lamp',
        'laptop',
        'man',
        'myScissor',
        'octopus',
        'pliers',
        'rabbit',
        'santa',
        'shark',
        'snake',
        'spiders',
        'two_balls',
        'woman'
    ]

    def __init__(self, root, train=True, transform=None, pre_transform=None,
                 pre_filter=None, split10=True):
        self.split10 = split10
        super(SHREC, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['shrec_16.zip']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def num_classes(self):
        return len(self.class_names)

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        print('Extracting zip...')
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        training_list = []
        test_list = []
        print('Processing Shrec...')
        raw_path = osp.join(self.raw_dir, 'shrec_16')

        for class_idx, class_name in enumerate(self.class_names):
            train_meshes = osp.join(raw_path, class_name, 'train')
            mesh_files = osls(train_meshes)
            idx = np.random.permutation(len(mesh_files))[:10] if self.split10 else np.arange(len(mesh_files))
            for file_i, filename in progressbar.progressbar(enumerate(mesh_files)):
                if file_i not in idx:
                    continue
                data = read_obj(osp.join(train_meshes, filename))
                data.y = class_idx
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                training_list.append(data)
            
            test_meshes = osp.join(raw_path, class_name, 'test')

            for filename in progressbar.progressbar(osls(test_meshes)):
                data = read_obj(osp.join(test_meshes, filename))
                data.y = class_idx
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                test_list.append(data)

        torch.save(self.collate(training_list), self.processed_paths[0])
        torch.save(self.collate(test_list), self.processed_paths[1])

        shutil.rmtree(osp.join(self.raw_dir, 'shrec_16'))

def read_obj(path):
    mesh = openmesh.read_trimesh(path)
    pos = torch.from_numpy(mesh.points()).to(torch.float)
    face = torch.from_numpy(mesh.face_vertex_indices())
    face = face.t().to(torch.long).contiguous()
    return Data(pos=pos, face=face)