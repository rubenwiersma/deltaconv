import os.path as osp
from os import listdir as osls
import shutil
import numpy as np
import progressbar

import openmesh
import torch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.io import read_ply

class ShapeSeg(InMemoryDataset):
    r"""The Shape Segmentation dataset proposed by Maron et al. in
    "Convolutional neural networks on surfaces via seamless toric covers"
    <https://dl.acm.org/citation.cfm?id=3073616>`_ ,
    containing meshes from Adobe, SCAPE, FAUST, and MIT for training
    and SHREC shapes for testing.

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

    url = 'https://www.dropbox.com/s/s3n05sw0zg27fz3/human_seg.tar.gz'
    mit_folders = {
        'crane': 18,
        'squat1': 25,
        'jumping': 15,
        'squat2': 25,
        'bouncing': 18,
        'march1': 25,
        'handstand': 18,
        'march2': 25
        }

    def __init__(self, root, train=True, transform=None, pre_transform=None,
                 pre_filter=None):
        super(ShapeSeg, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['shapeseg.zip']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download {} from {} and move it to {}'.
            format(self.raw_file_names, self.url, self.raw_dir))

    def process(self):
        print('Extracting zip...')
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        shapeseg_path = osp.join(self.raw_dir, 'ShapeSeg')

        data_list = []
        #Adobe
        print('Processing Adobe')
        adobe_path = osp.join(shapeseg_path, 'Adobe', 'raw')
        extract_zip(osp.join(adobe_path, 'adobe.zip'), adobe_path)
        adobe_meshes = osp.join(adobe_path, 'meshes')
        adobe_meshes = osp.join(adobe_meshes, '{}.ply')
        adobe_segs = osp.join(adobe_path, 'segs', '{}.pt')
        for i in progressbar.progressbar(range(41)):
            data = read_ply(adobe_meshes.format(i))
            data.y = torch.load(adobe_segs.format(i))
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), osp.join(shapeseg_path, 'adobe.pt'))
            
        #FAUST
        print('Processing FAUST')
        faust_path = osp.join(shapeseg_path, 'FAUST', 'raw')
        extract_zip(osp.join(faust_path, 'faust.zip'), faust_path)
        faust_meshes = osp.join(faust_path, 'meshes')
        faust_meshes = osp.join(faust_meshes, 'tr_reg_{0:03d}.ply')
        faust_segs = torch.load(osp.join(faust_path, 'segs', 'faust_seg.pt'))
        for i in progressbar.progressbar(range(100)):
            data = read_ply(faust_meshes.format(i))
            data.y = faust_segs
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), osp.join(shapeseg_path, 'faust.pt'))

        #MIT
        print('Processing MIT')
        mit_path = osp.join(shapeseg_path, 'MIT', 'raw')
        extract_zip(osp.join(mit_path, 'mit.zip'), mit_path)
        mit_meshes = osp.join(mit_path, 'meshes')
        mit_seg = osp.join(mit_path, 'segs')
        for filename in progressbar.progressbar(osls(mit_meshes)):
            data = read_obj(osp.join(mit_meshes, filename))
            seg_path = osp.join(mit_seg, filename.replace('.obj', '.eseg'))
            segs = torch.from_numpy(np.loadtxt(seg_path)).long()
            data.y = edge_to_vertex_labels(data.face, segs, data.num_nodes)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), osp.join(shapeseg_path, 'mit.pt'))

        #SCAPE
        print('Processing SCAPE')
        scape_path = osp.join(shapeseg_path, 'SCAPE', 'raw')
        extract_zip(osp.join(scape_path, 'scape.zip'), scape_path)
        scape_meshes = osp.join(scape_path, 'meshes')
        scape_meshes = osp.join(scape_meshes, '{}.ply')
        scape_segs = torch.load(osp.join(scape_path, 'segs', 'scape_seg.pt'))
        for i in progressbar.progressbar(range(71)):
            data = read_ply(scape_meshes.format(i))
            data.y = scape_segs
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), osp.join(shapeseg_path, 'scape.pt'))

        torch.save(self.collate(data_list), self.processed_paths[0])
        data_list = []

        #SHREC
        print('Processing SHREC')
        shrec_path = osp.join(shapeseg_path, 'SHREC', 'raw')
        extract_zip(osp.join(shrec_path, 'shrec.zip'), shrec_path)
        shrec_meshes = osp.join(shrec_path, 'meshes')
        shrec_meshes = osp.join(shrec_meshes, '{}.ply')
        shrec_segs = osp.join(shrec_path, 'segs', '{}.pt')
        for i in progressbar.progressbar(range(18)):
            data = read_ply(shrec_meshes.format(i))
            data.y = torch.load(shrec_segs.format(i))
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[1])


        shutil.rmtree(osp.join(self.raw_dir, 'ShapeSeg'))

def edge_to_vertex_labels(faces, labels, n_nodes):
    edge2key = set()
    edge_index = torch.LongTensor(0, 2)
    for face in faces.transpose(0, 1):
        edges = torch.stack([face[:2], face[1:], face[::2]], dim=0)
        for idx, edge in enumerate(edges):
            edge = edge.sort().values
            edges[idx] = edge
            if tuple(edge.tolist()) not in edge2key:
                edge2key.add(tuple(edge.tolist()))
                edge_index = torch.cat((edge_index, edge.view(1, -1)))

    res = torch.LongTensor(n_nodes).fill_(0)
    res[edge_index[:, 0]] = labels
    res[edge_index[:, 1]] = labels

    return res - 1

def read_obj(path):
    mesh = openmesh.read_trimesh(path)
    pos = torch.from_numpy(mesh.points()).to(torch.float)
    face = torch.from_numpy(mesh.face_vertex_indices())
    face = face.t().to(torch.long).contiguous()
    return Data(pos=pos, face=face)