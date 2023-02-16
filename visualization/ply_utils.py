import numpy as np
from plyfile import PlyData, PlyElement

def save_ply(data, filename, i=None):
    """
    Saves point cloud Data objects with the following fields:
    pos - positions
    norm - normals
    vec - vectors at points, in 3D coordinates
    color - color of each point
    y - label of each point
    scalar - a scalar function at each point
    """
    pos = data.pos
    norm = None
    if hasattr(data, 'norm'):
        norm = data.norm
    elif hasattr(data, 'normal'):
        norm = data.normal
    if i is not None:
        pos = pos[data.batch == i]
        if norm is not None:
            norm = norm[data.batch == i]
    pos = pos.cpu().numpy() 
    pos.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if norm is not None:
        norm = norm.cpu().numpy()
        pos = np.lib.recfunctions.rec_append_fields(pos, ('nx', 'ny', 'nz'), (norm[:, 0], norm[:, 1], norm[:, 2]), ('f4', 'f4', 'f4'))
    if hasattr(data, 'vec') and data.vec is not None:
        vec = data.vec.cpu().numpy()
        pos = np.lib.recfunctions.rec_append_fields(pos, ('vx', 'vy', 'vz'), (vec[:, 0], vec[:, 1], vec[:, 2]), ('f4', 'f4', 'f4'))
    if hasattr(data, 'color') and data.color is not None:
        color = data.color.cpu().numpy()
        pos = np.lib.recfunctions.rec_append_fields(pos, ('r', 'g', 'b'), (color[:, 0], color[:, 1], color[:, 2]), ('f4', 'f4', 'f4'))
    if hasattr(data, 'y') and data.y is not None and len(data.y.size()) > 0:
        y = data.y.cpu().numpy()
        if len(y.shape) > 1:
            y = y.flatten()
        pos = np.lib.recfunctions.rec_append_fields(pos, 'label', y, 'i4')
    if hasattr(data, 'scalar') and data.scalar is not None:
        scalar = data.scalar.cpu().numpy()
        if len(scalar.shape) > 1:
            scalar = scalar.flatten()
        pos = np.lib.recfunctions.rec_append_fields(pos, 'scalar', scalar, 'f4')

    pos = PlyElement.describe(pos.squeeze(), 'vertex')
    PlyData([pos]).write(filename)


def save_feature(filename, pos, normal, x_basis, y_basis, xs, vs=None, batch=None, i=None, y=None):
    """
    Saves point cloud Data objects with the following fields:
    pos - positions
    norm - normals
    vec - vectors at points, in 3D coordinates
    color - color of each point
    y - label of each point
    scalar - a scalar function at each point
    """
    if i is not None:
        pos = pos[batch == i]
        normal = normal[batch == i]
        x_basis = x_basis[batch == i]
        y_basis = y_basis[batch == i]
        xs = xs[batch == i]
        if vs is not None:
            vs = vs.view(-1, 2, vs.size(1))[batch == i]
        if y is not None:
            y = y[batch == i]

    pos = pos.cpu().numpy() 
    pos.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    normal = normal.cpu().numpy()
    pos = np.lib.recfunctions.rec_append_fields(pos, ('nx', 'ny', 'nz'), (normal[:, 0], normal[:, 1], normal[:, 2]), ('f4', 'f4', 'f4'))
    if y is not None:
        y = y.cpu().numpy()
        if len(y.shape) > 1:
            y = y.flatten()
        pos = np.lib.recfunctions.rec_append_fields(pos, 'label', y, 'i4')
    
    for j in range(xs.size(1)):
        ply_out = pos
        
        x = xs[:, j].cpu().numpy()
        if len(x) > 1:
            x = x.flatten()
        ply_out = np.lib.recfunctions.rec_append_fields(ply_out, 'scalar', x, 'f4')

        if vs is not None:
            v = vs[:, :, j]
            v = v[:, 0:1] * x_basis + v[:, 1:] * y_basis
            v = v.cpu().numpy()
            ply_out = np.lib.recfunctions.rec_append_fields(ply_out, ('vx', 'vy', 'vz'), (v[:, 0], v[:, 1], v[:, 2]), ('f4', 'f4', 'f4'))

        ply_out = PlyElement.describe(ply_out.squeeze(), 'vertex')
        PlyData([ply_out]).write('{}_shape{}_feat{}.ply'.format(filename, i, j))