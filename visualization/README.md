# Visualization
The figures in the paper are rendered in Blender using Animation Nodes. Since Animation Nodes is not actively developed, we would advise using [Blender Toolbox](https://github.com/HTDerekLiu/BlenderToolbox) instead. There are examples for [rendering point clouds](https://github.com/HTDerekLiu/BlenderToolbox/blob/master/demos/demo_pointCloudScalars.py) as well as [vector fields](https://github.com/HTDerekLiu/BlenderToolbox/blob/master/demos/demo_vectorField.py) in the demos folder.

## Our process
We used the scripts in `ply_utils.py` to export the features in the network per point into a .ply file, specifically the `save_feature` function (note, you need to install plyfile in your Python environment: `pip install plyfile`).

This ply file contains the point positions, a scalar per vertex with the feature activations, a vector per vertex with the vector features, and the local tangent basis at each point. The Animation Nodes script in Blender reads the ply file and adds vectors to the points.

You can find the [Animation Nodes script in a Blender file here](https://github.com/rubenwiersma/deltaconv/visualization/pointcloud_features.blend) -- it works with [Animation Nodes](https://animation-nodes.com/) and should work with Blender <= 2.9.7. Instructions for how to use this file are in the [blender pointcloud repository](https://github.com/rubenwiersma/blender-pointcloud).

The vector features are encoded as two coefficients for a tangent basis. If you want a 3D vector to visualize, you can simply compute:
```
vector_3d = v_x  x_basis + v_y  * y_basis
```

The tangent basis is computed and [assigned here]( https://github.com/rubenwiersma/deltaconv/blob/9768cb8f0d29c14065c750682be58f5386f77701/deltaconv/models/deltanet_base.py#L61).

## Alternatives
A simpler route to visualize features is with [Polyscope](https://polyscope.run/py/) (`pip install polyscope`). You can [add scalar quantities](https://polyscope.run/py/structures/point_cloud/scalar_quantities/) to visualize scalar features and [vector quantities](https://polyscope.run/py/structures/point_cloud/vector_quantities/) to visualize the vector features.

For pretty renders, we advise using the aforementioned [Blender Toolbox](https://github.com/HTDerekLiu/BlenderToolbox).