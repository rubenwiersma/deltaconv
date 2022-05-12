
# Farthest Point Sampling
The Farthest Point Sampling (FPS) routine is implemented in C++ so that it runs fast on a CPU. FPS typically only has to be computed once, so it can be run as a precomputation step. If you need fewer samples inside the network, you can simply truncate the list of selected points.

In our experiments, we noticed the C++ implementation is much faster than existing implementations for FPS in PyG. However, if you have issues building the C++ components, feel free to replace this variant with your favorite FPS implementation. The CMake requirements can be turned off in `setup.py`, lines 79 and 82.