#ifndef SAMPLING_H
#define SAMPLING_H

#include "geometrycentral/utilities/vector3.h"
#include "geometrycentral/utilities/knn.h"

#include "Eigen/Dense"

using geometrycentral::Vector3;
using Neighbors_t = std::vector<std::vector<size_t>>;

// Data structure for Priority Queue
struct VertexPair {
	int		vId;
	double	distance;
	bool	operator> (const VertexPair &ref) const { return distance > ref.distance; }
	bool	operator< (const VertexPair &ref) const { return distance < ref.distance; }
};

// Creates a kNN graph on a set of points using functionality from Geometry Central.
Neighbors_t generateKNN(const std::vector<Vector3>& points, size_t k, bool selfLoops = true);

// Farthest point sampling on a point cloud, based on geodesic distances on a kNN graph.
// The distances are computed using Dijkstra on the kNN graph.
Eigen::VectorXi constructGeodesicFPS(const std::vector<Vector3>& points, const size_t numSamples);

// Computes the shortest distance between two points using Dijkstra's algorithm.
void computeDijkstra(const std::vector<Vector3>& points, int source, const Neighbors_t& neigh, Eigen::VectorXd &D);

#endif // !SAMPLING_H
