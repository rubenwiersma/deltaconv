#include "sampling.h"
#include <queue>

// Creates a kNN graph on a set of points using functionality from Geometry Central.
Neighbors_t generateKNN(const std::vector<Vector3>& points, size_t k, bool selfLoops) {

  geometrycentral::NearestNeighborFinder finder(points);

  std::vector<std::vector<size_t>> result;
  for (size_t i = 0; i < points.size(); i++) {
    result.emplace_back(finder.kNearestNeighbors(i, k));
    if (selfLoops)
      result.back().insert(result.back().begin(), i); // add the center point to the front
  }

  return result;
}

// Farthest point sampling on a point cloud, based on geodesic distances on a kNN graph.
// The distances are computed using Dijkstra on the kNN graph.
Eigen::VectorXi constructGeodesicFPS(const std::vector<Vector3>& points, const size_t numSamples) {
	// Generate the kNN graph to perform distance computations on.
	Neighbors_t neigh = generateKNN(points, 10);
	// Store the results in sampleID
	Eigen::VectorXi sampleIdx(numSamples);

	// Store the results of the Dijkstra approach in a N-dimensional vector.
	// We can reuse this distance vector for every sampled point,
	// as we are looking for the farthest point from all the previously sampled points.
	Eigen::VectorXd D(points.size());
	D.setConstant(std::numeric_limits<double>::infinity());

	// Will be used to obtain a seed for the random number engine.
	std::random_device rd;
	// Standard mersenne_twister_engine seeded with rd().
	std::mt19937 gen(rd());
	// From 0 to (number of points - 1).
	std::uniform_int_distribution<> dist(0, points.size() - 1);	
	// Pick a random point to start with.
	sampleIdx(0) = dist(gen);

	for (size_t i = 1; i < numSamples; i++) {
		Eigen::VectorXi::Index maxIndex;
		// Update distances.
		computeDijkstra(points, sampleIdx(i-1), neigh, D);
		// Pick the point with the largest distance.
		D.maxCoeff(&maxIndex);
		// Add the point with the largest distance to the samples.
		sampleIdx(i) = maxIndex;
	}

	return sampleIdx;
}

// Computes the shortest distance between two points using Dijkstra's algorithm.
void computeDijkstra(const std::vector<Vector3>& points, int source, const Neighbors_t& neigh, Eigen::VectorXd &D) {
	std::priority_queue<VertexPair, std::vector<VertexPair>, std::greater<VertexPair>> DistanceQueue;

	D(source) = 0.0;
	VertexPair vp{ source, D(source) };
	DistanceQueue.push(vp);

	while (!DistanceQueue.empty()){
		VertexPair start = DistanceQueue.top();
		geometrycentral::Vector3 startPos = points[start.vId];
		DistanceQueue.pop();

		for (int vNeigh : neigh[start.vId]){
			double dist, distTemp;
			geometrycentral::Vector3 neighPos = points[vNeigh];
			dist = (neighPos - startPos).norm();
			distTemp = start.distance + dist;
			if(distTemp < D(vNeigh)){
				D(vNeigh) = distTemp;
				VertexPair neigh{ vNeigh, distTemp };
				DistanceQueue.push(neigh);
			}
		}
	}

}