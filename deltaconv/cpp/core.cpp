#include "sampling.h"

#include "geometrycentral/numerical/linear_algebra_utilities.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Eigen/Dense"

namespace py = pybind11;

using namespace geometrycentral;

Eigen::MatrixXi geodesicFPS(const DenseMatrix<double>& vMat, size_t nSamples) {

  std::vector<Vector3> points(vMat.rows());

  for (size_t iP = 0; iP < points.size(); iP++) {
    points[iP] = Vector3{vMat(iP, 0), vMat(iP, 1), vMat(iP, 2)};
  }

  return constructGeodesicFPS(points, nSamples);
}


// Actual binding code
// clang-format off
PYBIND11_MODULE(deltaconv_bindings, m) {
  m.doc() = "Fast implementation of geodesic farthest-point sampling.";

  m.def("geodesicFPS", &geodesicFPS, "Sample points using farthest point sampling with geodesic distances.",
      py::arg("vMat"), py::arg("nSamples"));
}

// clang-format on
