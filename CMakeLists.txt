cmake_minimum_required(VERSION 3.1.0)
project(deltaconv)


# Recurse in to pybind
add_subdirectory(deps/pybind11)

# set location of eigen for geometry-central
set(GC_EIGEN_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen" CACHE PATH "my path")

# geometry-central
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(deps/geometry-central)

pybind11_add_module(deltaconv_bindings 
  deltaconv/cpp/sampling.cpp
  deltaconv/cpp/core.cpp
)

include_directories(deltaconv_bindings ${CMAKE_CURRENT_SOURCE_DIR}/deltaconv/cpp)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(deltaconv_bindings PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(deltaconv_bindings PRIVATE geometry-central)
