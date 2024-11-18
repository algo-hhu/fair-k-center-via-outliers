#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <cstdint>

#include "cpp/fair_kcenter.hpp"

namespace fc=fairclustering;

// Splits points into weight, color and coordinates.
// Processes the data into a n-color-dim array points,
// where each index is one color.
// Number and weightsum give the number
// and total weight of points in either array.


void transformCenters(double* centers, std::vector<fc::Point*>& vec_centers){
    assert(vec_centers.size()!=0);
    std::size_t dim=(*vec_centers[0]).dim();
    for(std::size_t i=0; i<vec_centers.size(); i++){
        for(std::size_t j=0; j<dim; j++){
            centers[i*dim+j]=(*vec_centers[i])[j];
        }
    }
}

void transformLabels(int* labels, fc::ClusteringProblem& problem){
    for(auto p: problem.points()){
        labels[p.idx()]=p.color();
    }
}

// Thank you https://github.com/dstein64/kmeans1d!

extern "C"
{
// "__declspec(dllexport)" causes the function to be exported when compiling on Windows.
// Otherwise, the function is not exported and the code raises
//   "AttributeError: function 'cluster' not found".
// Exporting is a Windows platform requirement, not just a Visual Studio requirement
// (https://stackoverflow.com/a/22288874/1509433). The _WIN32 macro covers the Visual
// Studio compiler (MSVC) and MinGW. The __CYGWIN__ macro covers gcc and clang under
// Cygwin.
#if defined(_WIN32) || defined(__CYGWIN__)
    __declspec(dllexport)
#endif
    double fairKCenter(
            double *array,
            std::size_t *colors,
            std::size_t n,
            std::size_t d,
            std::size_t k,
            std::size_t nc,
            int *labels,
            double *centers
        )
    {

        std::vector<int> number(nc, 0);
        // One vector of points per color

        fc::ClusteringProblem problem;
        problem.read_input(array, colors, nc,  n, k, d);

        const fc::Index red = (problem.n_points(0) <= problem.n_points(1) ? 0 : 1);
        const fc::Index blue = 1 - red;

        fc::build_fairlets(problem, red, blue);

        auto gonz_centers = fc::gonzales(problem.points_of_color(red).begin(), problem.points_of_color(red).end(), problem.k());
        double radius = fc::assign_points(problem, gonz_centers);

        transformCenters(centers, gonz_centers);
        transformLabels(labels, problem);
        return radius;
    }
} // extern "C"

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _coremodule = {
    PyModuleDef_HEAD_INIT,
    "fair_kcenter._core",
    NULL,
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__core(void)
{
    return PyModule_Create(&_coremodule);
}
