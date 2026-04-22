#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <cstdint>
#include <utility>
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
        std::size_t cluster_id=p.cluster();
        if(p.is_outlier()){
            cluster_id=problem.k();
        }
        labels[p.pos()]=cluster_id;
    }
}

void transformFairlets(int* fairlets, fc::ClusteringProblem& problem){
    for(auto p: problem.points()){
        std::size_t buddy_id=p.pos();
        if(p.buddy()!=NULL){
            buddy_id=p.buddy()->pos();
        }
        fairlets[p.pos()]=buddy_id;
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
    void fairKCenter(
            double *array,
            std::size_t *colors,
            std::size_t *color_ratio,
            std::size_t n,
            std::size_t d,
            std::size_t k,
            std::size_t nc,
            int *labels,
            double *centers,
            int *fairlets,
            double* cost,
            bool without_anchors
        )
    {
        std::vector<int> number(nc, 0);
        // One vector of points per color
        fc::ClusteringProblem problem;
        problem.read_input(array, colors, nc, n, k, d, color_ratio, without_anchors);

        cost[0]=fc::build_fairlets(problem);
        if(fairlets!=NULL){
            transformFairlets(fairlets, problem);
        }

        std::vector<fc::Point*> gonz_centers;
        if(without_anchors){
            gonz_centers = fc::gonzales(problem.points_of_color(problem.main_col()).begin(), problem.points_of_color(problem.main_col()).end(), problem.k());
            cost[1] = problem.assign_points(gonz_centers);
        }
        else{
            gonz_centers = fc::gonzales(problem.points().begin(), problem.points().end(), problem.k());
            cost[1] = problem.assign_points(gonz_centers);
        }

        transformLabels(labels, problem);
        transformCenters(centers, gonz_centers);
        return;
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
