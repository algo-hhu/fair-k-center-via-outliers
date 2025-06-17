#ifndef FC_FAIRKCENTER_H_
#define FC_FAIRKCENTER_H_

#include "clusteringproblem.h"
#include "gonzales.h"
#include "fairlets.h"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace fairclustering{

typedef struct Clustering{
    std::vector<std::vector<double>> centers;
    std::vector<std::vector<double>> points;
    double radius;
    size_t dim;
} Clustering;

void print_clustering(Clustering& c);
void print_points(PointSet* set);

std::vector<double> fair_k_center(std::vector<std::vector<double>>& X, std::vector<size_t>& colors, size_t n_colors, size_t k);

}
#endif
