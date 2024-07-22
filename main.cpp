#include "clusteringproblem.h"
#include "gonzales.h"
#include "fairlets.h"

#include <lemon/smart_graph.h>
#include <lemon/preflow.h>

#include <iostream>
#include <vector>

#include <cmath>

namespace fc = fairclustering;

int main() {
    fc::ClusteringProblem problem;
    problem.read_input(std::cin);

    std::cout << "Eingabe:" << std::endl;
    for(auto& p : problem.points()) {
        std::cout << p << std::endl;
    }

    const fc::Index red = (problem.n_points(0) <= problem.n_points(1) ? 0 : 1);
    const fc::Index blue = 1 - red;

    double max_fairlet_radius = fc::build_fairlets(problem, red, blue);

    auto centers = fc::gonzales(problem.points_of_color(red).begin(), problem.points_of_color(red).end(), problem.k());
    double r = fc::assign_points(problem, *centers);

    std::cout << "Zentren:" << std::endl;
    for(auto& c : *centers) {
        std::cout << *c << std::endl;
    }

    std::cout << "Clustering mit Radius " << r << " [Fairlet-Radius " << max_fairlet_radius << "]:" << std::endl;
    for(auto& p : problem.points()) {
        std::cout << p << std::endl;
    }

    delete centers;

    return 0;
}
