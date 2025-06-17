#include "clusteringproblem.h"
#include "gonzales.h"
#include "fairlets.h"

#include "lemon/smart_graph.h"
#include "lemon/preflow.h"

#include <iostream>
#include <vector>

#include <cmath>

namespace fc = fairclustering;

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

int main() {
    fc::ClusteringProblem problem;
    problem.read_input(std::cin);


    /*
    std::cout << "Eingabe:" << std::endl;
    for(auto& p : problem.points()) {
        std::cout << p << std::endl;
    }*/

    const fc::Index red = (problem.n_points(0) <= problem.n_points(1) ? 0 : 1);
    //const fc::Index blue = 1 - red;

    double max_fairlet_radius = fc::build_fairlets(problem);

    auto centers = fc::gonzales(problem.points_of_color(red).begin(), problem.points_of_color(red).end(), problem.k());
    double r = problem.assign_points(centers);

    std::cout << "Zentren:" << std::endl;
    for(auto& c : centers) {
        std::cout << *c << std::endl;
    }


    std::cout << "Clustering mit Radius " << r << " [Fairlet-Radius " << max_fairlet_radius << "]:" << std::endl;
    /*
    for(auto& p : problem.points()) {
        std::cout << p << std::endl;
    }
    */

    return 0;
}
