#include "fair_kcenter.hpp"


namespace fairclustering
{

void print_clustering(Clustering& c){
    //TODO

}
void print_points(PointSet* set){
    //TODO
}

/*
std::vector<double> fair_k_center(std::vector<std::vector<double>>& X, std::vector<size_t>& colors, size_t n_colors, size_t k){
    if(X.size()==0){
        throw std::invalid_argument( "received empty pointset" );
    }
    if(X.size()!=colors.size()){
        throw std::invalid_argument( "size of pointset" + std::to_string(X.size()) + "not equal size of colors" + std::to_string(colors.size()) );
    }
    if(X.size()<k){
        throw std::invalid_argument("number of clusters" + std::to_string(k)+ "larger number of points" +  std::to_string(X.size()));
    }
    ClusteringProblem problem;
    problem.read_input(X,colors, n_colors, k);

    //print_points()

    const Index red = (problem.n_points(0) <= problem.n_points(1) ? 0 : 1);
    const Index blue = 1 - red;

    build_fairlets(problem, red, blue);

    Clustering c;

    std::vector<Point*> centers = gonzales(problem.points_of_color(red).begin(), problem.points_of_color(red).end(), problem.k());
    double r = assign_points(problem, centers);

    //print_clustering()

}
*/
}
