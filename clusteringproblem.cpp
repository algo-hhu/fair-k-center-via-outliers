#include "clusteringproblem.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace fairclustering {

std::ostream& operator<<(std::ostream& str, const Point& p) {
    const auto cur_precision = str.precision();
    const auto cur_flags = str.flags();
    
    str << "[ ";
    str << std::fixed << std::setprecision(3);
    for(auto coord : p.coords()) {
        str << std::setprecision(2) << coord << " ";
    }

    str.flags(cur_flags);
    str.precision(cur_precision);

    str << " | color: " << (p.color() == Point::not_assigned ? "-" : std::to_string(p.color()));
    str << " cluster: " << (p.cluster() == Point::not_assigned ? "-" : std::to_string(p.cluster()));

    str << " ]";

    return str;
}

void read_point(std::istream& is, Point* p) {
    Index color;
    is >> color;
    p->assign_color(color);
    
    for(std::size_t i=0; i < p->dim(); ++i) {
        is >> (*p)[i];
    }   
}


double assign_points(ClusteringProblem& p, const std::vector<Point*>& centers) {
    double max_radius = 0.0;
    for(std::size_t i=0; i < p.n(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        Index min_center = 0;
        for(std::size_t j=0; j < p.k(); ++j) {
            const double dist = sq_dist(p.points()[i], *centers[j]);
            if(dist < min_dist) {
                min_dist = dist;
                min_center = j;
            }
        }
        max_radius = std::max(max_radius, min_dist);

        p.points()[i].assign_cluster(min_center);
    }

    return std::sqrt(max_radius);
}

std::vector<double> all_pairwise_distances(ClusteringProblem& p) {
    const std::size_t n_pairs = (p.n()*p.n()-1) / 2;
    std::vector<double> distances;
    distances.reserve(n_pairs);

    for(auto x=p.points().begin(); x != p.points().end(); ++x) {
        for(auto y=x+1; y != p.points().end(); ++y) {
            distances.push_back(sq_dist(*x, *y));
        }
    }

    std::sort(distances.begin(), distances.end());
    return distances;
}

}
