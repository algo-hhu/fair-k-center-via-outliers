#include "clusteringproblem.h"

#include <vector>
#include <iostream>
#include <iomanip>

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


void assign_points(ClusteringProblem& p, const std::vector<Point*>& centers) {
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

        p.points()[i].assign_cluster(min_center);
    }
}

}
