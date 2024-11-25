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

//currently n^3, I guess one can imporve this to n^2log(n)
std::vector<double> all_pairwise_distances_by_anc(ClusteringProblem& p) {
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

void ClusteringProblem::sort_by_colors(){
            std::size_t n=(*_points).size();
            auto color_order = [](const Point& p, const Point& q) {
                return p.color() < q.color();
            };

            std::sort(_points->begin(), _points->end(), color_order);

            _first_point_of_color.resize(_n_colors);
            _first_point_of_color[0] = _points->begin();
            for(Index i=1; i < _n_colors; ++i) {
                _first_point_of_color[i] = _first_point_of_color[i-1] + _n_points_of_color[i-1];
            }

            for(std::size_t i=0; i < n; ++i) {
                _points->at(i).set_idx(i);
            }
        }

void ClusteringProblem::read_input(std::istream& is) {
    std::size_t n;
    is >> n >> _dim >> _k >> _n_colors;

    if(_n_colors>2 || _n_colors==1){
        throw std::invalid_argument("Currently only 2 colors are supported.");
    }

    delete _points;
    _points = new PointSet(n, Point(_dim));

    _n_points_of_color.resize(_n_colors, 0);
    std::size_t i=0;
    for(auto& p : *_points) {
        read_point(is, &p);
        p.set_pos(i);
        _n_points_of_color[p.color()]++;
        i++;
    }
    sort_by_colors();
}

void ClusteringProblem::read_input(std::vector<std::vector<double>>& X, std::vector<size_t>& colors, std::size_t n_colors, std::size_t k, std::size_t ratio=1) {
    std::size_t n=X.size();
    _dim=X[0].size();
    _k=k;

    if(n_colors>2 || n_colors==1){
        throw std::invalid_argument("Currently only 2 colors are supported.");
    }

    _n_colors=n_colors;

    delete _points;
    _points = new PointSet(n, Point(_dim));

    _n_points_of_color.resize(_n_colors, 0);
    for(size_t i=0; i<n; i++) {
        for(size_t j=0; j<_dim; j++){
            (*_points)[i][j]=X[i][j];
        if(colors[i]>=_n_colors){
            throw std::invalid_argument("Colors must be n the range 0,..,n_colors.");
        }
        (*_points)[i].assign_color(colors[i]);
        (*_points)[i].set_pos(i);
        _n_points_of_color[(*_points)[i].color()]++;
        }
    }

    if(ratio==0){
        throw std::invalid_argument("Color-ratio must be at least 1.");
    }
    //only works with 2 colors
    if(_n_points_of_color[0]<_n_points_of_color[1] && ratio*_n_points_of_color[0]>_n_points_of_color[1]){
        throw std::invalid_argument("Color-ratio can not be achieved.");
    }
    else if(_n_points_of_color[1]<=_n_points_of_color[0] && ratio*_n_points_of_color[1]>_n_points_of_color[0]){
        throw std::invalid_argument("Color-ratio can not be achieved.");
    }
    _ratio=ratio;

    sort_by_colors();
}

void ClusteringProblem::read_input(double* X, std::size_t* colors, std::size_t n_colors, std::size_t n, std::size_t k, std::size_t dim, std::size_t ratio=1) {
    _dim=dim;
    _k=k;
    _n_colors=n_colors;

    if(n_colors>2 || n_colors==1){
        throw std::invalid_argument("Currently only 2 colors are supported.");
    }

    delete _points;
    _points = new PointSet(n, Point(_dim));

    _n_points_of_color.resize(_n_colors, 0);

    for(size_t i=0; i<n; i++) {
        for(size_t j=0; j<dim; j++){
            (*_points)[i][j]=X[i*dim+j];
        }
        (*_points)[i].assign_color(colors[i]);
        (*_points)[i].set_pos(i);
        if(colors[i]>=_n_colors){
            throw std::invalid_argument("Colors must be n the range 0,..,n_colors.");
        }
        _n_points_of_color[(*_points)[i].color()]++;
    }

    if(ratio==0){
        throw std::invalid_argument("Color-ratio must be at least 1.");
    }
    //only works with 2 colors
    if(_n_points_of_color[0]<_n_points_of_color[1] && ratio*_n_points_of_color[0]>_n_points_of_color[1]){
        throw std::invalid_argument("Color-ratio can not be achieved.");
    }
    else if(_n_points_of_color[1]<=_n_points_of_color[0] && ratio*_n_points_of_color[1]>_n_points_of_color[0]){
        throw std::invalid_argument("Color-ratio can not be achieved.");
    }
    _ratio=ratio;
    sort_by_colors();
}

}
