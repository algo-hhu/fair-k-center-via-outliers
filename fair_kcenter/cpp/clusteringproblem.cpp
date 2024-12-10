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

std::pair<bool, double> ClusteringProblem::center_update(std::vector<std::vector<Point*>>& fairlets, std::vector<Point*>& centers){
    bool changed=false;
    double radius=0;

    //compute closest center to each fairlet
    //O(n*k*d)
    for(auto& ft: fairlets){
        double min_dist=std::numeric_limits<double>::max();
        Index center=std::numeric_limits<Index>::max();
        for(std::size_t i=0; i<centers.size(); i++){
            double max_dist=0;
            for(auto& pt: ft){
                double dist=sq_dist(*pt, *centers[i]);
                if(dist>max_dist){
                    max_dist=dist;
                }
            }
            if(min_dist>max_dist){
                min_dist=max_dist;
                center=i;
            }
        }
        for(auto& pt: ft){
            pt->assign_cluster(center);
        }
    }

    //compute clustering based on center assignment
    //O(n)
    std::vector<std::vector<Point*>> clustering(centers.size());
    for(auto& pt: points()){
        if(pt.is_outlier()) continue;
        Index c=pt.cluster();
        clustering[c].push_back(&pt);
    }

    //update centers, center aware: center must be part of the cluster
    //O(n^2*d) TODO:reuse computed distances
    for(std::size_t i=0; i<centers.size(); i++){
        if(clustering[i].size()==0) continue;
        Index best_c;
        double min_dist=std::numeric_limits<double>::max();
        for(auto c: clustering[i]){
            double max_c_dist=0;
            for(auto pt: clustering[i]){
                double dist=sq_dist(*c, *pt);
                if(dist>max_c_dist){
                    max_c_dist=dist;
                }
            }
            if(max_c_dist<min_dist){
                best_c=c->idx();
                min_dist=max_c_dist;
            }
        }

        if(best_c!=centers[i]->idx()){
            changed=true;
            centers[i]=&points(best_c);
        }

        if(radius<min_dist){
            radius=min_dist;
        }
    }
    return std::pair(changed, radius);

}

double ClusteringProblem::assign_points(std::vector<Point*>& centers) {
    std::pair<bool, double> a={true, std::numeric_limits<double>::max()};
    std::size_t i=0;

    //compute fairlets
    std::vector<std::vector<Point*>> fairlets(_n_points_of_color[_main_col]);
    Index offset=_first_point_of_color[_main_col]->idx();
    for(auto& pt: points()){
        if(pt.is_outlier()) continue;
        if(pt.color()==_main_col){
            Index fairlet=pt.idx()-offset;
            fairlets[fairlet].push_back(&pt);
        }
        else{
            Index fairlet=pt.buddy()->idx()-offset;
            fairlets[fairlet].push_back(&pt);
        }
    }


    while(a.first && i<100){
        a=center_update(fairlets, centers);
        i++;
    }
    return std::sqrt(a.second);
}

void ClusteringProblem::all_pairwise_distances() {
    _pairwise_distances.resize(n());
    for(std::size_t i=0; i<n(); i++){
        _pairwise_distances.at(i).resize(n());
        for(std::size_t j=0; j<=i; j++){
            _pairwise_distances.at(i).at(j)=_pairwise_distances.at(j).at(i);
        }
        for(std::size_t j=i; j<n(); j++){
            _pairwise_distances.at(i).at(j)=sq_dist(points(i), points(j));
        }
    }
}

//currently n^3 maybe one can improve this
void ClusteringProblem::all_pairwise_distances_by_anc() {
    _pairwise_distances.resize(n());
    for(std::size_t i=0; i<n(); i++){
        _pairwise_distances.at(i).resize(n());
        for(std::size_t j=0; j<=i; j++){
            _pairwise_distances.at(i).at(j)=_pairwise_distances.at(j).at(i);
        }
        for(std::size_t j=i; j<n(); j++){
            double min_dist=std::numeric_limits<double>::max();
            for(std::size_t l=0; l<n(); l++){
                double curr=std::max(sq_dist(points(i), points(l)), sq_dist(points(j), points(l)));
                if(curr<min_dist){
                    min_dist=curr;
                }
            }
            _pairwise_distances.at(i).at(j)=min_dist;
        }
    }
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

void ClusteringProblem::read_input(std::istream& is, bool outlier) {
    std::size_t n;
    is >> n >> _dim >> _k >> _n_colors;

    if(_n_colors==1){
        throw std::invalid_argument("Number of colors must be at least 2.");
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

    _ratios=std::vector<std::size_t>(_n_colors,1);

    std::size_t min=n;
    for(std::size_t col=0; col<_n_colors; col++){
        if(_ratios[col]<1) throw std::invalid_argument("Provided color ratios must be >0.");
        if(_n_points_of_color[col]<min){
            _main_col=col;
            min=_n_points_of_color[col];
        }
    }

    for(std::size_t col=0; col<_n_colors; col++){
        if(_n_points_of_color[_main_col]*_ratios[col]>_n_points_of_color[col] && _main_col!=col){
            throw std::invalid_argument("Color-ratios can not be achieved.");
        }
    }
    sort_by_colors();
    if(outlier){
        all_pairwise_distances();
    }
    else if(_n_colors!=2){
        throw std::invalid_argument("Fair k-center clustering without outliers only supported for 2 colors.");
    }
    else{
        all_pairwise_distances_by_anc();
    }
}

void ClusteringProblem::read_input(const std::vector<std::vector<double>>& X, const std::vector<size_t>& colors, std::size_t n_colors, std::size_t k, const std::vector<std::size_t>& ratios, bool outlier) {
    std::size_t n=X.size();
    _dim=X[0].size();
    _k=k;

    if(_n_colors==1){
        throw std::invalid_argument("Number of colors must be at least 2.");
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

    if(ratios.empty()){
        _ratios=std::vector<std::size_t>(_n_colors,1);
    }
    else if(ratios.size()!=_n_colors){
        throw std::invalid_argument("Provided color ratio does not match number of colors.");
    }
    else{
        _ratios=ratios;
    }

    std::size_t min=n;
    for(std::size_t col=0; col<_n_colors; col++){
        if(_ratios[col]<1) throw std::invalid_argument("Provided color ratios must be >0.");
        if(_n_points_of_color[col]<min){
            _main_col=col;
            min=_n_points_of_color[col];
        }
    }

    for(std::size_t col=0; col<_n_colors; col++){
        if(_n_points_of_color[_main_col]*_ratios[col]>_n_points_of_color[col] && _main_col!=col){
            throw std::invalid_argument("Color-ratios can not be achieved.");
        }
    }

    sort_by_colors();

    if(outlier){
        all_pairwise_distances();
    }
    else if(_n_colors!=2){
        throw std::invalid_argument("Fair k-center clustering without outliers only supported for 2 colors.");
    }
    else{
        all_pairwise_distances_by_anc();
    }
}

void ClusteringProblem::read_input(const double* X, const std::size_t* colors, std::size_t n_colors, std::size_t n, std::size_t k, std::size_t dim, const std::size_t* ratios, bool outlier) {
    _dim=dim;
    _k=k;
    _n_colors=n_colors;

    if(_n_colors==1){
        throw std::invalid_argument("Number of colors must be at least 2.");
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

    if(ratios==NULL){
        _ratios=std::vector<std::size_t>(_n_colors,1);
    }
    else{
        _ratios=std::vector<std::size_t>(_n_colors);
        for(std::size_t i=0; i<_n_colors; i++){
            _ratios[i]=ratios[i];
        }
    }

    std::size_t min=n;
    for(std::size_t col=0; col<_n_colors; col++){
        if(_ratios[col]<1) throw std::invalid_argument("Provided color ratios must be >0.");
        if(_n_points_of_color[col]<min){
            _main_col=col;
            min=_n_points_of_color[col];
        }
    }

    for(std::size_t col=0; col<_n_colors; col++){
        if(_n_points_of_color[_main_col]*_ratios[col]>_n_points_of_color[col] && _main_col!=col){
            throw std::invalid_argument("Color-ratios can not be achieved.");
        }
    }

    sort_by_colors();

    if(outlier){
        all_pairwise_distances();
    }
    else if(_n_colors!=2){
        throw std::invalid_argument("Fair k-center clustering without outliers only supported for 2 colors.");
    }
    else{
        all_pairwise_distances_by_anc();
    }
}

}
