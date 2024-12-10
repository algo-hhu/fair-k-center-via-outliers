#ifndef FC_CLUSTERINGPROBLEM_H_
#define FC_CLUSTERINGPROBLEM_H_

#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cassert>
#include <ranges>
#include <span>
#include <stdexcept>

namespace fairclustering {

typedef std::size_t Index;
class Point;

class Point {
    public:
        static const Index not_assigned = std::numeric_limits<Index>::max();

        Point(std::size_t dim, Index color=Point::not_assigned, Index pos=0) :
            _coords(dim, 0.0),
            _color(color),
            _cluster(Point::not_assigned),
            _pos(pos),
            _buddy(nullptr),
            _is_outlier(false)
        {
        }
        virtual ~Point() {
        }

        double operator[](std::size_t i) const {
            return _coords[i];
        }

        double& operator[](std::size_t i) {
            return _coords[i];
        }

        std::size_t dim() const {
            return _coords.size();
        }

        Index color() const {
            return _color;
        }

        void assign_color(Index color) {
            _color = color;
        }

        Index cluster() const {
            return _cluster;
        }

        void assign_cluster(Index cluster_idx) {
            _cluster = cluster_idx;
        }

        Point* buddy() const {
            return _buddy;
        }

        void assign_buddy(Point& buddy) {
            _buddy = &buddy;
        }

        const std::vector<double> coords() const {
            return _coords;
        }

        const Index idx() const {
            return _idx;
        }

        void set_idx(Index idx) {
            _idx = idx;
        }

        const Index pos() const{
            return _pos;
        }

        void set_pos(Index pos){
            _pos=pos;
        }

        bool is_outlier() const {
            return _is_outlier;
        }

        void mark_as_outlier() {
            _is_outlier = true;
        }

    private:
        std::vector<double> _coords;
        Index _color;
        Index _cluster;
        Index _idx;
        Index _pos;
        Point* _buddy;
        Index _index;
        bool _is_outlier;
};

typedef std::vector<Point> PointSet;

std::ostream& operator<<(std::ostream& str, const Point& p);
void read_point(std::istream& is, Point* p);

inline double sq_dist(const Point& p, const Point& q) {
    assert(p.dim() == q.dim());

    double squared_distance = 0.0;
    for(Index i=0; i < p.dim(); ++i) {
        squared_distance += (p[i] - q[i])*(p[i] - q[i]);
    }

    return squared_distance;
}

class ClusteringProblem {
    public:
        ClusteringProblem() :
            _points(nullptr),
            _dim(0),
            _k(0),
            _n_colors(0),
            _n_points_of_color(),
            _main_col(0)
        {
        }

        virtual ~ClusteringProblem() {
            delete _points;
        }

        void read_input(std::istream& is, bool outlier=true);

        void read_input(const std::vector<std::vector<double>>& X, const std::vector<std::size_t>& colors, std::size_t n_colors, std::size_t k, const std::vector<std::size_t>& ratios=std::vector<std::size_t>(), bool outlier=true);

        void read_input(const double* X, const std::size_t* colors, std::size_t n_colors, std::size_t n, std::size_t k, std::size_t dim, const std::size_t* ratios=NULL, bool outlier=true);

        void sort_by_colors();

        PointSet& points() {
            return *_points;
        }

        Point& points(std::size_t i) {
            return _points->at(i);
        }

        const Point& points(std::size_t i) const {
            return _points->at(i);
        }

        std::size_t dim() const {
            return _dim;
        }

        std::size_t k() const {
            return _k;
        }

        std::size_t n() const {
            return _points->size();
        }

        std::size_t n_colors() const {
            return _n_colors;
        }

        std::size_t main_col() const {
            return _main_col;
        }

        std::size_t n_points(Index color) const {
            assert(color < _n_colors);
            return _n_points_of_color[color];
        }

        auto points_of_color(Index color) const {
            return std::span(_first_point_of_color[color], _n_points_of_color[color]);
        }
        double assign_points(std::vector<Point*>& centers);
        std::pair<bool, double> center_update(std::vector<std::vector<Point*>>& fairlets, std::vector<Point*>& centers);

        /*
        void set_ratio(std::size_t ratio){
            _ratio=ratio;
        }*/

       std::size_t color_ratio(Index color) const {
            assert(color < _n_colors);
            return _ratios[color];
        }

        double pairwise_distance (Index i, Index j) const{
            return _pairwise_distances.at(i).at(j);
        }

        std::span<const double> pairwise_distances(Index i){
            return std::span<const double>(_pairwise_distances.at(i).begin(), _pairwise_distances.at(i).end());
        }

    private:
        PointSet* _points;
        std::size_t _dim;
        std::size_t _k;
        std::size_t _n_colors;
        std::vector<PointSet::iterator> _first_point_of_color;
        std::vector<std::size_t> _n_points_of_color;
        std::vector<std::size_t> _ratios; //clusters will have color-ratio 1:_ratio in the end, default 1:1
        std::vector<std::vector<double>> _pairwise_distances;
        std::size_t _main_col;

        void all_pairwise_distances();
        void all_pairwise_distances_by_anc();
};

} // namespace fairclustering
#endif
