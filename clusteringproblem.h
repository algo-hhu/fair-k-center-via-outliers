#ifndef FC_CLUSTERINGPROBLEM_H_
#define FC_CLUSTERINGPROBLEM_H_

#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cassert>
#include <ranges>

namespace fairclustering {

typedef std::size_t Index;
class Point;

class Point {
    public:
        static const Index not_assigned = std::numeric_limits<Index>::max();

        Point(std::size_t dim, Index color=Point::not_assigned) :
            _coords(dim, 0.0),
            _color(color),
            _cluster(Point::not_assigned),
            _buddy(nullptr)
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

    private:
        std::vector<double> _coords;
        Index _color;
        Index _cluster;
        Index _idx;
        Point* _buddy;
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
            _n_points_of_color()
        {
        }

        virtual ~ClusteringProblem() {
            delete _points;
        }

        void read_input(std::istream& is) {
            int n; 
            is >> n >> _dim >> _k >> _n_colors;
            
            delete _points;
            _points = new PointSet(n, Point(_dim));
            
            _n_points_of_color.resize(_n_colors, 0);
            for(auto& p : *_points) {
                read_point(is, &p);
                _n_points_of_color[p.color()]++;
            }

            auto color_order = [](const Point& p, const Point& q) {
                return p.color() < q.color();
            };
            
            std::sort(_points->begin(), _points->end(), color_order);

            _first_point_of_color.resize(_n_colors);
            _first_point_of_color[0] = _points->begin();
            for(Index i=1; i < _n_colors; ++i) {
                _first_point_of_color[i] = _first_point_of_color[i-1] + _n_points_of_color[i-1];
            }

            for(std::size_t i=0; i < n(); ++i) {
                _points->at(i).set_idx(i);
            }
        }

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

        std::size_t n_points(Index color) const {
            assert(color < _n_colors);
            return _n_points_of_color[color];
        }

        auto points_of_color(Index color) const {
            return std::span(_first_point_of_color[color], _n_points_of_color[color]);
        }

    private:
        PointSet* _points;
        std::size_t _dim;
        std::size_t _k;
        std::size_t _n_colors;
        std::vector<PointSet::iterator> _first_point_of_color;
        std::vector<std::size_t> _n_points_of_color;
};

void assign_points(ClusteringProblem& p, const std::vector<Point*>& centers);

} // namespace fairclustering
#endif
