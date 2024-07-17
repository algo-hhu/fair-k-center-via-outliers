#include "clusteringproblem.h"

#include <lemon/smart_graph.h>
#include <lemon/preflow.h>

#include <iostream>
#include <vector>

namespace fc = fairclustering;

using Point=fc::Point;
using PointSet = fc::PointSet;
using Index = fc::Index;
using Graph = lemon::SmartDigraph;
using Node = Graph::Node;
using Arc = Graph::Arc;

/**
 * @brief Center-aware version of fairlet gonzales with outliers
 */
template<class iterator_t>
std::vector<Point*>* gonzales(iterator_t begin, iterator_t end, std::size_t k) {
    std::vector<Point*>* centers = new std::vector<Point*>(k, nullptr);

    double max_dist = 0.0;
    for(auto x=begin; x != end; ++x) {
        for(auto y=x+1; y != end; ++y) {
            const double pq_dist = fc::sq_dist(*x, *y);
            if(pq_dist > max_dist) {
                max_dist = pq_dist;
                centers->at(0) = &(*x);
                centers->at(1) = &(*y);
            }
        }
    }

    centers->at(0)->assign_cluster(0);
    centers->at(1)->assign_cluster(1);

    // choose center number c
    for(std::size_t c=2; c < k; ++c) {
        double max_dist = 0.0;
        
        // for all points, check distance to all fixed centers
        for(auto x=begin; x != end; ++x) {
            // skip points that have already been assigned, or ar the partner in a fairlet
            if(x->cluster() != fc::Point::not_assigned || x->buddy() != nullptr) {
                continue;
            }
            
            // compute closest center for point x in current center set
            double min_dist = std::numeric_limits<double>::max();
            for(std::size_t j=0; j < c; ++j) {
                const double dist = fc::sq_dist(*x, *centers->at(j));
                if(dist < min_dist) {
                    min_dist = dist;
                }
            }

            if(min_dist > max_dist) {
                max_dist = min_dist;
                centers->at(c) = &(*x);
            }
        }

        centers->at(c)->assign_cluster(c);
    }

    return centers;
}

Graph* build_fairlet_graph(fc::ClusteringProblem& p, Index red, Index blue, Node* s, Node* t) {
    Graph* g = new Graph();
    g->reserveNode(p.n() + 2);

    for(auto x : p.points()) {
        g->addNode();
    }

    *s = g->addNode(); // s has index n
    *t = g->addNode(); // t has index n+1

    for(auto& x : p.points_of_color(red)) {
        g->addArc(*s, g->nodeFromId(x.idx()));
        for(auto& y : p.points_of_color(blue)) {
            g->addArc(g->nodeFromId(x.idx()), g->nodeFromId(y.idx()));
        }
    }

    for(auto y : p.points_of_color(blue)) {
        g->addArc(g->nodeFromId(y.idx()), *t);
    }

    return g;
}

std::vector<double> compute_distances(fc::ClusteringProblem& p) {
    const std::size_t n_pairs = (p.n()*p.n()-1) / 2;
    std::vector<double> distances;
    distances.reserve(n_pairs);

    for(auto x=p.points().begin(); x != p.points().end(); ++x) {
        for(auto y=x+1; y != p.points().end(); ++y) {
            distances.push_back(fc::sq_dist(*x, *y));
        }
    }

    std::sort(distances.begin(), distances.end());
    return distances;
}

void update_capacities(const fc::ClusteringProblem& p, const Graph* g, double r, Index red, Index blue, lemon::Preflow<Graph>::CapacityMap* capacities) {
    for(auto x : p.points_of_color(red)) {
        for(Graph::OutArcIt e(*g, g->nodeFromId(x.idx())); e != lemon::INVALID; ++e) {
            const int j = g->id(g->target(e));

            if(fc::sq_dist(x, p.points(j)) <= r) {
                capacities->set(e, 1);
            }
            else {
                capacities->set(e, 0);
            }
        }
    }
}

void assign_buddies(fc::ClusteringProblem& p, const Graph* g, const lemon::Preflow<Graph>::FlowMap& flow, Index blue) {
    for(auto& x : p.points_of_color(blue)) {
        bool has_buddy = false;
        for(Graph::InArcIt e(*g, g->nodeFromId(x.idx())); e != lemon::INVALID; ++e) {
            const int j = g->id(g->source(e));

            if(flow[e] > 0) {
                p.points(x.idx()).assign_buddy(p.points(j));
                has_buddy = true;
                break;
            }
        }
        if(!has_buddy) x.mark_as_outlier();
    }
}

double build_fairlets(fc::ClusteringProblem& p, Index red, Index blue) {
    Node s, t;
    Graph* g = build_fairlet_graph(p, red, blue, &s, &t);
    auto capacities = lemon::Preflow<Graph>::CapacityMap(*g, 1);   
    lemon::Preflow<Graph> preflow(*g, capacities, s, t);   

    std::vector<double> distances = compute_distances(p);
    double min_r = std::numeric_limits<double>::max();
    std::size_t left = 0;
    std::size_t right = distances.size()-1;

    while(left < right) {
        int middle = (left + right) / 2;

        update_capacities(p, g, distances.at(middle), red, blue, &capacities);
        preflow.runMinCut();
        if((unsigned int) preflow.flowValue() == p.n_points(red)) {
            right = middle - 1;
        } else {
            left = middle + 1;
        }

        if(left >= right){
            min_r = distances.at(middle);
            break;
        } 
    }

    preflow.startSecondPhase();
    assign_buddies(p, g, preflow.flowMap(), blue);

    delete g;
    return min_r;
}

int main() {
    fc::ClusteringProblem problem;
    problem.read_input(std::cin);

    std::cout << "Eingabe:" << std::endl;
    for(auto& p : problem.points()) {
        std::cout << p << std::endl;
    }

    const Index red = (problem.n_points(0) <= problem.n_points(1) ? 0 : 1);
    const Index blue = 1 - red;

    double r = build_fairlets(problem, red, blue);

    auto centers = gonzales(problem.points_of_color(red).begin(), problem.points_of_color(red).end(), problem.k());
    assign_points(problem, *centers);

    std::cout << "Zentren:" << std::endl;
    for(auto& c : *centers) {
        std::cout << *c << std::endl;
    }

    std::cout << "Clustering mit Radius " << r << ":" << std::endl;
    for(auto& p : problem.points()) {
        std::cout << p << std::endl;
    }

    delete centers;

    return 0;
}
