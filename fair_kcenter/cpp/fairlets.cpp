#include "fairlets.h"

#include "clusteringproblem.h"

#include <lemon/smart_graph.h>
#include <lemon/preflow.h>

#include <cmath>

namespace fairclustering {



double build_fairlets(ClusteringProblem& p, Index red, Index blue) {
    Node s, t;
    Graph* g = build_fairlet_graph(p, red, blue, &s, &t);
    auto capacities = lemon::Preflow<Graph>::CapacityMap(*g, 1);   
    lemon::Preflow<Graph> preflow(*g, capacities, s, t);   

    std::vector<double> distances = all_pairwise_distances(p);
    double min_r = std::numeric_limits<double>::max();
    int left = 0;
    int right = distances.size()-1;

    while(left < right) {
        int middle = (left + right) / 2;

        update_capacities(p, g, distances.at(middle), red, blue, &capacities);
        preflow.runMinCut();
        if((unsigned int) preflow.flowValue() == p.n_points(red)) {
            right = middle - 1; 
            min_r = distances.at(middle);
        } else {
            left = middle + 1;
        }

        if(left >= right){
            break;
        } 
    }

    preflow.startSecondPhase();
    set_buddies_from_flow(p, g, preflow.flowMap(), blue);

    delete g;
    return std::sqrt(min_r);
}

Graph* build_fairlet_graph(ClusteringProblem& p, Index red, Index blue, Node* s, Node* t) {
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

void update_capacities(const ClusteringProblem& p, const Graph* g, double r, Index red, Index blue, lemon::Preflow<Graph>::CapacityMap* capacities) {
    for(auto x : p.points_of_color(red)) {
        for(Graph::OutArcIt e(*g, g->nodeFromId(x.idx())); e != lemon::INVALID; ++e) {
            const int j = g->id(g->target(e));

            if(sq_dist(x, p.points(j)) <= r) {
                capacities->set(e, 1);
            }
            else {
                capacities->set(e, 0);
            }
        }
    }
}

void set_buddies_from_flow(ClusteringProblem& p, const Graph* g, const lemon::Preflow<Graph>::FlowMap& flow, Index blue) {
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

}