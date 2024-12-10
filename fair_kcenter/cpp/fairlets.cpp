#include "fairlets.h"

#include "clusteringproblem.h"

#include <lemon/smart_graph.h>
#include <lemon/preflow.h>

#include <cmath>

namespace fairclustering {

void pairwise_distances_flat(ClusteringProblem & p, std::vector<double>& dist){
    std::vector<double> flat_dist(p.n()*p.n());

    for(std::size_t i=0; i<p.n(); i++){
        auto it=p.pairwise_distances(i);
        flat_dist.insert(flat_dist.end(),it.begin(), it.end());
    }

    sort(flat_dist.begin(),flat_dist.end());

    if(dist.size()!=0) dist.clear();
    dist.reserve((p.n()*(p.n()-1))/2);

    double curr_dist=flat_dist.at(0);
    dist.push_back(curr_dist);
    for(auto it: flat_dist){
        if(curr_dist!=it){
            dist.push_back(it);
            curr_dist=it;
        }
    }
}

double build_fairlets(ClusteringProblem& p) {
    std::vector<double> distances;
    pairwise_distances_flat(p, distances);
    double max_rad_all_col=std::numeric_limits<double>::max();
    for(std::size_t col=0; col<p.n_colors(); col++){
        if(col==p.main_col()) continue;
        Node s, t;
        Graph* g = build_fairlet_graph(p, p.main_col(), col, &s, &t);
        auto capacities = lemon::Preflow<Graph>::CapacityMap(*g, 1);
        lemon::Preflow<Graph> preflow(*g, capacities, s, t);

        double min_r = std::numeric_limits<double>::max();
        int left = 0;
        int right = distances.size()-1;

        double cap=distances.at(right);
        while(left <= right) {
            int middle = (left + right) / 2;
            update_capacities(p, g, distances.at(middle), p.main_col(), col, &capacities, s);
            preflow.runMinCut();
            if((unsigned int) preflow.flowValue() == p.n_points(p.main_col())*p.color_ratio(col)) {
                cap=distances.at(middle);
                right = middle - 1;
                min_r = distances.at(middle);
            } else {
                left = middle + 1;
            }
        }

        update_capacities(p, g, cap, p.main_col(), col, &capacities, s);
        preflow.runMinCut();

        preflow.startSecondPhase();
        set_buddies_from_flow(p, g, preflow.flowMap(), col);

        delete g;
        if(max_rad_all_col>min_r) max_rad_all_col=min_r;
    }
    return std::sqrt(max_rad_all_col);
}

Graph* init_graph(ClusteringProblem& p, Index red, Index blue, Node* s, Node* t){
    Graph* g = new Graph();
    g->reserveNode(p.n() + 2);

    for(std::size_t i=0; i<p.n(); i++) {
        g->addNode();
    }

    *s = g->addNode(); // s has index n_red+n_blue
    *t = g->addNode(); // t has index n_red+n_blue+1

    for(auto& x : p.points_of_color(red)) {
        g->addArc(*s, g->nodeFromId(x.idx()));
    }
    for(auto y : p.points_of_color(blue)) {
        g->addArc(g->nodeFromId(y.idx()), *t);
    }

    return g;
}

Graph* build_fairlet_graph(ClusteringProblem& p, Index red, Index blue, Node* s, Node* t) {
    Graph* g = init_graph(p, red, blue, s, t);

    for(auto& x : p.points_of_color(red)) {
        for(auto& y : p.points_of_color(blue)) {
            if(x.idx()!=y.idx()){
                g->addArc(g->nodeFromId(x.idx()), g->nodeFromId(y.idx()));
            }
        }
    }

    return g;
}


void update_capacities(const ClusteringProblem& p, const Graph* g, double r, Index red, Index blue, lemon::Preflow<Graph>::CapacityMap* capacities, Node& s) {
    std::size_t ratio=p.color_ratio(blue);
    for(Graph::OutArcIt e(*g, s); e != lemon::INVALID; ++e) {
        capacities->set(e, ratio);
    }
    for(auto x : p.points_of_color(red)) {
        for(Graph::OutArcIt e(*g, g->nodeFromId(x.idx())); e != lemon::INVALID; ++e) {
            const int j = g->id(g->target(e));

            if(p.pairwise_distance(x.idx(), j) <= r) {
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
