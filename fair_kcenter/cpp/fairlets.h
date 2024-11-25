#ifndef FC_FAIRLETS_H_
#define FC_FAIRLETS_H_

#include "clusteringproblem.h"
#include <stdexcept>
#include <lemon/smart_graph.h>
#include <lemon/preflow.h>

namespace fairclustering {

using Graph = lemon::SmartDigraph;
using Node = Graph::Node;
using Arc = Graph::Arc;

double build_fairlets_vanilla(ClusteringProblem& p, Index red, Index blue);
double build_fairlets_outlier(ClusteringProblem& p, Index red, Index blue);
Graph* build_fairlet_graph(ClusteringProblem& p, Index red, Index blue, Node* s, Node* t);
void update_capacities(const ClusteringProblem& p, const Graph* g, double r, Index red, Index blue, lemon::Preflow<Graph>::CapacityMap* capacities);
void set_buddies_from_flow(ClusteringProblem& p, const Graph* g, const lemon::Preflow<Graph>::FlowMap& flow, Index blue);

}

#endif
