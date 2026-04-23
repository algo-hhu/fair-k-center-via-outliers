#ifndef FC_GONZALES_H_
#define FC_GONZALES_H_

#include "clusteringproblem.h"

#include <vector>

namespace fairclustering {

/**
 * @brief Center-aware version of fairlet gonzales with outliers
 *
 * For now, the algorithm ignores any point that is not the anchor of a fairlet.
 */
template<class iterator_t>
std::vector<Point*> gonzales(iterator_t begin, iterator_t end, std::size_t k) {
    if(k == 1) {
        std::vector<Point*> centers(1, nullptr);
        centers[0] = &(*begin);
        centers[0]->assign_cluster(0);
        return centers;
    }   
    
    std::vector<Point*> centers(k, nullptr);
    double max_dist = 0.0;
    int i=0;
    for(auto x=begin; x != end; ++x) {
        i++;
        //if(x->buddy() != nullptr) continue;
        for(auto y=x+1; y != end; ++y) {
            //if(y->buddy() != nullptr) continue;

            const double pq_dist = sq_dist(*x, *y);
            if(pq_dist > max_dist) {
                max_dist = pq_dist;
                centers[0] = &(*x);
                centers[1] = &(*y);
            }
        }
    }

    centers[0]->assign_cluster(0);
    centers[1]->assign_cluster(1);


    // choose center number c: Find point whose minimum distance to all fixed centers is maximal
    // among those points that:
    // - are not themselves fixed centers
    // - are not a partner (i.e., not the anchor) in a fairlet
    // - are not outliers
    for(std::size_t c=2; c < k; ++c) {
        double max_dist = 0.0;
        // for all points, check distance to all fixed centers
        for(auto x=begin; x != end; ++x) {
            if(    x->cluster() != Point::not_assigned
                //|| x->buddy() != nullptr
            ) {
                continue;
            }

            // compute closest center for point x in current center set
            double min_dist = std::numeric_limits<double>::max();
            for(std::size_t j=0; j < c; ++j) {
                const double dist = sq_dist(*x, *centers[j]);
                if(dist < min_dist) {
                    min_dist = dist;
                }
            }


            if(min_dist >= max_dist) {
                max_dist = min_dist;
                centers[c] = &(*x);
            }
        }
        centers[c]->assign_cluster(c);
    }

    return centers;
}
} // namespace fairclustering

#endif  // FC_GONZALES_H_
