import numpy as np
import fair_kcenter as fk
from fair_kcenter.util import JsonEncoder

if __name__ == "__main__":

    X=np.array([[-1,0,0],[-2,0,0],[-3,0,0],[0,1,0],[0,2,0],[0,3,0],[0,4,0],[-4,0,0]])
    col=np.array([0,1,2,0,1,2,2,2])

    #Optional: if you want you can specify color ratios which should be achieved 
    color_ratios=np.array([1,1,2])

    n_clusters=2

    #Optional: if you want to know the fairlest labels
    flets=True

    kcenter = fk.Fairkcenter(
        n_clusters=n_clusters,
        flets=flets,
    )

    kcenter.fit(
        X=X,
        col=col,
        col_ratios=color_ratios,
    )

    print(kcenter.labels_)
    print(kcenter.cluster_centers_)
