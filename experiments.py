import argparse

import numpy as np
import pandas as pd
import fair_kcenter as fk
from fair_kcenter.util import JsonEncoder

from subsampling import balance_by_sampling, subsampling_uniform

datasets = {
    "bank": {"file": "bank_processed", "critical_attribute": "married", "exclude_columns": [], "subsample": 1000},
    "census": {"file": "census_processed", "critical_attribute": "sex", "exclude_columns": [], "subsample": 600},
    "diabetes": {"file": "diabetes_processed", "critical_attribute": "gender", "exclude_columns": [], "subsample": 1000},
    "income": {"file": "acs_income_2018_1-Year_person_CA_features_processed", "critical_attribute": "SEX", "exclude_columns": [], "subsample": 1000}
}

def run_experiment(dataset, color_ratios, n_clusters, with_fairlet_labels, with_anchors, normalize, subsample, force_balance, num_iterations=5):
    ds = datasets[dataset]

    suffix = "-normalized" if normalize else ""
    df = pd.read_csv(f"instances/{ds['file']}{suffix}.csv")
    df = df.drop(columns=ds["exclude_columns"])
        
    # --- Split into X and col ---
    col = df[ds["critical_attribute"]].to_numpy()
    X = df.drop(columns=ds["critical_attribute"]).to_numpy()

    # if subsample:
    #     X, col = subsampling_uniform(X, col, ds["subsample"], seed=4711)
    
    costs = np.empty(num_iterations)

    for i in range(num_iterations):
        X_sub, col_sub = subsampling_uniform(X, col, subsample_size, seed=4711+i)

        # print(f"Subsample size: {subsample_size}, color distribution: {np.bincount(col_sub)}")

        if force_balance:
            X, col = balance_by_sampling(X, col, 4711-i)

        # call fair kcenter
        kcenter = fk.Fairkcenter(
            n_clusters=n_clusters,
            flets=with_fairlet_labels,
        )

        kcenter.fit(
            X=X,
            col=col,
            col_ratios=np.array(color_ratios)
        )

        assigned_centers = kcenter.cluster_centers_[kcenter.labels_]
        costs[i] = np.max(np.linalg.norm(X - assigned_centers, axis=1))

    return costs.mean()

def parse_cli():
    parser = argparse.ArgumentParser("Call fair k-center code.")
 
    parser.add_argument(
        "dataset",
        type=str,
        choices=datasets.keys(),
        help="Dataset to use. Must be one of {', '.join(datasets.keys())}.",
    )
    parser.add_argument(
        "color_ratios",
        type=int,
        nargs="+",
        help="List of integers specifying ratios.",
    )
 
    parser.add_argument(
        "--with-fairlet-labels",
        action="store_true",
        default=False,
        help="Print fairlet labels (default: False).",
    )
    
    parser.add_argument(
        "--with-anchors",
        action="store_true",
        default=False,
        help="Use anchors for fairlets  (default: anchors disabled).",
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Use normalized data (default: False).",
    )

    parser.add_argument(
        "--subsample",
        action="store_true",
        default=False,
        help="Use a random subsample of the data (default: False)."
    )

    parser.add_argument(
        "--force-balance",
        action="store_true",
        default=False,
        help="Subsample dataset to have 1:1:...:1 ratio"
    )
    
    return parser.parse_args()

N_CLUSTERS = [2, 3, 5, 10]

if __name__ == "__main__":
    args = parse_cli()
    kwargs = vars(args)
    for n_clusters in N_CLUSTERS:
        mean_cost = run_experiment(n_clusters=n_clusters, **kwargs)
        print(f"n_clusters={n_clusters}: mean cost = {mean_cost}")

