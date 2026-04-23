import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fair_kcenter as fk
from fair_kcenter.util import JsonEncoder

from subsampling import balance_by_sampling, subsampling_uniform

datasets = {
    "bank": {"file": "bank_processed", "critical_attribute": "married", "exclude_columns": [], "subsample": 1000},
    "census": {"file": "census_processed", "critical_attribute": "sex", "exclude_columns": ["race"], "subsample": 600},
    "diabetes": {"file": "diabetes_processed", "critical_attribute": "gender", "exclude_columns": ["race"], "subsample": 1000},
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
    costs_chierichetti = np.empty(num_iterations)

    for i in range(num_iterations):
        X_i, col_i = X, col

        if subsample:
            X_i, col_i = subsampling_uniform(X_i, col_i, ds["subsample"], seed=4711+i)



        kcenter = fk.Fairkcenter(
            n_clusters=n_clusters,
            flets=with_fairlet_labels,
            without_anchors=True
        )

        kcenter.fit(
            X=X_i,
            col=col_i,
            col_ratios=np.array(color_ratios)
        )

        costs[i] = kcenter.inertia_

        X_i, col_i = balance_by_sampling(X_i, col_i, 4711-i)
        kcenter = fk.Fairkcenter(
            n_clusters=n_clusters,
            flets=with_fairlet_labels,
            without_anchors=False
        )

        kcenter.fit(
            X=X_i,
            col=col_i,
            col_ratios=np.array([1,1])
        )

        costs_chierichetti[i] = kcenter.inertia_

    return costs.mean(), costs_chierichetti.mean()

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

    mean_costs, mean_costs_chierichetti = [], []
    for n_clusters in N_CLUSTERS:
        mean_cost, mean_cost_chierichetti = run_experiment(n_clusters=n_clusters, **kwargs)
        print(f"n_clusters={n_clusters}: mean cost = {mean_cost}, mean cost (Chierichetti) = {mean_cost_chierichetti}")
        mean_costs.append(mean_cost)
        mean_costs_chierichetti.append(mean_cost_chierichetti)

    plt.plot(N_CLUSTERS, mean_costs, marker="o", label="Fair k-center")
    plt.plot(N_CLUSTERS, mean_costs_chierichetti, marker="o", label="Chierichetti")
    plt.xlabel("k")
    plt.ylabel("Mean clustering cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results_{args.dataset}.png", dpi=150)
    plt.show()

