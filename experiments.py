import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fair_kcenter as fk
from fair_kcenter.util import JsonEncoder
from pathlib import Path
import os

from subsampling import balance_by_sampling, subsampling_uniform, ratio_balance_by_sampling

datasets = {
    "bank": {"file": "bank_processed", "critical_attribute": "married", "exclude_columns": ["row_id"], "subsample": 1000},
    "census": {"file": "census_processed", "critical_attribute": "sex", "exclude_columns": ["row_id", "race"], "subsample": 600},
    "diabetes": {"file": "diabetes_processed", "critical_attribute": "gender", "exclude_columns": ["row_id", "race"], "subsample": 1000},
    "income": {"file": "acs_income_2018_1-Year_person_CA_features_processed", "critical_attribute": "SEX", "exclude_columns": ["row_id"], "subsample": 1000}
}

N_CLUSTERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]
SEED = 51503

def find_true_color_ratios(col: np.ndarray) -> list[int]:
    _, counts = np.unique(col, return_counts=True)
    return counts
    
def round_color_ratios_to_minority_color(ratios: list[int]) -> list[int]:
    min_ratio = min(ratios)
    return [int(r // min_ratio) for r in ratios]

def run_experiment_fixed_cluster_size(dataset, n_clusters, seed):
    ds = datasets[dataset]

    our_cost = []
    ref_cost = []
    for filename in sorted(Path(f"instances/normalized/{dataset}/subsamples/cut").glob(f"{dataset}_processed_*.csv")):
        df = pd.read_csv(filename)
        df = df.drop(columns=ds["exclude_columns"])
            
        # --- Split into X and col ---
        col = df[ds["critical_attribute"]].to_numpy()
        X = df.drop(columns=ds["critical_attribute"]).to_numpy()

        # First cluster with outliers and rounded ratios
        true_ratios = find_true_color_ratios(col)
        ideal_ratios = round_color_ratios_to_minority_color(true_ratios)

        print(f"Subsample {filename.name}: True ratios: {true_ratios}, ideal ratios: {ideal_ratios}")

        kcenter_us = fk.Fairkcenter(
            n_clusters=n_clusters,
            flets=True,
            without_anchors=True
        )

        kcenter_us.fit(
            X=X,
            col=col,
            col_ratios=np.array(ideal_ratios)
        )

        our_cost.append(kcenter_us.inertia_)

        # Then cluster with Chierichetti's approach 
        ref_without_anchors = True
        if len(ideal_ratios) == 2 and ideal_ratios[0] == 1 and ideal_ratios[1] == 1:
            ref_without_anchors = False

        X_balanced, col_balanced = ratio_balance_by_sampling(X, col, ratios=np.array(ideal_ratios, dtype=int), seed=seed)
        kcenter_ref = fk.Fairkcenter(
            n_clusters=n_clusters,
            flets=True,
            without_anchors=ref_without_anchors
        )

        kcenter_ref.fit(
            X=X_balanced,
            col=col_balanced,
            col_ratios=np.array(ideal_ratios)
        )

        ref_cost.append(kcenter_ref.inertia_)

        seed += 1
        

    return our_cost, ref_cost

def run_experiment(dataset, cluster_sizes, seed):
    our_costs, ref_costs = [], []
    for n_clusters in cluster_sizes:
        our_cost, ref_cost = run_experiment_fixed_cluster_size(dataset, n_clusters, seed)
        our_costs.append(our_cost)
        ref_costs.append(ref_cost)
        print(f"{dataset} | {n_clusters} clusters | our cost = {our_cost}, ref cost = {ref_cost}")
        seed += len(our_cost)

    return our_costs, ref_costs

def plot_experiment_results(dataset, cluster_sizes, our_costs, ref_costs):
    our_min  = [np.min(c)  for c in our_costs]
    our_max  = [np.max(c)  for c in our_costs]
    our_mean = [np.mean(c) for c in our_costs]
    ref_min  = [np.min(c)  for c in ref_costs]
    ref_max  = [np.max(c)  for c in ref_costs]
    ref_mean = [np.mean(c) for c in ref_costs]

    plt.figure(figsize=(10, 6))

    plt.plot(cluster_sizes, our_mean, marker="o", label="our method (mean)", linewidth=2, markersize=8, color="tab:blue")
    plt.fill_between(cluster_sizes, our_min, our_max, alpha=0.2, color="tab:blue", label="our method (min–max)")

    plt.plot(cluster_sizes, ref_mean, marker="s", label="chierichetti method (mean)", linewidth=2, markersize=8, color="tab:orange")
    plt.fill_between(cluster_sizes, ref_min, ref_max, alpha=0.2, color="tab:orange", label="chierichetti method (min–max)")

    plt.xlabel("Number of clusters", fontsize=14)
    plt.ylabel("Cluster cost", fontsize=14)
    plt.title(f"Cluster cost comparison on {dataset}.", fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(ticks=cluster_sizes, labels=[str(k) for k in cluster_sizes], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plot_path = f"plots/cluster_cost/"
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f"{plot_path}/cost_{dataset}_normalized_cut.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def parse_cli():
    parser = argparse.ArgumentParser("Call fair k-center code.")
 
    parser.add_argument(
        "dataset",
        type=str,
        choices=datasets.keys(),
        help="Dataset to use. Must be one of {', '.join(datasets.keys())}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for subsampling (default: {SEED}).",
    )
    # parser.add_argument(
    #     "color_ratios",
    #     type=int,
    #     nargs="+",
    #     help="List of integers specifying ratios.",
    # )
 
    # parser.add_argument(
    #     "--with-fairlet-labels",
    #     action="store_true",
    #     default=False,
    #     help="Print fairlet labels (default: False).",
    # )
    
    # parser.add_argument(
    #     "--with-anchors",
    #     action="store_true",
    #     default=False,
    #     help="Use anchors for fairlets  (default: anchors disabled).",
    # )
    
    # parser.add_argument(
    #     "--normalize",
    #     action="store_true",
    #     default=False,
    #     help="Use normalized data (default: False).",
    # )

    # parser.add_argument(
    #     "--subsample",
    #     action="store_true",
    #     default=False,
    #     help="Use a random subsample of the data (default: False)."
    # )

    # parser.add_argument(
    #     "--force-balance",
    #     action="store_true",
    #     default=False,
    #     help="Subsample dataset to have 1:1:...:1 ratio"
    # )
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_cli()
    kwargs = vars(args)

    our_costs, ref_costs = run_experiment(args.dataset, N_CLUSTERS, args.seed)
    plot_experiment_results(args.dataset, N_CLUSTERS, our_costs, ref_costs)
