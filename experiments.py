import argparse
from datetime import datetime
import logging
import subprocess

import numpy as np
import pandas as pd
import fair_kcenter as fk
from fair_kcenter.util import JsonEncoder
from pathlib import Path
import os
import resource

from subsampling import balance_by_sampling, subsampling_uniform, ratio_balance_by_sampling
from cluster_cost_plot import cluster_cost_plot 

datasets = {
    "bank":         {"data_dir": "bank",     "file_prefix": "bank",                                          "critical_attribute": "married", "exclude_columns": ["row_id"]},
    "census":       {"data_dir": "census",   "file_prefix": "census",                                        "critical_attribute": "sex",     "exclude_columns": ["row_id", "race"]},
    "diabetes":     {"data_dir": "diabetes", "file_prefix": "diabetes",                                      "critical_attribute": "gender",  "exclude_columns": ["row_id", "race"]},
    "income":       {"data_dir": "income",   "file_prefix": "acs_income_2018_1-Year_person_CA_features",     "critical_attribute": "SEX",     "exclude_columns": ["row_id"]},
    "censusrace":   {"data_dir": "census",   "file_prefix": "census",                                        "critical_attribute": "race",    "exclude_columns": ["row_id", "sex"]},
    "diabetesrace": {"data_dir": "diabetes", "file_prefix": "diabetes",                                      "critical_attribute": "race",    "exclude_columns": ["row_id", "gender"]}
}

#N_CLUSTERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]
N_CLUSTERS = [1, 2, 5, 10, 20, 30, 40, 50, 100]
SEED = 51503
LOG_DIR = Path("logs") / "cluster_cost"
PLOT_DIR = Path("plots") / "cluster_cost"


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_r_usage():
    return resource.getrusage(resource.RUSAGE_SELF)

def get_cpu_time(start, stop):
    user_cpu = stop.ru_utime - start.ru_utime
    sys_cpu = stop.ru_stime - start.ru_stime
    
    return user_cpu + sys_cpu

def _setup_logger() -> logging.Logger:
    log_dir = LOG_DIR
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"run_normalized_cut_{get_timestamp()}.log"

    logger = logging.getLogger("experiments")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

logger = _setup_logger()



def find_true_color_ratios(col: np.ndarray) -> list[int]:   
    colors, counts = np.unique(col, return_counts=True)
    sorted_colors = np.argsort(colors)
    return [counts[i] for i in sorted_colors]
    
def round_color_ratios_to_minority_color(ratios: list[int]) -> list[int]:
    min_ratio = min(ratios)
    return [int(r // min_ratio) for r in ratios]

def run_experiment_fixed_cluster_size(dataset, n_clusters, seed, outfile=None):
    logger.info(f"--- Subexperiment for k={n_clusters}...")
    ds = datasets[dataset]

    our_cost = []
    ref_cost = []
    our_fairlet_cost = []
    ref_fairlet_cost = []

    instances = sorted(Path(f"instances/normalized/{ds['data_dir']}/subsamples/cut").glob(f"{ds['file_prefix']}_processed_*.csv"))
    n_subsamples_processed = 1
    for filename in instances:
        df = pd.read_csv(filename)
        df = df.drop(columns=ds["exclude_columns"])
            
        # --- Split into X and col ---
        col = df[ds["critical_attribute"]].to_numpy()
        X = df.drop(columns=ds["critical_attribute"]).to_numpy()

        # First cluster with outliers and rounded ratios
        true_ratios = find_true_color_ratios(col)
        ideal_ratios = round_color_ratios_to_minority_color(true_ratios)

        logger.info(f"Subsample {n_subsamples_processed:05}/{len(instances)} [{filename.name}, k={n_clusters}]: True ratios: {true_ratios}, ideal ratios: {ideal_ratios}")

        start_us = get_r_usage()
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
        stop_us = get_r_usage()
        cpu_us = get_cpu_time(start_us, stop_us)
        logger.info(f"Clustered in {cpu_us:.2f} seconds (user+sys CPU time).")

        our_cost.append(kcenter_us.inertia_)
        our_fairlet_cost.append(kcenter_us.inertia_fairlets_)

        # Then cluster with Chierichetti's approach 
        ref_without_anchors = True
        # if len(ideal_ratios) == 2 and ideal_ratios[0] == 1 and ideal_ratios[1] == 1:
        #     ref_without_anchors = False

        X_balanced, col_balanced = ratio_balance_by_sampling(X, col, ratios=np.array(ideal_ratios, dtype=int), seed=seed)

        start_ref = get_r_usage()
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
        stop_ref = get_r_usage()
        cpu_ref = get_cpu_time(start_ref, stop_ref)
        logger.info(f"Reference clustered in {cpu_ref:.2f} seconds (user+sys CPU time).")
        
        ref_cost.append(kcenter_ref.inertia_)
        ref_fairlet_cost.append(kcenter_ref.inertia_fairlets_)
        
        if outfile:
            with open(outfile, "a") as f:
                f.write(f"{dataset};{filename};{ds['critical_attribute']};{n_clusters};{seed};{true_ratios};{ideal_ratios};{our_cost[-1]};{ref_cost[-1]};{our_fairlet_cost[-1]};{ref_fairlet_cost[-1]};{cpu_us};{cpu_ref}\n")

        seed += 1
        n_subsamples_processed += 1

    return our_cost, ref_cost, our_fairlet_cost, ref_fairlet_cost

def run_experiment(dataset, cluster_sizes, seed, outfile=None):
    our_costs, ref_costs, our_fairlet_costs, ref_fairlet_costs = [], [], [], []
    n_custersizes_processed = 1
    for n_clusters in cluster_sizes:
        logger.info(f"Run for choice {n_custersizes_processed} of {len(cluster_sizes)} choices for k.")
        our_cost, ref_cost, our_fairlet_cost, ref_fairlet_cost = run_experiment_fixed_cluster_size(dataset, n_clusters, seed, outfile=outfile)
        our_costs.append(our_cost)
        ref_costs.append(ref_cost)
        our_fairlet_costs.append(our_fairlet_cost)
        ref_fairlet_costs.append(ref_fairlet_cost)
        logger.info(f"{dataset} | {n_clusters} clusters | our cost = {our_cost}, ref cost = {ref_cost}")
        seed += len(our_cost)
        n_custersizes_processed += 1

    return our_costs, ref_costs, our_fairlet_costs, ref_fairlet_costs


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

    return parser.parse_args()



if __name__ == "__main__":
    args        = parse_cli()
    kwargs      = vars(args)
    start       = get_timestamp()
    result_file = LOG_DIR / f"results_{args.dataset}_normalized_cut_{start}.csv"
    plot_file   = PLOT_DIR / f"cost_{args.dataset}_normalized_cut_{start}.pdf"

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    logger.info(f"Starting experiment with arguments: {kwargs} at time {start}...")
    logger.info(f"Running with cluster sizes {N_CLUSTERS}.")
    logger.info(f"Using {datasets[args.dataset]['critical_attribute']} as color column.")
    logger.info(f"Excluding columns {datasets[args.dataset]['exclude_columns']}.")
    logger.info(f"Results will be written to '{result_file}'.")
    logger.info(f"Plots will be written to '{plot_file}'.")
    
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        logger.info(f"Currently at git commit: {commit_hash}")
    except Exception as e:
        logger.warning(f"Could not retrieve git commit hash: {e}")

    with open(result_file, "a") as f:
        f.write(f"dataset;subsample;color;k;seed;true_ratios;ideal_ratios;our_cost;ref_cost;our_fairlet_cost;ref_fairlet_cost;our_cpu;ref_cpu\n")

    our_costs, ref_costs, our_fairlet_costs, ref_fairlet_costs = run_experiment(args.dataset, N_CLUSTERS, args.seed, result_file)
    cluster_cost_plot(result_file, plot_file)

    stop = get_timestamp()
    logger.info(f"Experiment completed at time {stop}.") 
