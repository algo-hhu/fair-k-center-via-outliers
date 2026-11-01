import argparse
import json
import logging
import sys
from os.path import join
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import fair_kcenter as fk
from fair_kcenter.util import JsonEncoder

logger = logging.getLogger(__name__)


def set_up_logger(log_file: Path, mode: str = "w+") -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s: %(levelname)s/%(filename)s:%(lineno)d] %(message)s"
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--k", type=int, required=True, help="Number of clusters.")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the input data. Should contain point coordinates and color of each point.")
    parser.add_argument("--ratios-path", type=Path, default=None, help="Path to the color ratios (optional).")
    parser.add_argument("--results-path", type=Path, default=None, help="Path where result should be stored (optional).")
    parser.add_argument("--no-outlier", action="store_true", help="If passed, an algorithm for the non-outlier version is performed on the data. Only allowed for two colors and color ratio 1:1.")
    parser.add_argument("--store-flets", action="store_true", help="If passed, the labeling of the fairlets is stored.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    set_up_logger(
        Path(
            join(
                args.results_path.parent,
                args.results_path.name.replace(".json", ".log"),
            )
        )
    )

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    data = pd.read_csv(args.data_path).to_numpy()
    colors = data[:, -1]
    X = data[:, 0:-1]
    logger.info(f"The data has the following shape: {X.shape}")

    color_ratios = None
    auto_ratios = True

    if args.ratios_path is not None:
        ratios_data = pd.read_csv(args.ratios_path).to_numpy(dtype=int)
        color_ratios = ratios_data[:, 0]
        auto_ratios = False

    kcenter = fk.Fairkcenter(
        n_clusters=args.k,
        flets=args.store_flets,
        without_anchors=not args.no_outlier,
    )

    kcenter.fit(
        X=X,
        col=colors,
        col_ratios=color_ratios,
        auto_ratios=auto_ratios,
    )

    res_eval: Dict[str, Any] = {}
    if args.store_flets:
        res_eval["fairlet_labels"] = kcenter.fairlets_
    res_eval.update(
        {
            "labels": kcenter.labels_,
            "centers": kcenter.cluster_centers_,
            "cost": kcenter.inertia_,
            "fairlet_cost": kcenter.inertia_fairlets_,
        }
    )

    if args.results_path is not None:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump(res_eval, f, indent=4, cls=JsonEncoder)
    else:
        with open("result.json", "w") as f:
            json.dump(res_eval, f, indent=4, cls=JsonEncoder)
