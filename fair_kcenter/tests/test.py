import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

import fair_kcenter as fk
from fair_kcenter.util import JsonEncoder

def store_fcenter(output: Path, sol)->None:
    result={
        "cost": sol.inertia_, 
        "k":sol.n_clusters, 
        "centers": sol.cluster_centers_, 
        "labels": sol.labels_
    }
    with open(output, "w") as file: 
        json.dump(result, file, indent=4, cls=JsonEncoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    df = pd.read_csv(args.input, sep=" ")
    colors = np.array(df.iloc[:, 0])
    X = np.array(df.iloc[:, 1:])

    sol = fk.Fairkcenter(args.k)
    sol.fit(X, colors)

    store_fcenter(args.output, sol)
