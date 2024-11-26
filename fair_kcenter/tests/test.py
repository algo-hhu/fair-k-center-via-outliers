import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fair_kcenter as fk
from fair_kcenter.util import JsonEncoder


def plot_by_dict(
    res: Dict[str, Any], X: np.ndarray, colors: np.ndarray, name: Path
) -> None:
    centers = res.get("centers", None)
    k = res.get("n_clusters", None)
    labels = res.get("labels", None)
    if centers is None:
        raise ValueError("Key 'centers' not found.")
    if k is None:
        raise ValueError("Key 'n_clusters' not found.")
    if labels is None:
        raise ValueError("Key 'labels' not found.")

    if X.shape[1] != 2:
        raise ValueError(f"Dimension should be 2 but is {X.shape[1]}.")

    fig, ax = plt.subplots()
    ax.scatter(
        X[:, 0], X[:, 1], s=50, c=colors, cmap="Accent", edgecolors="k", zorder=2
    )
    for i in range(0, k):
        center = centers[i]
        ax.scatter(
            center[0],
            center[1],
            marker="X",
            s=100,
            color="red",
            edgecolors="k",
            zorder=3,
        )
    for j in range(X.shape[0]):
        pt = X[j]
        if labels[j] == k:
            continue
        center = centers[labels[j]]
        ax.plot([center[0], pt[0]], [center[1], pt[1]], color="k", zorder=1, alpha=0.5)
    ax.set_aspect("equal")
    # plt.axis("off")
    plt.savefig(name)


def plot_by_dict_fairlets(
    res: Dict[str, Any], X: np.ndarray, colors: np.ndarray, name: Path
) -> None:
    fairlets = res.get("fairlets", None)

    if fairlets is None:
        raise ValueError("Key 'fairlets' not found.")

    if X.shape[1] != 2:
        raise ValueError(f"Dimension should be 2 but is {X.shape[1]}.")

    fig, ax = plt.subplots()
    ax.scatter(
        X[:, 0], X[:, 1], s=50, c=colors, cmap="Accent", edgecolors="k", zorder=2
    )
    for j in range(X.shape[0]):
        if j == fairlets[j]:
            continue
        pt = X[j]
        pt_buddy = X[fairlets[j]]
        ax.plot(
            [pt_buddy[0], pt[0]], [pt_buddy[1], pt[1]], color="k", zorder=1, alpha=0.5
        )
    ax.set_aspect("equal")
    # plt.axis("off")
    plt.savefig(name)


def solution_as_dict(sol: fk.Fairkcenter) -> Dict:
    result = {
        "cost": sol.inertia_,
        "fairlet_cost": sol.inertia_fairlets,
        "n_clusters": sol.n_clusters,
        "centers": sol.cluster_centers_,
        "labels": sol.labels_,
    }
    if sol.fairlets is not None:
        result.update({"fairlets": sol.fairlets})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    df = pd.read_csv(args.input, sep=" ")
    colors = np.array(df.iloc[:, 0])
    X = np.array(df.iloc[:, 1:])

    sol = fk.Fairkcenter(args.k)
    sol.fit(X, colors, flets=True)

    res = solution_as_dict(sol)
    file_name = os.path.splitext(os.path.basename(args.input))[0]
    plot_by_dict(
        res=res, X=X, colors=colors, name=Path(f"plots/{file_name}_{args.k}.png")
    )
    plot_by_dict_fairlets(
        res=res,
        X=X,
        colors=colors,
        name=Path(f"plots/{file_name}_fairlets_{args.k}.png"),
    )
    with open(f"results/{file_name}_{args.k}", "w") as file:
        json.dump(res, file, indent=4, cls=JsonEncoder)
