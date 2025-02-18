import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# from typing import Optional


# red points are on upper grid part, blue points on lower grid part
def seperated(n: int, m: int, ratios: list) -> np.ndarray:
    size = n * m
    total_rat = sum(ratios)

    num_pts = [math.floor(size * r / total_rat) for r in ratios]
    if sum(num_pts) != size:
        num_pts[-1] += size - sum(num_pts)

    points = np.empty((size + 1, 3), dtype=int)
    points[0] = np.array([size, 2, len(ratios)])

    left: int = 1
    right: int = 1
    col: int = 0
    for num in num_pts:
        right += num
        points[left:right] = np.array(
            [[col, i % n, i // n] for i in range(left - 1, right - 1)]
        )
        left = right
        col += 1

    return points


"""
# draw red points uniformly at random
def random(n: int, m: int, ratio: float, seed: Optional[int]) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    size = n * m
    num_r = math.floor(size * ratio)
    red_pts = np.random.choice(range(size), size=num_r, replace=False)
    blue_pts = np.array([i for i in range(size) if i not in red_pts])

    points = np.empty(shape=(size + 1, 3), dtype=int)
    points[0] = np.array([size, 2, 2])
    points[red_pts + 1] = np.array([[0, i % n, i // n] for i in red_pts])
    points[blue_pts + 1] = np.array([[1, i % n, i // n] for i in blue_pts])

    return points


# select red points evenly over the whole grid
def even(n: int, m: int, ratio: float) -> np.ndarray:
    size = n * m
    num_r = math.floor(size * ratio)
    grid = np.ones(shape=(n, m))
    t = math.floor(math.sqrt(num_r * m / n))
    s = num_r // t
    step_m = math.ceil(m / t)
    step_n = math.ceil(n / s)

    for i in range(s):
        for j in range(t):
            x = step_n * i if (step_n * i) < n else n - 1
            y = step_m * j if (step_m * j) < m else m - 1
            grid[x][y] = 0
    points = np.concatenate(
        (
            np.array([[size, 2, 2]]),
            np.array(
                [[grid[i][j], i, j] for i in range(n) for j in range(m)], dtype=int
            ),
        )
    )
    return points
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="grid_instances.py",
        description="Script to create grid instances for fair clustering",
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    # parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument(
        "--type", type=str, default="seperated", choices=["seperated", "random", "even"]
    )
    parser.add_argument("--result-file", type=Path, default=None)

    args = parser.parse_args()

    ratios = [2, 1, 5]

    points = None
    if args.type == "seperated":
        points = seperated(args.n, args.m, ratios)

    # elif args.type == "random":
    #    points = random(args.n, args.m, ratios, args.seed)
    # elif args.type == "even":
    #    points = even(args.n, args.m, ratios)

    df = pd.DataFrame(points)
    if args.result_file is not None:
        df.to_csv(args.result_file)
    else:
        df.to_csv(
            f"{args.type}_{args.n}_{args.m}_{ratios}.csv",
            sep=" ",
            index=False,
            header=False,
        )
