import numpy as np


def balance_by_sampling(X: np.ndarray, y: np.ndarray, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    keep_indices = []
    for label in unique:
        indices = np.where(y == label)[0]
        chosen = rng.choice(indices, size=min_count, replace=False)
        keep_indices.append(chosen)

    keep_indices = np.sort(np.concatenate(keep_indices))
    return X[keep_indices], y[keep_indices]


# X = np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]])
# y = np.array([0, 1, 0, 1, 0, 0, 2])

# X_balanced, y_balanced = balance_by_sampling(X, y)

# print("Balanced X:", X_balanced)
# print("Balanced y:", y_balanced)

def ratio_balance_by_sampling(X: np.ndarray, y: np.ndarray, ratios: np.ndarray, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(y, return_counts=True)
    ratios = np.asarray(ratios)
    t = int(np.min(counts / ratios))

    keep_indices = []
    for label, ratio in zip(unique, ratios):
        indices = np.where(y == label)[0]
        chosen = rng.choice(indices, size=t * ratio, replace=False)
        keep_indices.append(chosen)

    keep_indices = np.sort(np.concatenate(keep_indices))
    return X[keep_indices], y[keep_indices]


# X = np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]])
# y = np.array([0, 1, 0, 1, 0, 0, 2])
# # 
# X_balanced, y_balanced = ratio_balance_by_sampling(X, y, ratios=[2, 2, 1])
# # 
# print("Balanced X:", X_balanced)
# print("Balanced y:", y_balanced)


def subsampling_uniform(X: np.ndarray, y: np.ndarray, num_samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=num_samples, replace=False)
    indices = np.sort(indices)
    return X[indices], y[indices]

# X = np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]])
# y = np.array([0, 1, 0, 1, 0, 0, 2])

# X_sampled, y_sampled = subsampling_uniform(X, y, num_samples=5)

# print("Sampled X:", X_sampled)
# print("Sampled y:", y_sampled)

def subsampling_fair(X: np.ndarray, y: np.ndarray, num_samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(y, return_counts=True)
    num_classes = len(unique)
    count_per_point = counts[np.searchsorted(unique, y)]
    weights = 1.0 / (count_per_point * num_classes)
    indices = rng.choice(len(X), size=num_samples, replace=False, p=weights)
    indices = np.sort(indices)
    return X[indices], y[indices]