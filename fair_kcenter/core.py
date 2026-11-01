import ctypes
import logging

import numpy as np

import fair_kcenter._core  # type: ignore

_DLL = ctypes.cdll.LoadLibrary(fair_kcenter._core.__file__)
logger = logging.getLogger(__name__)


class Fairkcenter:
    def __init__(
        self,
        n_clusters: int,
        flets: bool = False,
        without_anchors: bool = True,
    ):
        self.n_clusters = n_clusters
        self._flets = flets
        self._without_anchors = without_anchors

    def fit(
        self,
        X: np.ndarray,
        col: np.ndarray,
        col_ratios: np.ndarray | None = None,
        auto_ratios: bool = False,
    ) -> "Fairkcenter":
        if not isinstance(X, np.ndarray) or not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")

        logger.info(
            "Compute fair k-center clusterig on dataset of shape"
            f"{X.shape} with k={self.n_clusters}"
        )

        if not self._without_anchors:
            logger.info("Compute solution with anchors")

        if self._flets:
            logger.info("Store fairlet assignment")

        self._col_names, col_sizes = np.unique(col, return_counts=True)
        self._n_samples = X.shape[0]

        if self._n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={self._n_samples} should be >= n_clusters={self.n_clusters}."
            )

        logger.info(f"Absolute distribution of colors {col_sizes}")
        self._n_col = int(self._col_names.max()) + 1

        if col_ratios is not None:
            if auto_ratios:
                logger.info(
                    "Use provided color ratios, automatic ratio setting will be ignored"
                )
            if np.all(col_ratios > 0):
                self._col_ratios = col_ratios.astype(int)
                if not np.issubdtype(col_ratios.dtype, np.integer):
                    logger.info(
                        f"Datatype of col_ratios {col_ratios.dtype} is convrerted to "
                        f"{np.integer}, col_ratio is now {self._col_ratios}"
                    )
            else:
                raise ValueError("Color ratios must be positive integers")
        else:
            if auto_ratios:
                logger.info("Set ratios atomatically")
                min = np.min(col_sizes)
                self._col_ratios = np.array([r // min for r in col_sizes], dtype=int)

            else:
                logger.info(
                    "No color ratios provided. All color ratios will be set to 1"
                )
                self._col_ratios = np.array([1] * self._n_col, dtype=int)

        logger.info(f"Color ratios are set to {self._col_ratios}")

        if not self._without_anchors:
            if self._n_col > 2:
                raise ValueError(
                    f"{self._n_col} colors provided but anchor version "
                    "is only available with 2 colors."
                )
            if len(self._col_ratios) != 2 or not np.all(self._col_ratios == 1):
                raise ValueError(
                    f"color ratios {self._col_ratios} provided "
                    "but anchor version only allows a 1:1 color ratio."
                )

        logger.info(f"Number of colors is set to {self._n_col}")
        if self._n_col != self._col_ratios.shape[0]:
            raise ValueError(
                f"provided color ratios of length={self._col_ratios.shape[0]} unequal "
                f"number of colors={self._n_col}."
            )
        self._n_features_in_ = X.shape[1]

        # assign every color a number betwenn 0 and n_col-1
        self._col = np.zeros(self._n_samples, dtype=int)
        for i, name in enumerate(self._col_names):
            pts = col == name
            self._col[pts] = i
            if self._col_ratios[i] == 1 and np.sum(pts) < self.n_clusters:
                raise ValueError(
                    f"Not enough points of color={name} to obtain n_clusters={self.n_clusters}."
                )

        self._X = np.ascontiguousarray(X.astype(np.float64))

        # Declare c types
        c_array_pts = self._X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_array_col = self._col.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
        c_array_col_ratio = self._col_ratios.ctypes.data_as(
            ctypes.POINTER(ctypes.c_size_t)
        )
        c_n = ctypes.c_size_t(self._n_samples)
        c_d = ctypes.c_size_t(self._n_features_in_)
        c_k = ctypes.c_size_t(self.n_clusters)
        c_nc = ctypes.c_size_t(self._n_col)
        c_labels = (ctypes.c_int32 * self._n_samples)()
        c_cost = (ctypes.c_double * 2)()
        c_without_anchors = ctypes.c_bool(self._without_anchors)
        if self._flets:
            c_fairlets = (ctypes.c_int32 * self._n_samples)()
        else:
            c_fairlets = ctypes.POINTER(ctypes.c_int32)()

        c_centers = (ctypes.c_double * self._n_features_in_ * self.n_clusters)()

        # Set the return type to double
        _DLL.fairKCenter.restype = ctypes.c_double
        _DLL.fairKCenter(
            c_array_pts,
            c_array_col,
            c_array_col_ratio,
            c_n,
            c_d,
            c_k,
            c_nc,
            c_labels,
            c_centers,
            c_fairlets,
            c_cost,
            c_without_anchors,
        )

        cost = np.ctypeslib.as_array(c_cost)

        self.inertia_ = cost[1]
        self.inertia_fairlets_ = cost[0]

        logger.info(
            f"Found solution with radius {self.inertia_} "
            f"and fairlet radius of {self.inertia_fairlets_}"
        )

        self.cluster_centers_ = np.ctypeslib.as_array(
            c_centers, shape=(self.n_clusters, self._n_features_in_)
        )

        self.labels_ = np.ctypeslib.as_array(c_labels)
        self.fairlets_ = None
        if self._flets:
            self.fairlets_ = np.ctypeslib.as_array(c_fairlets)

        return self
