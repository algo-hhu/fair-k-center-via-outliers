import ctypes
import logging
from typing import Any, Optional

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
        col_ratios: Optional[np.ndarray] = None,
        auto_ratios: bool = False,
    ) -> "Fairkcenter":
        self._validate_input(X, col)
        self._log_configuration(X)

        self._col_names, col_sizes = np.unique(col, return_counts=True)
        self._n_samples = X.shape[0]
        self._n_features_in_ = X.shape[1]
        self._check_enough_samples()

        logger.info(f"Absolute distribution of colors {col_sizes}")
        self._n_col = int(self._col_names.max()) + 1

        self._col_ratios = self._resolve_col_ratios(col_ratios, auto_ratios, col_sizes)
        logger.info(f"Color ratios are set to {self._col_ratios}")

        if not self._without_anchors:
            self._validate_anchor_constraints()

        logger.info(f"Number of colors is set to {self._n_col}")
        self._validate_ratio_length()

        self._col = self._encode_colors(col)
        self._X = np.ascontiguousarray(X.astype(np.float64))

        self._run_solver()
        return self

    # ------------------------------------------------------------------
    # Validierung
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_input(X: np.ndarray, col: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if not isinstance(col, np.ndarray):
            raise ValueError("col must be a numpy array")

    def _check_enough_samples(self) -> None:
        if self._n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={self._n_samples} should be >= "
                f"n_clusters={self.n_clusters}."
            )

    def _log_configuration(self, X: np.ndarray) -> None:
        logger.info(
            f"Compute fair k-center clustering on dataset of shape {X.shape} "
            f"with k={self.n_clusters}"
        )
        if not self._without_anchors:
            logger.info("Compute solution with anchors")
        if self._flets:
            logger.info("Store fairlet assignment")

    def _validate_anchor_constraints(self) -> None:
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

    def _validate_ratio_length(self) -> None:
        if self._n_col != self._col_ratios.shape[0]:
            raise ValueError(
                f"provided color ratios of length={self._col_ratios.shape[0]} "
                f"unequal number of colors={self._n_col}."
            )

    # ------------------------------------------------------------------
    # Farbverhältnisse
    # ------------------------------------------------------------------
    def _resolve_col_ratios(
        self,
        col_ratios: Optional[np.ndarray],
        auto_ratios: bool,
        col_sizes: np.ndarray,
    ) -> np.ndarray:
        if col_ratios is not None:
            if auto_ratios:
                logger.info(
                    "Use provided color ratios, automatic ratio setting will be ignored"
                )
            return self._validate_provided_ratios(col_ratios)

        if auto_ratios:
            logger.info("Set ratios automatically")
            ratios: np.ndarray = (col_sizes // np.min(col_sizes)).astype(int)
            return ratios

        logger.info("No color ratios provided. All color ratios will be set to 1")
        return np.ones(self._n_col, dtype=int)

    @staticmethod
    def _validate_provided_ratios(col_ratios: np.ndarray) -> np.ndarray:
        if not np.all(col_ratios > 0):
            raise ValueError("Color ratios must be positive integers")
        ratios = col_ratios.astype(int)
        if not np.issubdtype(col_ratios.dtype, np.integer):
            logger.info(
                f"Datatype of col_ratios {col_ratios.dtype} is converted to "
                f"int, col_ratios is now {ratios}"
            )
        return ratios

    # ------------------------------------------------------------------
    # Farben kodieren
    # ------------------------------------------------------------------
    def _encode_colors(self, col: np.ndarray) -> np.ndarray:
        """Assign every color a number between 0 and n_col-1."""
        encoded = np.zeros(self._n_samples, dtype=int)
        for i, name in enumerate(self._col_names):
            pts = col == name
            encoded[pts] = i
            if self._col_ratios[i] == 1 and np.sum(pts) < self.n_clusters:
                raise ValueError(
                    f"Not enough points of color={name} "
                    f"to obtain n_clusters={self.n_clusters}."
                )
        return encoded

    # ------------------------------------------------------------------
    # C-Bibliothek aufrufen
    # ------------------------------------------------------------------
    def _run_solver(self) -> None:
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
        c_centers = (ctypes.c_double * self._n_features_in_ * self.n_clusters)()

        if self._flets:
            c_fairlets = (ctypes.c_int32 * self._n_samples)()
        else:
            c_fairlets = ctypes.POINTER(ctypes.c_int32)()

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

        self._store_results(c_cost, c_centers, c_labels, c_fairlets)

    def _store_results(
        self,
        c_cost: Any,
        c_centers: Any,
        c_labels: Any,
        c_fairlets: Any,
    ) -> None:
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
        self.fairlets_ = np.ctypeslib.as_array(c_fairlets) if self._flets else None
