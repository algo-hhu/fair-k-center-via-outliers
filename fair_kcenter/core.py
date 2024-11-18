import ctypes

import numpy as np

import fair_kcenter._core  # type: ignore

_DLL = ctypes.cdll.LoadLibrary(fair_kcenter._core.__file__)


class Fairkcenter:
    def __init__(
        self,
        n_clusters: int,
    ):
        self.n_clusters = n_clusters

    def fit(
        self,
        X: np.ndarray,
        col: np.ndarray,
        flets: bool = False,
    ) -> "Fairkcenter":
        if not isinstance(X, np.ndarray) or not isinstance(X, np.ndarray):
            raise ValueError("X muss ein numpy Array sein.")
        self.n_samples = X.shape[0]
        self.col_names = np.unique(col)
        self.n_col = self.col_names.shape[0]
        if self.n_col > 2:
            raise ValueError(
                f"Number of colors is {self.n_col} but currently only 2 colors are allowed."
            )
        self.n_features_in_ = X.shape[1]

        # assign every color a number betwenn 0 and n_col-1
        self._col = np.zeros(self.n_samples, dtype=int)
        for i, name in enumerate(self.col_names):
            self._col[col == name] = i

        if self.n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={self.n_samples} should be >= n_clusters={self.n_clusters}."
            )
        if self.n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={self.n_samples} should be >= n_clusters={self.n_clusters}."
            )

        self._X = np.ascontiguousarray(X.astype(np.float64))

        # Declare c types
        c_array_pts = self._X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_array_col = self._col.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        c_n = ctypes.c_uint(self.n_samples)
        c_d = ctypes.c_uint(self.n_features_in_)
        c_k = ctypes.c_uint(self.n_clusters)
        c_nc = ctypes.c_uint(self.n_col)
        c_labels = (ctypes.c_int * self.n_samples)()
        if flets:
            c_fairlets = (ctypes.c_int * self.n_samples)()
        else:
            c_fairlets = ctypes.POINTER(ctypes.c_int)()
        c_centers = (ctypes.c_double * self.n_features_in_ * self.n_clusters)()

        # Set the return type to double
        _DLL.fairKCenter.restype = ctypes.c_double
        cost = _DLL.fairKCenter(
            c_array_pts,
            c_array_col,
            c_n,
            c_d,
            c_k,
            c_nc,
            c_labels,
            c_centers,
            c_fairlets,
        )

        self.inertia_ = cost

        self.cluster_centers_ = np.ctypeslib.as_array(
            c_centers, shape=(self.n_clusters, self.n_features_in_)
        )

        self.labels_ = np.ctypeslib.as_array(c_labels)
        self.fairlets = None
        if c_fairlets is not None:
            self.fairlets = np.ctypeslib.as_array(c_fairlets)
        self._n_features_out = len(self.cluster_centers_)

        return self
