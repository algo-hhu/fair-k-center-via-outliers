# fair k-center

This package computes an approximate $k$-center clustering under fairness constraints. In this fairness model every point has a color. The task of the clustering is to partition the points into clusters such that the color ratios of the data set are preserved in the clusters. For example if we have the two colors red and blue and the ratio of red to blue points is 1:2 in the dataset then we want that every cluster has a ratio of 1:2 of red and blue points. 

A fair clustering can be computed for arbitrary many colors $col_1,\ldots, col_s$ as long as the underlying color ratio is of the form $1\colon t_2:\ldots\colon t_s$ with $t_i\in\mathbb N$. The computed $k$-center clustering will satisfy this color ratio in all clusters.

## fair k-center with outliers
If the dataset has no underlying color ratio of this form, one can pass a desired color ratio of the form $1\colon t_2:\ldots\colon t_s$ with $t_i\in\mathbb N$ that should be achieved. If possible, the algortihm will achieve this color ratio in all clustres by declaring some points of color $col_2,\ldots, col_s$ as outliers if needed. Points declared as outliers are not part of any cluster. Futhermore the algorithm will not declare any points of color $col_1$ as outliers. 

### Parameters:
- `n_clusters: int` 
  Number of clusters k of the k-center solution
- `flets: bool, default=False` 
  If set to True the labeling of the fairlet assignment is stored
- `outlier: bool, default=True` 
  Should be set to False only when the data has 2 colors and the color ratio in the data is exactly 1:1. If set to False the clustering is computed with an algortihm that can't handle outliers. This may produce better results in the special case of 2 colors and color ratio 1:1.  

### Attributes:
- `labels_: ndarray of shape (n_samples,) dtype=int` 
  Labels between 0 and n_clusters-1 indicating which points belong to the same cluster and which center represents the cluster. 
- `fairlets_: ndarray of shape (n_samples,) dtype=int` 
  Labels between 0 and n_samples-1 indicating which points belong to the same fairlet and which center represents the fairlet.
- `cluster_centers_: ndarray of shape (n_clusters, n_features)` 
  Coordinates of the cluster centers, center at position i represents all points with label i.
- `inertia_: float` 
  $k$-center cost (radius) of the clustering 
- `inertia_fairlets_: float` 
  radius of the fairlets


### `fit(X, col, col_ratios=None, auto_ratios=False):`
- `X: ndarray of shape (n_samples, n_features)` 
  Coordinates of points which are clustered
- `col: ndarray of shape (n_samples, )` 
  Colors of points
- `col_ratios: ndarray of shape (n_colors, ), default=None`  
  Color ratios that should be satisfied in each cluster. If the numbers are not integers, they are converted to integers in the method. The colors are sorted inreasingly, then the i-th color ratio refers to the i-th smallest color. At least one of the ratios should be 1.
- `auto_ratios: bool, default=False` 
  If set to True and no col_ratios are given the ratios are picked automatically in the following way. First the color with the fewest number of points $m$ is determined. For every color we determine the number of points of this color $m'$ and set the respective ratio to $\lfloor\frac{m'}{m}\rfloor$.
  If neither `auto_ratios` is set to True nor `col_ratios` is passed, the color ratios are set to $1:1:...:1$.


## Usage

Install [poetry](https://python-poetry.org/docs/#installation)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install the package
```bash
poetry install
```

Run the program for fair_kcenter
```bash
poetry run python -m fair_kcenter --data-path your_data.csv --ratios-path your_ratios.csv --verbose --results-path your_result.json --k 3
```

Your `data-path` should be a file with a header containing the comma-separated data points and their color in the last column. Example:
```csv
0,1,2,3,4
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,1
4.7,3.2,1.3,0.2,1
4.6,3.1,1.5,0.2,0
5.0,3.6,1.4,0.2,2
...
```

Your `ratios-path` should be a file with a header containing comma-separated values. It should have the same length as the number of distinct colors. Example:

```csv
ratios
1
3
2
...
```

## Example:

```python
import numpy as np
import fair_kcenter as fk
from fair_kcenter.util import JsonEncoder

if __name__ == "__main__":

    X=np.array([[-1,0,0],[-2,0,0],[-3,0,0],[0,1,0],[0,2,0],[0,3,0],[0,4,0],[-4,0,0]])
    col=np.array([0,1,2,0,1,2,2,2])

    #Optional: if you want you can specify color ratios which should be achieved 
    color_ratios=np.array([1,1,2])

    n_clusters=2

    #Optional: if you want to know the fairlest labels
    flets=True

    kcenter = fk.Fairkcenter(
        n_clusters=n_clusters,
        flets=flets,
    )

    kcenter.fit(
        X=X,
        col=col,
        col_ratios=color_ratios,
    )

    print(kcenter.labels_)
    print(kcenter.cluster_centers_)

```

<!--
## Citation

If you use this code, please cite [the following paper]():

```
```
-->
