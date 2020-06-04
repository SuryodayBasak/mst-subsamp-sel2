import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons,\
                             make_circles,\
                             make_classification,\
                             make_blobs


datasets = [make_blobs(n_samples=2000, centers=2, n_features=2, random_state=0),
            make_moons(n_samples = 1000, noise=0.3, random_state=0),
            make_circles(n_samples = 1000, noise=0.3, factor=0.2, random_state=1)
           ]

for X, y in datasets: 
    y_idx0 = np.where(y == 0)
    y_idx1 = np.where(y == 1)
    plt.scatter(X[y_idx0, 0], X[y_idx0, 1])
    plt.scatter(X[y_idx1, 0], X[y_idx1, 1])
    plt.show()
