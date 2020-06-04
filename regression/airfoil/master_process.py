import numpy as np
import data_api as da
import multiprocessing
import time
from knn import KNNSubsampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as skmse
from sklearn.metrics import accuracy_score as acc
from sklearn import neighbors
import sys

# Load data.
data = da.Airfoil()
X, y = data.Data()
_, nFeats = np.shape(X)

# Values of parameter k to iterate over.
K_VALS = [3, 5, 7, 9, 11, 13, 15]


"""
Some important variables.
"""
r = 30
N = 2000
k = 5
A = 1
n_iters = 30 
k_vals = [3, 5, 7, 9, 11, 13, 15]
perc = [0.05, 0.1, 0.25, 0.5]
n_samples = len(X)

# Repeat each trial 10 times.
for i in range (0, 1):
    x_train, x_test, y_train, y_test = train_test_split(X, y,\
                                                        test_size=0.2)

    x_train, x_verif, y_train, y_verif = train_test_split(x_train,\
                                                          y_train,\
                                                          test_size=0.33)
   

    # Our method.
    for k in K_VALS:
        for p in perc:
            ids = np.array([[i for i in range(len(x_train))]])
            x_train_p = np.append(x_train, ids.T, axis = 1)
            weights = np.ones(len(x_train))

            # Iterative procedure
            for resample in range(0, n_iters):
                iter_idx = np.random.choice(len(x_train_p),\
                                            size = int(p * n_samples),\
                                            replace = False,\
                                            p = weights/weights.sum())
                X_iter = x_train_p[iter_idx]
                y_iter = y_train[iter_idx]
                X_ids = X_iter[:, -1].astype(int)

                # Generate test sets for finding neighborhoods.
    
                sampler = KNNSubsampler(X_iter, y_iter, k)
                sampler.find_all_neighbors(x_verif)
                sampler.find_neighborhood_std()        
                reweights = sampler.reweight()

                weights[X_ids] += reweights * A

            # Results with optimized subsampling
            best_idx = np.random.choice(len(x_train_p),\
                                        size = int(p * n_samples),\
                                        replace = False,\
                                        p = weights/weights.sum())

            X_train_p = x_train_p[best_idx]
            y_train_p = y_train[best_idx]

            clf = neighbors.KNeighborsRegressor(k)
            clf.fit(X_train_p[:,:-1], y_train_p)
            y_pred = clf.predict(x_test)
            print("p-sampling,",k,",",p,",",skmse(y_pred, y_test))

            # Uniform subsampling.

            best_idx = np.random.choice(len(x_train_p),\
                                        size = int(p * n_samples),\
                                        replace = False)

            X_train_p = x_train_p[best_idx]
            y_train_p = y_train[best_idx]

            clf = neighbors.KNeighborsRegressor(k)
            clf.fit(X_train_p[:,:-1], y_train_p)
            y_pred = clf.predict(x_test)
            print("u-sampling,",",",k,",",p,",",skmse(y_pred, y_test))

