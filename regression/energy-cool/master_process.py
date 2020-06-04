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
data = da.EnergyCool()
X, y = data.Data()
_, nFeats = np.shape(X)

# Values of parameter k to iterate over.
#K_VALS = [3, 5, 7, 9, 11, 13, 15]
K_VALS = [15]

"""
Some important variables.
"""
#r = 30
#k = 5
A = 1
n_iters = 1000
k_vals = [3, 5, 7, 9, 11, 13, 15]
perc = [0.05, 0.1, 0.25]
n_samples = len(X)

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
        np.savetxt('learned_weights/'+str(k)+'-'+str(p).replace('.','_')+'.csv',\
                   weights, delimiter=',')
