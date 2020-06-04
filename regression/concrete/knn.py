import numpy as np
from numpy import linalg as LA

class KNNSubsampler:
    def __init__(self, X, y, k):
        self.X = X[:, :-1]
        self.X_ids = X[:, -1]
        self.y = y
        self.k = k
        self.nbhs = None
        self.nbhVars = None

    def find_all_neighbors(self, X_test):
        neighbors = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(LA.norm(x - self.X[i])) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            neighbors.append(self.X_ids[sorted(k_idx)])
        neighbors = np.array(neighbors)
        neighbors = np.unique(neighbors, axis = 0)
        self.nbhs = neighbors

    def find_neighborhood_std(self):
        variances = []
        for hood in self.nbhs:
            hood_lbls = self.y[np.isin(self.X_ids, hood)]

            var = np.var(hood_lbls, ddof = 1)
            variances.append(1 - var)
        self.nbhVars = np.array(variances)

    def reweight(self):
        weight_updates = []
       
        for i in range(len(self.X_ids)):
            pt_id = self.X_ids[i]
            sample_weight = 0
            n_hoods = 0
            for j in range(len(self.nbhs)):
                hood = self.nbhs[j]
                if (np.isin(pt_id, hood)):
                    sample_weight += self.nbhVars[j]
                    n_hoods += 1

            if n_hoods == 0:
                sample_weight = -1
            else:
                sample_weight /= n_hoods

            weight_updates.append(sample_weight)
        
        # Normalize.
        min_w = min(weight_updates)
        max_w = max(weight_updates)
        weight_updates = (weight_updates - min_w)/(max_w - min_w)
        weight_updates = np.array(weight_updates) / len(weight_updates)
        return weight_updates
