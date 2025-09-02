# graphs.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

def knn_same_label_graph(labels, k=10, features=None, metric="euclidean"):
    """
    Build a symmetric k-NN adjacency among points that share the SAME label (0/1).
    Optionally use 'features' (n x d) to compute neighbors; if None, use index space.

    Returns: (W, D, L) as scipy.sparse CSR matrices, shape (n, n)
    """
    y = np.asarray(labels)
    n = len(y)

    def build_block(idx, feats=None):
        if len(idx) <= 1:
            return sparse.csr_matrix((n, n))
        if feats is None:
            # simple index features (works but weaker); for speed when nothing else is provided
            feats = np.arange(len(idx), dtype=float).reshape(-1, 1)
        nn = NearestNeighbors(n_neighbors=min(k+1, len(idx)), metric=metric)
        nn.fit(feats)
        nbrs = nn.kneighbors(feats, return_distance=False)
        rows, cols = [], []
        for r, neigh in enumerate(nbrs):
            # drop self
            neigh = [idx[j] for j in neigh if j != r]
            rows.extend([idx[r]] * len(neigh))
            cols.extend(neigh)
        data = np.ones(len(rows), dtype=np.float32)
        Wb = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        return Wb

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    feats0 = None if features is None else features[idx0]
    feats1 = None if features is None else features[idx1]

    W0 = build_block(idx0, feats0)
    W1 = build_block(idx1, feats1)
    W = (W0 + W1).maximum((W0 + W1).T)  # symmetrize

    deg = np.asarray(W.sum(axis=1)).ravel()
    D = sparse.diags(deg)
    L = D - W
    return W.tocsr(), D.tocsr(), L.tocsr()
