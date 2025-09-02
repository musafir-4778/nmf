# deep_semi_nmf.py
import numpy as np
from numpy.linalg import pinv
from scipy import sparse

def _pos_neg(A):
    Ap = (np.abs(A) + A) / 2.0
    An = (np.abs(A) - A) / 2.0
    return Ap, An

def semi_nmf(X, k, iters=200, seed=0):
    """
    Semi-NMF on X (p x n) -> Z (p x k), H (k x n), with H >= 0.
    Updates follow Ding et al.; same family used in Deep (Semi-)NMF paper.
    """
    rng = np.random.default_rng(seed)
    p, n = X.shape
    Z = rng.standard_normal((p, k)).astype(np.float32) * 0.01
    H = np.clip(rng.random((k, n)).astype(np.float32), 1e-6, None)

    for _ in range(iters):
        # Z closed form (least squares)
        Z = (X @ H.T) @ pinv(H @ H.T)
        # H multiplicative
        ZtX = Z.T @ X              # k x n
        ZtZ = Z.T @ Z              # k x k
        num_p, num_n = _pos_neg(ZtX)
        den_p, den_n = _pos_neg(ZtZ @ H)
        H *= np.sqrt((num_p + den_n) / (np.maximum(num_n + den_p, 1e-12)))
        H = np.maximum(H, 1e-12)
    return Z.astype(np.float32), H.astype(np.float32)

def deep_pretrain(X, ks, iters=150, seed=42):
    """
    Layer-wise Semi-NMF pretraining.
    X: p x n
    ks: list like [k1,k2,k3]
    Returns lists Zs, Hs for each layer.
    """
    Zs, Hs = [], []
    Xcur = X
    for i, k in enumerate(ks):
        Z, H = semi_nmf(Xcur, k, iters=iters, seed=seed + i)
        Zs.append(Z); Hs.append(H)
        Xcur = H
    return Zs, Hs

def _update_H_with_graph(CtX, CtC, H, lam, W=None, D=None):
    """
    Graph-regularized multiplicative update for H with Laplacian L = D - W.
    Uses standard GNMF-style split:
      H *= sqrt( ( (CtX)^+ + (CtC)^- H + lam * H W ) / ( (CtX)^- + (CtC)^+ H + lam * H D ) )
    """
    num_p, num_n = _pos_neg(CtX)
    den_pC, den_nC = _pos_neg(CtC @ H)

    num = num_p + den_nC
    den = num_n + den_pC
    if W is not None and D is not None and lam > 0:
        # right-multiply by sparse W/D
        num = num + lam * (H @ W)
        den = den + lam * (H @ D)

    H *= np.sqrt(num / np.maximum(den, 1e-12))
    H = np.maximum(H, 1e-12)
    return H

def deep_finetune(X, Zs, Hs, graphs=None, lams=None, iters=100):
    """
    Joint alternating optimization over 3 layers (generalize as needed).
    graphs: list of (W,D) or None for each layer's H (same length as Hs)
    lams: list of lambdas per layer (e.g., [lam1, lam2, 0.0])
    """
    if graphs is None: graphs = [None]*len(Hs)
    if lams   is None: lams   = [0.0]*len(Hs)

    for _ in range(iters):
        H1, H2, H3 = Hs
        Z1, Z2, Z3 = Zs

        # Z updates (closed-form with pseudo-inverses)
        H1_tilde = Z2 @ Z3 @ H3
        Z1 = (X @ H1_tilde.T) @ pinv(H1_tilde @ H1_tilde.T)

        C = Z1
        H2_tilde = Z3 @ H3
        Z2 = pinv(C) @ X @ H2_tilde.T @ pinv(H2_tilde @ H2_tilde.T)

        C = Z1 @ Z2
        Z3 = pinv(C) @ X @ H3.T @ pinv(H3 @ H3.T)
        Zs = [Z1, Z2, Z3]

        # H3 (no graph reg by default)
        C = Z1 @ Z2 @ Z3
        CtX = C.T @ X
        CtC = C.T @ C
        H3 = _update_H_with_graph(CtX, CtC, H3, lam=0.0)
        Hs[2] = H3

        # H2 with possible graph reg
        C = Z1 @ Z2
        CtX = C.T @ X
        CtC = C.T @ C
        W2 = D2 = None
        if graphs[1] is not None:
            W2, D2 = graphs[1]
        H2 = _update_H_with_graph(CtX, CtC, H2, lam=lams[1], W=W2, D=D2)
        Hs[1] = H2

        # H1 with possible graph reg
        C = Z1
        CtX = C.T @ X
        CtC = C.T @ C
        W1 = D1 = None
        if graphs[0] is not None:
            W1, D1 = graphs[0]
        H1 = _update_H_with_graph(CtX, CtC, H1, lam=lams[0], W=W1, D=D1)
        Hs[0] = H1

    return Zs, Hs
