# deep_semi_nmf_torch.py
import torch
from tqdm import trange

# Base ridge; code will auto-increase on failure
BASE_RIDGE = 1e-3
MAX_JITTER_TRIES = 6  # up to BASE_RIDGE * 10**6

def _pos_neg(A):
    Ap = (A.abs() + A) * 0.5
    An = (A.abs() - A) * 0.5
    return Ap, An

@torch.no_grad()
def _solve_right_spd(A, B, lam=BASE_RIDGE):
    """
    Compute X = B @ (A + lam*I)^-1 without forming the inverse.
    A: (k,k) symmetric PSD; B: (m,k).
    Uses Cholesky with jitter escalation; falls back to robust solves.
    """
    assert A.dim() == 2 and A.shape[0] == A.shape[1]
    k = A.shape[0]
    I = torch.eye(k, device=A.device, dtype=A.dtype)

    for t in range(MAX_JITTER_TRIES):
        lam_t = lam * (10 ** t)
        M = A + lam_t * I
        L, info = torch.linalg.cholesky_ex(M, upper=False)
        if int(info.item()) == 0:
            Y = torch.cholesky_solve(B.t(), L, upper=False)  # (k,m)
            return Y.t()
        try:
            X_t = torch.linalg.solve(M, B.t())
            return X_t.t()
        except Exception:
            continue

    # final fallbacks
    try:
        X_t, *_ = torch.linalg.lstsq(M, B.t())
        return X_t.t()
    except Exception:
        Minv = torch.linalg.pinv(M)
        return (B @ Minv)

@torch.no_grad()
def semi_nmf_torch(X, k, iters=200, seed=0, device=None, dtype=torch.float32, desc="Semi-NMF"):
    device = device or X.device
    g = torch.Generator(device=device).manual_seed(seed)
    p, n = X.shape
    Z = torch.randn(p, k, generator=g, device=device, dtype=dtype) * 0.01
    H = torch.rand(k, n, generator=g, device=device, dtype=dtype).clamp_min_(1e-6)

    for _ in trange(iters, desc=desc, leave=False):
        # Z update: Z = (X @ H^T) @ (H H^T + λI)^-1
        HHt = H @ H.t()
        RHS = X @ H.t()
        Z = _solve_right_spd(HHt, RHS, lam=BASE_RIDGE)

        # H multiplicative update
        ZtX = Z.t() @ X
        ZtZ = Z.t() @ Z
        num_p, num_n = _pos_neg(ZtX)
        den_p, den_n = _pos_neg(ZtZ @ H)
        H = H * torch.sqrt((num_p + den_n) / (num_n + den_p + 1e-12))
        H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=0.0)
        H.clamp_min_(1e-12)

    return Z, H

@torch.no_grad()
def deep_pretrain_torch(X, ks, iters=150, seed=42, device=None, dtype=torch.float32):
    Zs, Hs = [], []
    Xcur = X
    for li, k in enumerate(ks, 1):
        Z, H = semi_nmf_torch(Xcur, k, iters=iters, seed=seed+li, device=device, dtype=dtype,
                              desc=f"Pretrain L{li} (k={k})")
        Zs.append(Z); Hs.append(H)
        Xcur = H
    return Zs, Hs

@torch.no_grad()
def _update_H_with_graph_torch(CtX, CtC, H, lam=0.0, W=None, D=None):
    num_p, num_n = _pos_neg(CtX)
    den_pC, den_nC = _pos_neg(CtC @ H)
    num = num_p + den_nC
    den = num_n + den_pC
    if (W is not None) and (D is not None) and (lam > 0):
        num = num + lam * (H @ W)  # W,D are torch.sparse
        den = den + lam * (H @ D)
    H = H * torch.sqrt(num / (den + 1e-12))
    H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=0.0)
    H.clamp_min_(1e-12)
    return H

@torch.no_grad()
def deep_finetune_torch(X, Zs, Hs, graphs=None, lams=None, iters=100):
    """
    Ridge-stable alternating optimization (no explicit inverses).
    """
    if graphs is None: graphs = [None]*len(Hs)
    if lams   is None: lams   = [0.0]*len(Hs)

    pbar = trange(iters, desc="Fine-tune (Deep WSF)", leave=True)
    for _ in pbar:
        H1, H2, H3 = Hs
        Z1, Z2, Z3 = Zs

        # Z1
        H1_tilde = Z2 @ Z3 @ H3
        A = H1_tilde @ H1_tilde.t()
        RHS = X @ H1_tilde.t()
        Z1 = _solve_right_spd(A, RHS, lam=BASE_RIDGE)

        # Z2
        C = Z1
        CtC = C.t() @ C
        T = Z3 @ H3
        TTt = T @ T.t()
        T1 = _solve_right_spd(CtC, (C.t() @ X).t(), lam=BASE_RIDGE).t()
        RHS2 = T1 @ T.t()
        Z2 = _solve_right_spd(TTt, RHS2, lam=BASE_RIDGE)

        # Z3
        C = Z1 @ Z2
        CtC = C.t() @ C
        T1b = _solve_right_spd(CtC, (C.t() @ X).t(), lam=BASE_RIDGE).t()
        HHt = H3 @ H3.t()
        RHS3 = T1b @ H3.t()
        Z3 = _solve_right_spd(HHt, RHS3, lam=BASE_RIDGE)

        Zs = [Z1, Z2, Z3]

        # H3 (no graph)
        C = Z1 @ Z2 @ Z3
        CtX = C.t() @ X
        CtC = C.t() @ C
        H3 = _update_H_with_graph_torch(CtX, CtC, H3, lam=0.0)
        Hs[2] = H3

        # H2 (+ graph)
        C = Z1 @ Z2
        CtX = C.t() @ X
        CtC = C.t() @ C
        W2 = D2 = None
        if graphs[1] is not None:
            W2, D2 = graphs[1]
        H2 = _update_H_with_graph_torch(CtX, CtC, H2, lam=lams[1], W=W2, D=D2)
        Hs[1] = H2

        # H1 (+ graph)
        C = Z1
        CtX = C.t() @ X
        CtC = C.t() @ C
        W1 = D1 = None
        if graphs[0] is not None:
            W1, D1 = graphs[0]
        H1 = _update_H_with_graph_torch(CtX, CtC, H1, lam=lams[0], W=W1, D=D1)
        Hs[0] = H1

    return Zs, Hs
