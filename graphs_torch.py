# graphs_torch.py
import numpy as np
from scipy import sparse
import torch
from tqdm import tqdm

def _scipy_to_torch_coo(M):
    M = M.tocoo()
    idx = np.vstack([M.row, M.col])
    idx_t = torch.from_numpy(idx).long()
    val_t = torch.from_numpy(M.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx_t, val_t, size=M.shape).coalesce()

@torch.no_grad()
def knn_same_label_graph(
    labels,
    k=10,
    features=None,           # numpy (n, d): use Xtr.T
    metric="cosine",         # cosine only in this CUDA impl
    device="cuda",
    qbatch=10_000,           # reduce if VRAM is tight
    verbose=True
):
    """
    CUDA-batched cosine kNN within each label (0/1). Returns CPU torch sparse COO (W, D).
    Loads ONE label block to GPU at a time; queries are processed in batches.
    Auto-reduces qbatch on CUDA OOM.
    """
    assert metric == "cosine", "Only cosine supported in CUDA path."
    y = np.asarray(labels); n = len(y)
    X = np.asarray(features, dtype=np.float32)

    rows, cols = [], []

    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        m = len(idx)
        if m <= 1:
            continue

        Xi = torch.from_numpy(X[idx]).to(device, torch.float32)
        Xi = torch.nn.functional.normalize(Xi, dim=1)

        if verbose:
            print(f"[graphs] label={lab} | block size={m} | qbatch={qbatch}")

        s = 0
        pbar = tqdm(total=m, desc=f"kNN(label={lab})", unit="rows", leave=False)
        while s < m:
            e = min(s + qbatch, m)
            b = e - s
            Q = Xi[s:e]  # (b, d)
            try:
                sims = Q @ Xi.T  # (b, m)
                # exclude self only for the slice diagonal
                ar = torch.arange(b, device=device)
                sims[ar, s + ar] = -1e9

                tk = min(k, m - 1)
                topk_idx = torch.topk(sims, k=tk, dim=1).indices  # (b, tk)
                topk_idx_cpu = topk_idx.cpu().numpy()

                for r, nbrs in enumerate(topk_idx_cpu):
                    gidx = idx[s + r]
                    neigh = [idx[j] for j in nbrs.tolist()]
                    rows.extend([gidx] * len(neigh))
                    cols.extend(neigh)

                s = e
                pbar.update(b)

            except RuntimeError as exc:
                if "CUDA out of memory" in str(exc):
                    torch.cuda.empty_cache()
                    if qbatch <= 1000:
                        pbar.close()
                        del Xi
                        raise RuntimeError(
                            f"CUDA OOM even with qbatch={qbatch}. "
                            f"Lower k or subsample."
                        )
                    qbatch = max(qbatch // 2, 1000)
                    if verbose:
                        print(f"[graphs] OOM: reducing qbatch -> {qbatch}")
                else:
                    pbar.close()
                    del Xi
                    raise
        pbar.close()
        del Xi
        torch.cuda.empty_cache()

    data = np.ones(len(rows), dtype=np.float32)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)  # symmetrize
    deg = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    D  = sparse.diags(deg)

    if verbose:
        nnz = W.nnz
        print(f"[graphs] built W with ~{nnz} edges (~{nnz/n:.1f} avg deg)")

    W_t = _scipy_to_torch_coo(W)  # CPU sparse tensors; move to CUDA later
    D_t = _scipy_to_torch_coo(D)
    return W_t, D_t
