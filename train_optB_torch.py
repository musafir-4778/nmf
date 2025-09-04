# train_optB_torch.py
import os, time, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from graphs_torch import knn_same_label_graph
from deep_semi_nmf_torch import deep_pretrain_torch, deep_finetune_torch, _pos_neg

# ---------------- settings ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32
DATA_DIR = os.path.join("celeba", "preproc_64x64_gray_optB")

KS        = [512, 256, 128]   # layer widths
PRE_ITERS = 120
FT_ITERS  = 60
LAMS      = [1e-3, 1e-3, 0.0] # H1 (Young), H2 (Smiling), H3
KNN_K     = 10
KNN_QBATCH= 10_000            # auto-reduces on OOM
CLUST_K   = 500

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# --------------- helpers ------------------
def clean_np(A, clip=None):
    A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=0.0)
    if clip is not None:
        hi = np.percentile(A, clip)
        A = np.clip(A, 0.0, hi)
    return A

# --------------- load data ----------------
Xtr = np.load(os.path.join(DATA_DIR, "X_train.npy"))       # (p, n)
Ytr_young = np.load(os.path.join(DATA_DIR, "young_train.npy"))
Ytr_smile = np.load(os.path.join(DATA_DIR, "smiling_train.npy"))
Xva = np.load(os.path.join(DATA_DIR, "X_val.npy"))
Yva_young = np.load(os.path.join(DATA_DIR, "young_val.npy"))
Yva_smile = np.load(os.path.join(DATA_DIR, "smiling_val.npy"))

print(f"Train X: {Xtr.shape} | Val X: {Xva.shape} | Device: {DEVICE}")

# tensors
Xtr_t = torch.from_numpy(Xtr).to(DEVICE, DTYPE)
Xva_t = torch.from_numpy(Xva).to(DEVICE, DTYPE)

# ----------- build graphs -----------------
t0 = time.time()
W1_cpu, D1_cpu = knn_same_label_graph(
    Ytr_young, k=KNN_K, features=Xtr.T, metric="cosine",
    device=DEVICE, qbatch=KNN_QBATCH, verbose=True
)
W2_cpu, D2_cpu = knn_same_label_graph(
    Ytr_smile, k=KNN_K, features=Xtr.T, metric="cosine",
    device=DEVICE, qbatch=KNN_QBATCH, verbose=True
)
W1, D1 = W1_cpu.to(DEVICE), D1_cpu.to(DEVICE)
W2, D2 = W2_cpu.to(DEVICE), D2_cpu.to(DEVICE)
graphs = [(W1, D1), (W2, D2), None]
print(f"[time] graph build: {time.time()-t0:.1f}s")

# --------------- pretrain -----------------
t1 = time.time()
Zs, Hs = deep_pretrain_torch(Xtr_t, KS, iters=PRE_ITERS, device=DEVICE, dtype=DTYPE)
print(f"[time] pretrain: {time.time()-t1:.1f}s")

# --------------- fine-tune ----------------
t2 = time.time()
Zs, Hs = deep_finetune_torch(Xtr_t, Zs, Hs, graphs=graphs, lams=LAMS, iters=FT_ITERS)
print(f"[time] fine-tune: {time.time()-t2:.1f}s")

# -------------- collect H(tr) -------------
H1_tr = clean_np(Hs[0].T.detach().cpu().numpy(), clip=99.9)
H2_tr = clean_np(Hs[1].T.detach().cpu().numpy(), clip=99.9)
H3_tr = clean_np(Hs[2].T.detach().cpu().numpy(), clip=99.9)

# ---------- simple val projection ----------
@torch.no_grad()
def project_layers_val(X, Zs, iters=40, device=DEVICE, dtype=DTYPE):
    Z1, Z2, Z3 = Zs
    # H1
    H1 = torch.clamp(torch.rand(Z1.shape[1], X.shape[1], device=device, dtype=dtype), min=1e-6)
    for _ in tqdm(range(iters), desc="VAL proj H1", leave=False):
        ZtX = Z1.t() @ X
        ZtZ = Z1.t() @ Z1
        num_p, num_n = _pos_neg(ZtX); den_p, den_n = _pos_neg(ZtZ @ H1)
        H1 = H1 * torch.sqrt((num_p + den_n) / (num_n + den_p + 1e-12))
        H1 = torch.nan_to_num(H1, nan=0.0, posinf=1e6, neginf=0.0)
        H1.clamp_min_(1e-12)
    # H2
    H2 = torch.clamp(torch.rand(Z2.shape[1], X.shape[1], device=device, dtype=dtype), min=1e-6)
    for _ in tqdm(range(iters//2), desc="VAL proj H2", leave=False):
        ZtX = Z2.t() @ H1; ZtZ = Z2.t() @ Z2
        num_p, num_n = _pos_neg(ZtX); den_p, den_n = _pos_neg(ZtZ @ H2)
        H2 = H2 * torch.sqrt((num_p + den_n) / (num_n + den_p + 1e-12))
        H2 = torch.nan_to_num(H2, nan=0.0, posinf=1e6, neginf=0.0)
        H2.clamp_min_(1e-12)
    # H3
    H3 = torch.clamp(torch.rand(Z3.shape[1], X.shape[1], device=device, dtype=dtype), min=1e-6)
    for _ in tqdm(range(iters//2), desc="VAL proj H3", leave=False):
        ZtX = Z3.t() @ H2; ZtZ = Z3.t() @ Z3
        num_p, num_n = _pos_neg(ZtX); den_p, den_n = _pos_neg(ZtZ @ H3)
        H3 = H3 * torch.sqrt((num_p + den_n) / (num_n + den_p + 1e-12))
        H3 = torch.nan_to_num(H3, nan=0.0, posinf=1e6, neginf=0.0)
        H3.clamp_min_(1e-12)
    return H1, H2, H3

t3 = time.time()
H1_va_t, H2_va_t, H3_va_t = project_layers_val(Xva_t, Zs, iters=40)
H1_va = clean_np(H1_va_t.T.cpu().numpy(), clip=99.9)
H2_va = clean_np(H2_va_t.T.cpu().numpy(), clip=99.9)
print(f"[time] val projection: {time.time()-t3:.1f}s")

# --------- attribute accuracy ----------
def attr_acc(H_tr, y_tr, H_va, y_va, sample=40_000):
    clf = LogisticRegression(max_iter=1000)
    if H_tr.shape[0] > sample:
        idx = np.random.RandomState(0).choice(H_tr.shape[0], size=sample, replace=False)
        clf.fit(H_tr[idx], y_tr[idx])
    else:
        clf.fit(H_tr, y_tr)
    return accuracy_score(y_va, clf.predict(H_va))

acc_young = attr_acc(H1_tr, Ytr_young, H1_va, Yva_young)
acc_smil  = attr_acc(H2_tr, Ytr_smile, H2_va, Yva_smile)
print(f"[VAL] Young@H1 acc: {acc_young:.3f} | Smiling@H2 acc: {acc_smil:.3f}")

# -------------- H3 clustering -------------
t4 = time.time()
km = KMeans(n_clusters=CLUST_K, n_init=10, random_state=0).fit(H3_tr)
sil = silhouette_score(H3_tr, km.labels_, sample_size=min(50_000, H3_tr.shape[0]), random_state=0)
print(f"[H3] Clustering: K={CLUST_K} | silhouette={sil:.3f} | time={time.time()-t4:.1f}s")

print("[DONE]")
print("\n[WATERMARK] Implementation by Lakshya Yadav (@lakshyayadav)")


def project_layers_val(X, Zs, iters=40, device=DEVICE):
    Z1,Z2,Z3 = Zs
    # multiplicative NNLS-ish per layer (H≥0) with NaN guards
    def proj(Z, X, iters):
        H = torch.rand(Z.shape[1], X.shape[1], device=device).clamp_min_(1e-6)
        for _ in range(iters):
            ZtX, ZtZ = Z.t() @ X, Z.t() @ Z
            np_, nn_ = _pos_neg(ZtX); dp, dn = _pos_neg(ZtZ @ H)
            H = H * torch.sqrt((np_ + dn) / (nn_ + dp + 1e-12))
            H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=0.0).clamp_min_(1e-12)
        return H
    H1 = proj(Z1, X, iters); H2 = proj(Z2, H1, iters//2); H3 = proj(Z3, H2, iters//2)
    return H1, H2