# train_optB.py
import os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from graphs import knn_same_label_graph
from deep_semi_nmf import deep_pretrain, deep_finetune

DATA_DIR = os.path.join("celeba", "preproc_64x64_gray_optB")

# --- Load data (use train for learning; val for quick eval) ---
Xtr = np.load(os.path.join(DATA_DIR, "X_train.npy"))       # (4096, n)
Ytr_young = np.load(os.path.join(DATA_DIR, "young_train.npy"))
Ytr_smile = np.load(os.path.join(DATA_DIR, "smiling_train.npy"))

Xva = np.load(os.path.join(DATA_DIR, "X_val.npy"))
Yva_young = np.load(os.path.join(DATA_DIR, "young_val.npy"))
Yva_smile = np.load(os.path.join(DATA_DIR, "smiling_val.npy"))

# (optional) subsample for a fast first run
# idx = np.random.RandomState(0).choice(Xtr.shape[1], size=20000, replace=False)
# Xtr, Ytr_young, Ytr_smile = Xtr[:, idx], Ytr_young[idx], Ytr_smile[idx]

print("Train X:", Xtr.shape, "| Val X:", Xva.shape)

# --- Build Laplacians (Young for H1, Smiling for H2) ---
# For neighbor features, we can use raw pixels (transpose to n x p)
W1, D1, _ = knn_same_label_graph(Ytr_young, k=10, features=Xtr.T, metric="cosine")
W2, D2, _ = knn_same_label_graph(Ytr_smile, k=10, features=Xtr.T, metric="cosine")
graphs = [(W1, D1), (W2, D2), None]

# --- Deep sizes and training params ---
ks = [512, 256, 128]       # layer widths (tune as needed)
pre_iters = 120
ft_iters  = 60
lams = [1e-3, 1e-3, 0.0]   # reg strengths for H1 (Young), H2 (Smiling), H3

# --- Pretrain & Fine-tune ---
Zs, Hs = deep_pretrain(Xtr, ks, iters=pre_iters)
Zs, Hs = deep_finetune(Xtr, Zs, Hs, graphs=graphs, lams=lams, iters=ft_iters)

H1_tr, H2_tr, H3_tr = Hs[0].T, Hs[1].T, Hs[2].T   # shapes: (n x k_i)

# --- Evaluate attribute alignment on VAL set ---
# Project val through learned Zs (fast: solve nonneg H with a few Semi-NMF iters at each layer)
from deep_semi_nmf import semi_nmf

# quick projection: treat previous H as "X" and run a few iters keeping Z fixed
def project_through_layers(X, Zs, iters=60):
    # layer-1
    Z1 = Zs[0]
    # initialize H1 by closed form then a few Semi-NMF updates with Z fixed
    H1 = np.clip((Z1.T @ Z1) @ np.linalg.pinv(Z1.T @ Z1) @ (Z1.T @ X), 1e-6, None)  # crude init
    # refine multiplicatively using Semi-NMF rule with fixed Z1
    for _ in range(iters):
        ZtX = Z1.T @ X
        ZtZ = Z1.T @ Z1
        num_p, num_n = _pos_neg(ZtX); den_p, den_n = _pos_neg(ZtZ @ H1)
        H1 *= np.sqrt((num_p + den_n) / (np.maximum(num_n + den_p, 1e-12))); H1=np.maximum(H1,1e-12)

    # layer-2
    Z2 = Zs[1]
    H2 = np.clip((Z2.T @ Z2) @ np.linalg.pinv(Z2.T @ Z2) @ (Z2.T @ H1), 1e-6, None)
    for _ in range(iters//2):
        ZtX = Z2.T @ H1; ZtZ = Z2.T @ Z2
        num_p, num_n = _pos_neg(ZtX); den_p, den_n = _pos_neg(ZtZ @ H2)
        H2 *= np.sqrt((num_p + den_n) / (np.maximum(num_n + den_p, 1e-12))); H2=np.maximum(H2,1e-12)

    # layer-3
    Z3 = Zs[2]
    H3 = np.clip((Z3.T @ Z3) @ np.linalg.pinv(Z3.T @ Z3) @ (Z3.T @ H2), 1e-6, None)
    for _ in range(iters//2):
        ZtX = Z3.T @ H2; ZtZ = Z3.T @ Z3
        num_p, num_n = _pos_neg(ZtX); den_p, den_n = _pos_neg(ZtZ @ H3)
        H3 *= np.sqrt((num_p + den_n) / (np.maximum(num_n + den_p, 1e-12))); H3=np.maximum(H3,1e-12)
    return H1, H2, H3

# import local helper from module namespace
from deep_semi_nmf import _pos_neg
H1_va, H2_va, H3_va = project_through_layers(Xva, Zs)

# --- Simple attribute prediction (how aligned are H1/H2?) ---
def attr_acc(H, y):
    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    # train on a slice of train reps to keep it quick
    ridx = np.random.RandomState(0).choice(H1_tr.shape[0], size=min(40000, H1_tr.shape[0]), replace=False)
    clf.fit(H[ridx], y[ridx])
    return accuracy_score(y, clf.predict(H))

acc_young = attr_acc(H1_tr, Ytr_young)
acc_smil  = attr_acc(H2_tr, Ytr_smile)
print(f"[Train] Young@H1 acc: {acc_young:.3f} | Smiling@H2 acc: {acc_smil:.3f}")

# validate
acc_young_val = accuracy_score(Yva_young, LogisticRegression(max_iter=1000).fit(H1_tr, Ytr_young).predict(H1_va.T))
acc_smil_val  = accuracy_score(Yva_smile,  LogisticRegression(max_iter=1000).fit(H2_tr, Ytr_smile ).predict(H2_va.T))
print(f"[Val]   Young@H1 acc: {acc_young_val:.3f} | Smiling@H2 acc: {acc_smil_val:.3f}")

# --- Unsupervised identity-style clustering on H3 (no ID labels needed) ---
k = 500  # pick a cluster count to probe structure
km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(H3_tr)
sil = silhouette_score(H3_tr, km.labels_, sample_size=min(50000, H3_tr.shape[0]), random_state=0)
print(f"H3 clustering: K={k} | train silhouette={sil:.3f}")
