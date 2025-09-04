# preprocess_celeba.py  (robust paths + autodetect)
import os, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob

# ----- base dirs -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # where this .py lives
DATA_DIR = os.path.join(BASE_DIR, "celeba")             # your folder in screenshot

# candidate image roots (some Kaggle zips nest a second "img_align_celeba")
CAND_IMG_DIRS = [
    os.path.join(DATA_DIR, "img_align_celeba"),
    os.path.join(DATA_DIR, "img_align_celeba", "img_align_celeba"),
    DATA_DIR  # last resort
]

# find a directory that actually has celebA files
IMG_DIR = None
IMG_EXT = ".jpg"
for d in CAND_IMG_DIRS:
    if os.path.isdir(d):
        # detect extension
        jpgs = glob(os.path.join(d, "*.jpg"))
        pngs = glob(os.path.join(d, "*.png"))
        if jpgs:
            IMG_DIR, IMG_EXT = d, ".jpg"
            break
        if pngs:
            IMG_DIR, IMG_EXT = d, ".png"
            break

if IMG_DIR is None:
    raise FileNotFoundError("Could not locate img_align_celeba directory with .jpg/.png files.")

ATTR_CSV  = os.path.join(DATA_DIR, "list_attr_celeba.csv")
SPLIT_CSV = os.path.join(DATA_DIR, "list_eval_partition.csv")

# sanity checks
for p in [DATA_DIR, IMG_DIR, ATTR_CSV, SPLIT_CSV]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required path: {p}")

OUT_DIR = os.path.join(DATA_DIR, "preproc_64x64_gray_optB")
os.makedirs(OUT_DIR, exist_ok=True)

SIZE = (64, 64)
TO_GRAY = True

# --- load splits ---
sp = pd.read_csv(SPLIT_CSV)
# must have columns: image_id, partition
if "image_id" not in sp.columns or "partition" not in sp.columns:
    raise ValueError(f"{SPLIT_CSV} must have columns ['image_id','partition'], got {sp.columns.tolist()}")
part_series = sp.set_index("image_id")["partition"]

# --- load attributes ---
attr = pd.read_csv(ATTR_CSV).set_index("image_id")
# convert +1/-1 -> 1/0 if needed
if set(np.unique(attr.iloc[:,0])).issuperset({-1, 1}):
    attr = (attr + 1) // 2

for col in ["Young", "Smiling"]:
    assert col in attr.columns, f"Missing attribute: {col}"

def collect(names):
    X_list, young, smile = [], [], []
    for fn in tqdm(names, desc="images"):
        # if names contain .jpg but real files are .png (or vice versa), swap
        fbase, _ = os.path.splitext(fn)
        path = os.path.join(IMG_DIR, fbase + IMG_EXT)
        if not os.path.exists(path):
            # try as-is (some CSVs already have correct ext)
            path_alt = os.path.join(IMG_DIR, fn)
            if os.path.exists(path_alt):
                path = path_alt
            else:
                raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path)
        if TO_GRAY: img = img.convert("L")
        img = img.resize(SIZE, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        X_list.append(arr.flatten() if TO_GRAY else arr.transpose(2,0,1).reshape(-1))

        rec = attr.loc[fn] if fn in attr.index else attr.loc[fbase + IMG_EXT]
        young.append(int(rec["Young"]))
        smile.append(int(rec["Smiling"]))
    X = np.stack(X_list, axis=1)  # p x n
    return X, np.array(young), np.array(smile)

def split_filenames(pval):
    names = part_series[part_series == pval].index.tolist()
    names.sort()
    return names

limit_train = limit_val = limit_test = None  # set small ints to test first (e.g., 5000)

for split_name, pval, limit in [("train",0,limit_train),("val",1,limit_val),("test",2,limit_test)]:
    fnames = split_filenames(pval)
    if limit:
        fnames = fnames[:limit]
    X, Y_young, Y_smile = collect(fnames)
    np.save(os.path.join(OUT_DIR, f"X_{split_name}.npy"), X)
    np.save(os.path.join(OUT_DIR, f"young_{split_name}.npy"), Y_young)
    np.save(os.path.join(OUT_DIR, f"smiling_{split_name}.npy"), Y_smile)
    print(split_name, "->", X.shape, Y_young.mean(), Y_smile.mean())





