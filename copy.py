SIZE, TO_GRAY = (64, 64), True

def collect(names):
    X_list, young, smile = [], [], []
    for fn in tqdm(names, desc="images"):
        fbase, _ = os.path.splitext(fn)
        path = os.path.join(IMG_DIR, fbase + IMG_EXT)
        if not os.path.exists(path): path = os.path.join(IMG_DIR, fn)
        img = Image.open(path)
        if TO_GRAY: img = img.convert("L")
        arr = np.asarray(img.resize(SIZE, Image.BILINEAR), np.float32) / 255.0
        X_list.append(arr.flatten() if TO_GRAY else arr.transpose(2,0,1).reshape(-1))
        rec = attr.loc[fn] if fn in attr.index else attr.loc[fbase + IMG_EXT]
        young.append(int(rec["Young"])); smile.append(int(rec["Smiling"]))
    return np.stack(X_list, axis=1), np.array(young), np.array(smile)
