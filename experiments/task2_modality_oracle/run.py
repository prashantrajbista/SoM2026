"""Recommendation 1 from notes/task2_literature_review.md: before adding any
fusion mechanism, check whether RGB and/or location carry ANY independent
signal for Task 2 -- i.e. does a model with RGB/location as the *only* input
(no CSI at all) beat the persistence baseline (-19.96 dB, from
src/make_submission.py:task2_persistence)?

Ridge regression, image/location -> full target channel, 5-fold CV, same
NMSE convention as src/make_submission.py (global ratio, not per-sample mean).

  python run.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from make_submission import load_mat, split_ri, nmse  # noqa: E402

DATA = ROOT / 'dataset' / 'Task2'
N_FOLDS = 5
SEED = 0


def pool_image(img, block=16):
    """(N,H,W,3) uint8 -> (N, h*w*3) via block-mean pooling, no extra deps."""
    n, h, w, c = img.shape
    h2, w2 = h // block, w // block
    img = img[:, :h2 * block, :w2 * block, :].astype(np.float32)
    img = img.reshape(n, h2, block, w2, block, c).mean(axis=(2, 4))
    return img.reshape(n, -1) / 255.0


def db(x):
    return 10 * np.log10(x)


def cv_nmse(X, y_flat, target_shape):
    """5-fold ridge regression, returns per-fold NMSE (dB) on the true (2,128,64) shape."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for tr, va in kf.split(Xs):
        model = Ridge(alpha=10.0).fit(Xs[tr], y_flat[tr])
        pred = model.predict(Xs[va]).reshape((-1,) + target_shape)
        truth = y_flat[va].reshape((-1,) + target_shape)
        folds.append(db(nmse(truth, pred)))
    return folds


def main():
    target = split_ri(load_mat(DATA / 'X_train.mat', 'X_train'))       # (500,2,128,64)
    prev = split_ri(load_mat(DATA / 'X_train_prev.mat', 'X_train_prev'))
    img = load_mat(DATA / 'imgs_train.mat', 'imgs_train')              # (500,384,216,3)
    loc = load_mat(DATA / 'location_train.mat', 'location_train')      # (500,2)

    shape = target.shape[1:]
    y = target.reshape(len(target), -1)
    img_feat = pool_image(img)

    persistence_folds = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for _, va in kf.split(y):
        persistence_folds.append(db(nmse(target[va], prev[va])))

    mean_floor_folds = []
    for tr, va in kf.split(y):
        pred = np.broadcast_to(target[tr].mean(axis=0, keepdims=True), target[va].shape)
        mean_floor_folds.append(db(nmse(target[va], pred)))

    results = {
        'persistence_db': persistence_folds,
        'train_mean_floor_db': mean_floor_folds,
        'location_only_db': cv_nmse(loc, y, shape),
        'image_only_db': cv_nmse(img_feat, y, shape),
        'image_plus_location_db': cv_nmse(np.hstack([img_feat, loc]), y, shape),
    }

    print(f"{'variant':<22}{'mean dB':>10}  folds")
    for k, v in results.items():
        print(f"{k:<22}{np.mean(v):>10.2f}  {[round(x, 2) for x in v]}")

    beats_persistence = {
        k: bool(np.mean(v) < np.mean(results['persistence_db']))
        for k, v in results.items() if k not in ('persistence_db',)
    }
    print('\nbeats persistence (lower dB is better):', beats_persistence)

    import json
    out = {'mean_db': {k: float(np.mean(v)) for k, v in results.items()},
           'folds_db': results, 'beats_persistence': beats_persistence}
    (Path(__file__).parent / 'results.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
