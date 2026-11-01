import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, brier_score_loss


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def all_metrics(y, p, thr=0.5):
    y = np.asarray(y); p = np.asarray(p); yh = (p >= thr).astype(int)
    tp = ((yh == 1) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
    out = dict(acc=accuracy_score(y, yh), bacc=balanced_accuracy_score(y, yh),
               sens=tp / max(1, (y == 1).sum()), spec=tn / max(1, (y == 0).sum()),
               f1=f1_score(y, yh, zero_division=0), brier=brier_score_loss(y, p), ece=ece(y, p))
    out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return out
