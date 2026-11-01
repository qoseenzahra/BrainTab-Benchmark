"""Comment 24: pooled out-of-site LOSO performance (all held-out-site predictions concatenated)."""
import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
R = sys.argv[1]; OUT = sys.argv[2]
from scipy.stats import rankdata
def fauc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def fbacc(y, p):
    h = p >= .5
    return 0.5 * (h[y == 1].mean() + (~h[y == 0]).mean())
def boot(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); a, b = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        a.append(fauc(y[i], p[i])); b.append(fbacc(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5]), np.percentile(b, [2.5, 97.5])
rows = []
for cohort, files in [("ABIDE-I", [f"{R}/cloud/loso_preds.csv", f"{R}/pc_w3/loso_preds.csv", f"{R}/pc_w5/loso_preds.csv", f"{R}/pc_w7/loso_preds.csv", f"{R}/pc_w2b/loso_preds.csv"]),
                      ("ADHD-200 (5 mixed sites)", [f"{R}/adhd/adhd_w3/adhdfull_loso_preds.csv"])]:
    d = pd.concat([pd.read_csv(f) for f in files])
    d["site"] = d.key.str.split("|").str[0]; d["cfg"] = d.key.str.split("|", n=1).str[1]
    for cfg, g in d.groupby("cfg"):
        rep, harm, model = cfg.split("|")
        y, p = g.y.values, g.p.values
        ca, cb = boot(y, p)
        rows.append(dict(cohort=cohort, rep=rep, harm=harm, model=model, n=len(g), n_sites=g.site.nunique(),
                         auc=roc_auc_score(y, p), auc_lo=ca[0], auc_hi=ca[1],
                         bacc=balanced_accuracy_score(y, p >= .5), bacc_lo=cb[0], bacc_hi=cb[1]))
out = pd.DataFrame(rows); out.to_csv(OUT, index=False)
pd.set_option("display.width", 200); print(out.round(3).to_string(index=False))
