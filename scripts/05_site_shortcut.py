"""Site-shortcut diagnostics (container-side, fast models only).
(1) site-prior baseline: predict ASD from the training-fold ASD rate of the test subject's site;
(2) site decodability of each representation (17-class balanced accuracy, 10-fold);
(3) LR/RF on PCA-100 with vs without label-free per-site standardisation (same folds as E1 repeat 0)."""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from braintab.compress import make_reducer
from braintab.harmonize import site_standardize
d = np.load(os.path.join(ROOT, "data/abide_features.npz")); meta = pd.read_csv(os.path.join(ROOT, "data/abide_meta.csv"))
q = meta.qc_pass.values; X = d["fisher_z"][q]; y = meta.y.values[q]; site = meta.site.values[q].astype(str)
strat = np.array([f"{s}_{c}" for s, c in zip(site, y)])
folds = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2026).split(X, strat))
out = {}
# (1) site prior
aucs, accs = [], []
for tr, te in folds:
    rate = pd.Series(y[tr]).groupby(site[tr]).mean()
    p = np.array([rate.get(s, y[tr].mean()) for s in site[te]])
    aucs.append(roc_auc_score(y[te], p)); accs.append(((p > 0.5) == y[te]).mean())
out["site_prior_auc_mean"], out["site_prior_auc_sd"] = float(np.mean(aucs)), float(np.std(aucs))
out["site_prior_acc_mean"] = float(np.mean(accs))
# (2) site decodability (repeat-0 folds)
sd = {}
for rep in ["pca20", "pca100", "plsauto"]:
    b = []
    for tr, te in folds[:10]:
        r = make_reducer(rep); A = r.fit_transform(X[tr], y[tr]); B = r.transform(X[te])
        c = LogisticRegression(max_iter=3000, C=1.0).fit(A, site[tr]); b.append(balanced_accuracy_score(site[te], c.predict(B)))
    sd[rep] = (float(np.mean(b)), float(np.std(b)))
out["site_decoding_bacc"] = sd; out["site_chance"] = 1 / 17
# (3) LR / RF on PCA-100 with label-free per-site standardisation
from sklearn.ensemble import RandomForestClassifier
Xs = site_standardize(X, site); res = {}
for nm, XX in [("none", X), ("sitez", Xs)]:
    for mn, mk in [("LR", lambda: LogisticRegression(max_iter=5000)), ("RF", lambda: RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=0))]:
        a = []
        for tr, te in folds[:10]:
            r = make_reducer("pca100"); A = r.fit_transform(XX[tr]); B = r.transform(XX[te])
            a.append(roc_auc_score(y[te], mk().fit(A, y[tr]).predict_proba(B)[:, 1]))
        res[f"{mn}@pca100|{nm}"] = (float(np.mean(a)), float(np.std(a)))
out["pca100_sitez_auc"] = res
json.dump(out, open(os.path.join(ROOT, "results", "site_shortcut.json"), "w"), indent=1); print(json.dumps(out, indent=1))
