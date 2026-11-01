"""Model zoo. Every model exposes fit(X, y) / predict_proba(X)[:, 1].
Optional dependencies (xgboost, tabicl, torch/tabm) are imported lazily so the
benchmark still runs (with fewer models) where they are unavailable."""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _set_threads():
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("BT_THREADS", "4")))
    except Exception:
        pass


_set_threads()


def lr_full(seed=0):
    # L2 logistic regression on all 19,900 edges; C chosen by inner 3-fold CV (AUC)
    return make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], cv=3, scoring="roc_auc", max_iter=3000, n_jobs=1))


def lr(seed=0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def rf(seed=0):
    return RandomForestClassifier(500, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=seed)


def xgb(seed=0):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
                         n_jobs=int(os.environ.get("BT_THREADS", "4")))


def hgb(seed=0):  # container stand-in for XGBoost during development
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed)


class TabICLWrap:
    """Thin wrapper around the official TabICL implementation (no architecture change)."""
    CKPT = {"v1": "tabicl-classifier-v1-20250208.ckpt", "v2": "tabicl-classifier-v2-20260212.ckpt"}

    def __init__(self, version="v1", seed=0, n_estimators=None):
        self.version, self.seed = version, seed
        self.n_estimators = n_estimators or int(os.environ.get("BT_TABICL_NEST", "32"))

    def fit(self, X, y):
        from tabicl import TabICLClassifier
        self.m_ = TabICLClassifier(checkpoint_version=self.CKPT[self.version], device="cpu",
                                   n_estimators=self.n_estimators, random_state=self.seed)
        self.m_.fit(np.asarray(X, np.float32), np.asarray(y))
        return self

    def predict_proba(self, X):
        P = self.m_.predict_proba(np.asarray(X, np.float32))
        cls = list(self.m_.classes_)
        p1 = P[:, cls.index(1)]
        return np.column_stack([1 - p1, p1])


class TabMWrap:
    """TabM (Gorishniy et al., ICLR 2025) via the official `tabm` package: k=32 efficient
    ensemble MLP, AdamW (lr 1e-3, wd 3e-4), batch 128, <=100 epochs, early stopping
    (patience 10) on a 15% stratified validation split of the context."""
    def __init__(self, seed=0, k=32, epochs=100, patience=10, batch=128):
        self.seed, self.k, self.epochs, self.patience, self.batch = seed, k, epochs, patience, batch

    def fit(self, X, y):
        import torch, tabm
        from sklearn.model_selection import train_test_split
        torch.manual_seed(self.seed)
        self.sc_ = StandardScaler().fit(X)
        Xs = self.sc_.transform(X).astype(np.float32)
        tr, va = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=self.seed)
        self.net_ = tabm.TabM.make(n_num_features=Xs.shape[1], cat_cardinalities=[], d_out=1, k=self.k)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=1e-3, weight_decay=3e-4)
        Xt, yt = torch.tensor(Xs), torch.tensor(y, dtype=torch.float32)
        best, bad, state = 1e9, 0, None
        lossf = torch.nn.BCEWithLogitsLoss()
        for ep in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(tr))
            for b in range(0, len(tr), self.batch):
                idx = torch.as_tensor(tr)[perm[b:b + self.batch]]
                out = self.net_(Xt[idx]).squeeze(-1)            # (B, k)
                loss = lossf(out, yt[idx][:, None].expand_as(out))
                opt.zero_grad(); loss.backward(); opt.step()
            self.net_.eval()
            with torch.no_grad():
                o = self.net_(Xt[va]).squeeze(-1)
                vl = lossf(o, yt[va][:, None].expand_as(o)).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                state = {k_: v.clone() for k_, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(state)
        return self

    def predict_proba(self, X):
        import torch
        self.net_.eval()
        with torch.no_grad():
            o = self.net_(torch.tensor(self.sc_.transform(X).astype(np.float32))).squeeze(-1)
            p1 = torch.sigmoid(o).mean(1).numpy()
        return np.column_stack([1 - p1, p1])


REGISTRY = {
    "LR": lr, "RF": rf, "XGBoost": xgb, "HGB": hgb,
    "TabICL-v1": lambda seed=0: TabICLWrap("v1", seed),
    "TabICL-v2": lambda seed=0: TabICLWrap("v2", seed),
    "TabM": lambda seed=0: TabMWrap(seed),
}


def available(names):
    ok = []
    for n in names:
        try:
            if n == "XGBoost":
                import xgboost  # noqa
            if n.startswith("TabICL"):
                import tabicl  # noqa
            if n == "TabM":
                import tabm, torch  # noqa
            ok.append(n)
        except Exception as e:  # pragma: no cover
            print(f"[models] {n} unavailable: {e}")
    return ok
