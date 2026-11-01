"""Compact representations for in-context learning on connectomes.

All reducers are fitted using TRAINING (context) subjects only.

CrossFitPLS: supervised partial-least-squares compression with cross-fitting.
Training-context embeddings are produced out-of-fold (inner K folds), so that the
context rows seen by an in-context learner have the same (honest) signal-to-noise
as test rows; the test embedding uses a PLS model fitted on all training rows.
Inner-fold PLS bases are sign/rotation-aligned to the full-data basis by an
orthogonal Procrustes step.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


class PCAReducer:
    def __init__(self, k, seed=0):
        self.k, self.seed = k, seed

    def fit_transform(self, X, y=None):
        self.sc_ = StandardScaler().fit(X)
        Z = self.sc_.transform(X)
        self.p_ = PCA(min(self.k, len(X) - 1), random_state=self.seed).fit(Z)
        return self.p_.transform(Z)

    def transform(self, X):
        return self.p_.transform(self.sc_.transform(X))


class CrossFitPLS:
    def __init__(self, k, inner=5, cross_fit=True, seed=0):
        self.k, self.inner, self.cross_fit, self.seed = k, inner, cross_fit, seed

    def _fit(self, X, y):
        return PLSRegression(self.k, scale=False).fit(X, y.astype(float))

    def fit_transform(self, X, y):
        self.sc_ = StandardScaler().fit(X)
        Z = self.sc_.transform(X)
        self.k = int(min(self.k, max(1, len(X) // 4)))
        self.full_ = self._fit(Z, y)
        if not self.cross_fit:
            return self.full_.transform(Z)
        R_full = self.full_.x_rotations_
        out = np.zeros((len(X), self.k))
        n_inner = int(min(self.inner, np.bincount(y).min()))
        if n_inner < 2:
            return self.full_.transform(Z)
        for tr, te in StratifiedKFold(n_inner, shuffle=True, random_state=self.seed).split(Z, y):
            m = self._fit(Z[tr], y[tr])
            # align inner rotation to the full rotation (orthogonal Procrustes)
            U, _, Vt = np.linalg.svd(m.x_rotations_.T @ R_full)
            Q = U @ Vt
            out[te] = ((Z[te] - m._x_mean) @ m.x_rotations_) @ Q
        return out

    def transform(self, X):
        return self.full_.transform(self.sc_.transform(X))


class AutoPLS:
    """In-fold PLS whose number of components is chosen by an inner stratified
    5-fold CV *inside the training set* (logistic-regression proxy, AUC).
    PLS1 components are nested, so one k_max fit per inner split suffices."""
    def __init__(self, grid=(2, 3, 5, 8, 12), inner=5, seed=0, cross_fit=False):
        self.grid, self.inner, self.seed, self.cross_fit = tuple(grid), inner, seed, cross_fit

    def fit_transform(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        kmax = max(self.grid)
        sc = StandardScaler().fit(X); Z = sc.transform(X)
        scores = {k: [] for k in self.grid}
        for tr, va in StratifiedKFold(self.inner, shuffle=True, random_state=self.seed).split(Z, y):
            s2 = StandardScaler().fit(Z[tr]); A, B = s2.transform(Z[tr]), s2.transform(Z[va])
            m = PLSRegression(kmax, scale=False).fit(A, y[tr].astype(float))
            TA, TB = m.transform(A), m.transform(B)
            for k in self.grid:
                c = LogisticRegression(max_iter=5000).fit(TA[:, :k], y[tr])
                scores[k].append(roc_auc_score(y[va], c.predict_proba(TB[:, :k])[:, 1]))
        self.k_ = max(self.grid, key=lambda k: np.mean(scores[k]))
        self.cv_scores_ = {k: float(np.mean(v)) for k, v in scores.items()}
        self.inner_ = CrossFitPLS(self.k_, cross_fit=self.cross_fit, seed=self.seed)
        return self.inner_.fit_transform(X, y)

    def transform(self, X):
        return self.inner_.transform(X)


def make_reducer(spec, seed=0):
    """'pca20' (unsupervised PCA), 'pls5' (in-fold PLS, 5 comps),
    'pls5cf' (cross-fitted PLS), 'plsauto' (nested choice of k)."""
    if spec.startswith("pca"):
        return PCAReducer(int(spec[3:]), seed)
    if spec in ("plsauto", "plsautocf"):
        return AutoPLS(seed=seed, cross_fit=spec.endswith("cf"))
    if spec.startswith("pls"):
        cf = spec.endswith("cf")
        return CrossFitPLS(int(spec[3:].replace("cf", "")), cross_fit=cf, seed=seed)
    raise ValueError(spec)
