"""Leakage-free site harmonization.

ComBat (Johnson et al., 2007; Fortin et al., 2017/2018) with parametric empirical-Bayes
priors. Parameters are estimated on TRAINING subjects only and then applied to test
subjects from the same sites. Diagnosis is never used as a covariate, so the
transformation of a test subject depends only on its site, age and sex.
"""
import numpy as np


def _design(cov):
    return np.column_stack([np.ones(len(cov))] + ([cov] if cov is not None and cov.size else []))


class ComBat:
    def __init__(self, eb=True):
        self.eb = eb

    def fit(self, X, site, cov=None):
        X = np.asarray(X, np.float64); site = np.asarray(site)
        self.sites_ = np.unique(site)
        S = (site[:, None] == self.sites_[None]).astype(float)        # n x s
        C = cov if cov is not None else np.zeros((len(X), 0))
        D = np.column_stack([S, C])
        B = np.linalg.lstsq(D, X, rcond=None)[0]
        n_s = S.sum(0)
        self.grand_ = (n_s / n_s.sum()) @ B[:len(self.sites_)]           # 1 x p
        self.beta_cov_ = B[len(self.sites_):]
        resid = X - D @ B
        self.var_ = (resid ** 2).mean(0) + 1e-12
        Z = (X - self.grand_ - C @ self.beta_cov_) / np.sqrt(self.var_)
        gam = np.stack([Z[site == s].mean(0) for s in self.sites_])
        dl = np.stack([Z[site == s].var(0, ddof=1) + 1e-12 for s in self.sites_])
        if self.eb:
            gam, dl = self._eb(Z, site, gam, dl)
        self.gamma_, self.delta_ = gam, dl
        self.cov_used_ = C.shape[1]
        return self

    def _eb(self, Z, site, gh, dh, it=50, tol=1e-4):
        gbar = gh.mean(1); t2 = gh.var(1, ddof=1)
        m = dh.mean(1); s2 = dh.var(1, ddof=1)
        a = (2 * s2 + m ** 2) / s2; b = (m * s2 + m ** 3) / s2
        G, Dl = gh.copy(), dh.copy()
        for k, s in enumerate(self.sites_):
            z = Z[site == s]; n = len(z)
            g, d = gh[k].copy(), dh[k].copy()
            for _ in range(it):
                g_new = (t2[k] * n * gh[k] + d * gbar[k]) / (t2[k] * n + d)
                ss = ((z - g_new) ** 2).sum(0)
                d_new = (0.5 * ss + b[k]) / (n / 2.0 + a[k] - 1.0)
                conv = max(np.abs(g_new - g).max() / (np.abs(g).max() + 1e-12),
                           np.abs(d_new - d).max() / (np.abs(d).max() + 1e-12))
                g, d = g_new, d_new
                if conv < tol:
                    break
            G[k], Dl[k] = g, d
        return G, Dl

    def transform(self, X, site, cov=None):
        X = np.asarray(X, np.float64); site = np.asarray(site)
        C = cov if cov is not None else np.zeros((len(X), 0))
        Z = (X - self.grand_ - C @ self.beta_cov_) / np.sqrt(self.var_)
        out = np.empty_like(Z)
        for i, s in enumerate(site):
            k = np.searchsorted(self.sites_, s)
            if k >= len(self.sites_) or self.sites_[k] != s:
                out[i] = Z[i]                      # unseen site: no site correction
            else:
                out[i] = (Z[i] - self.gamma_[k]) / np.sqrt(self.delta_[k])
        return (out * np.sqrt(self.var_) + self.grand_ + C @ self.beta_cov_).astype(np.float32)


def site_standardize(X, site):
    """Per-site z-scoring (label-free). Usable for unseen sites at test time
    because it only needs the unlabeled feature distribution of that site."""
    X = np.asarray(X, np.float64).copy(); site = np.asarray(site)
    for s in np.unique(site):
        m = site == s
        mu = X[m].mean(0); sd = X[m].std(0) + 1e-8
        X[m] = (X[m] - mu) / sd
    return X.astype(np.float32)
