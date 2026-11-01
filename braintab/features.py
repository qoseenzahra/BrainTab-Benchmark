"""Connectivity features: Pearson/Fisher-z, Ledoit-Wolf covariances, tangent space."""
import numpy as np


def zscore_ts(x):
    x = x - x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0          # zero-variance ROI -> stays zero (corr = 0)
    return x / sd


def ledoit_wolf(x):
    """Ledoit-Wolf shrunk covariance of standardized time series (T x R)."""
    from sklearn.covariance import LedoitWolf
    c = LedoitWolf(assume_centered=True).fit(zscore_ts(x)).covariance_
    return c


def correlation(x):
    z = zscore_ts(x)
    c = (z.T @ z) / z.shape[0]
    np.fill_diagonal(c, 1.0)
    return np.clip(c, -0.999999, 0.999999)


def triu(m, k=1):
    i, j = np.triu_indices(m.shape[-1], k=k)
    return m[..., i, j]


def fisher_z_vectors(ts_list):
    return np.stack([np.arctanh(triu(correlation(x))) for x in ts_list]).astype(np.float32)


def _sym_fn(m, fn):
    w, v = np.linalg.eigh(m)
    return (v * fn(w)) @ v.T


def _sqrt_inv(m):
    return _sym_fn(m, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-10)))


def log_euclid_mean(covs):
    return _sym_fn(np.mean([_sym_fn(c, lambda w: np.log(np.maximum(w, 1e-10))) for c in covs], 0), np.exp)


class TangentSpace:
    """Tangent-space embedding (Varoquaux et al., 2010; Dadi et al., 2019).

    The reference point is estimated ONLY from training covariances
    (log-Euclidean mean), so no test information leaks into the embedding.
    Off-diagonal elements are scaled by sqrt(2) to preserve the norm.
    """
    def fit(self, covs):
        self.ref_ = log_euclid_mean(covs)
        self.whiten_ = _sqrt_inv(self.ref_)
        return self

    def transform(self, covs):
        R = covs.shape[-1]
        iu = np.triu_indices(R, k=0)
        scale = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
        out = np.empty((len(covs), len(iu[0])), np.float32)
        W = self.whiten_
        for n, c in enumerate(covs):
            t = _sym_fn(W @ c @ W, lambda w: np.log(np.maximum(w, 1e-10)))
            out[n] = t[iu] * scale
        return out
