"""BrainTab-Benchmark v1 - main experiment runner (ABIDE-I and ADHD-200).

Usage:  python scripts/02_run_benchmark.py <experiment> [--models A,B,...] [--quick]
Experiments: main, harm, fewshot, loso, groups, perm, samplesize, adhd
All preprocessing (scaling, ComBat, PCA, PLS) is fitted on training/context subjects only.
Results are appended row-by-row to results/<experiment>.csv, so runs are resumable.
"""
import sys, os, time, argparse, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from braintab.compress import make_reducer
from braintab.harmonize import ComBat, site_standardize
from braintab.models import REGISTRY, available, lr_full
from braintab.metrics import all_metrics
from braintab.groups import roi_lobes, edge_masks

RES = os.environ.get("BT_RESULTS", os.path.join(ROOT, "results")); os.makedirs(RES, exist_ok=True)
COMPACT_MODELS = ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]


# ---------------------------------------------------------------- data
def load_abide(qc=True):
    # revision: BT_ABIDE_FEATURES selects an alternative atlas (e.g. abide_features_aal.npz); default CC200
    d = np.load(os.path.join(ROOT, "data", os.environ.get("BT_ABIDE_FEATURES", "abide_features.npz")))
    meta = pd.read_csv(os.path.join(ROOT, "data", "abide_meta.csv"))
    m = meta.qc_pass.values if qc else np.ones(len(meta), bool)
    meta = meta[m].reset_index(drop=True)
    X = d["fisher_z"][m].astype(np.float32)
    cov = np.column_stack([meta.AGE_AT_SCAN.values, (meta.SEX.values == 2).astype(float)])
    return X, meta.y.values.astype(int), meta.site.values.astype(str), cov, meta


def load_adhd():
    up = os.path.join(ROOT, "data")
    X = np.load(os.path.join(up, "adhd_X_fc_schaefer200.npy")).astype(np.float64)
    X = np.arctanh(np.clip(X, -0.999999, 0.999999)).astype(np.float32)
    y = np.load(os.path.join(up, "adhd_y.npy")).astype(int)
    site = np.load(os.path.join(up, "adhd_sites.npy"), allow_pickle=True).astype(str)
    site = np.array([s.split("_")[0] for s in site])      # Peking_1/2/3 -> Peking (one acquisition centre)
    return X, y, site


def load_adhd_full():
    """Full ADHD-200 release (Athena pipeline, CC200 time courses; QC_Athena pass; ADHD vs TDC)."""
    d = np.load(os.path.join(ROOT, "data", "adhd_full_features.npz"))
    meta = pd.read_csv(os.path.join(ROOT, "data", "adhd_full_meta.csv"))
    X = d["fisher_z"].astype(np.float32)
    cov = np.column_stack([meta.age.values, meta.male.values.astype(float)])
    return X, meta.y.values.astype(int), meta.site.values.astype(str), cov, meta


# ---------------------------------------------------------------- io
class Sink:
    def __init__(self, name):
        self.path = os.path.join(RES, f"{name}.csv")
        self.ppath = os.path.join(RES, f"{name}_preds.csv")
        self.done = set()
        if os.path.exists(self.path):
            self.done = set(pd.read_csv(self.path)["key"].astype(str))

    def has(self, key):
        return key in self.done

    def add(self, key, row, idx=None, y=None, p=None):
        row = dict(key=key, **row)
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=not os.path.exists(self.path), index=False)
        if idx is not None:
            pd.DataFrame(dict(key=key, idx=idx, y=y, p=p)).to_csv(
                self.ppath, mode="a", header=not os.path.exists(self.ppath), index=False)
        self.done.add(key)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- core
def prepare(Xtr, Xte, ytr, rep, harm=None, str_=None, ste=None, ctr=None, cte=None, seed=0):
    if harm == "combat":
        cb = ComBat().fit(Xtr, str_, ctr)
        Xtr, Xte = cb.transform(Xtr, str_, ctr), cb.transform(Xte, ste, cte)
    r = make_reducer(rep, seed)
    A = r.fit_transform(Xtr, ytr); B = r.transform(Xte)
    info = {"k": getattr(r, "k_", getattr(r, "k", None))}
    return A, B, info


def fit_predict(name, A, ytr, B, seed=0):
    t0 = time.time()
    m = REGISTRY[name](seed).fit(A, ytr)
    p = m.predict_proba(B)[:, 1]
    return p, time.time() - t0


def run_split(sink, tag, X, y, tr, te, rep, models, harm=None, site=None, cov=None, seed=0, extra=None):
    """Fit representation once on the training split, then every model."""
    todo = [m for m in models if not sink.has(f"{tag}|{rep}|{harm}|{m}")]
    # TabM (trained network, slowest model) is evaluated on the PLS representations only
    todo = [m for m in todo if not (m == "TabM" and not rep.startswith("pls"))]
    if rep == "full":
        if os.environ.get("BT_SKIP_FULL") or sink.has(f"{tag}|full|{harm}|LR-full"):
            return
        todo = ["LR-full"]
    if not todo:
        return
    if rep == "full":
        Xtr, Xte = X[tr], X[te]
        if harm == "combat":
            cb = ComBat().fit(Xtr, site[tr], cov[tr]); Xtr, Xte = cb.transform(Xtr, site[tr], cov[tr]), cb.transform(Xte, site[te], cov[te])
        t0 = time.time(); p = lr_full().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        met = all_metrics(y[te], p)
        sink.add(f"{tag}|full|{harm}|LR-full", dict(tag=tag, rep="full", harm=harm, model="LR-full", n_train=len(tr),
                 n_test=len(te), k=None, sec=time.time() - t0, **(extra or {}), **met), te, y[te], p)
        return
    A, B, info = prepare(X[tr], X[te], y[tr], rep, harm,
                         None if site is None else site[tr], None if site is None else site[te],
                         None if cov is None else cov[tr], None if cov is None else cov[te], seed)
    for m in todo:
        try:
            p, sec = fit_predict(m, A, y[tr], B, seed)
        except Exception as e:
            log("  !!", m, "failed:", repr(e)[:200]); continue
        met = all_metrics(y[te], p)
        sink.add(f"{tag}|{rep}|{harm}|{m}", dict(tag=tag, rep=rep, harm=harm, model=m, n_train=len(tr), n_test=len(te),
                 k=info["k"], sec=sec, **(extra or {}), **met), te, y[te], p)
        log(f"  {tag} {rep:8s} {str(harm):6s} {m:10s} acc={met['acc']:.3f} auc={met['auc']:.3f} ({sec:.1f}s)")


def _split_ok(f):
    """BT_SPLITS='10-29' restricts a worker to a range of split indices (for parallel workers)."""
    r = os.environ.get("BT_SPLITS", "")
    if not r:
        return True
    a, b = (int(v) for v in r.split("-"))
    return a <= f <= b


def _reps(default):
    r = os.environ.get("BT_REPS", "")
    return [x for x in r.split(",") if x] if r else default


def strat_labels(site, y):
    return np.array([f"{s}_{c}" for s, c in zip(site, y)])


# ---------------------------------------------------------------- experiments
def exp_main(models, quick=False):
    """E1: repeated (5x) stratified 10-fold CV, N=873; representation comparison."""
    X, y, site, cov, _ = load_abide()
    sink = Sink("main")
    reps = _reps(["plsauto", "plsautocf", "pca20", "pca100", "full"])
    rsk = RepeatedStratifiedKFold(n_splits=10, n_repeats=1 if quick else 3, random_state=2026)
    for f, (tr, te) in enumerate(rsk.split(X, strat_labels(site, y))):
        if not _split_ok(f):
            continue
        log(f"[main] split {f}")
        for rep in reps:
            # revision: every model is now evaluated on all 30 identical splits (BT_NOBUDGET=1)
            ms = [m for m in models if os.environ.get("BT_NOBUDGET") or
                  not (f >= 10 and m.startswith("TabICL") and rep in ("pca20", "plsautocf"))]
            run_split(sink, f"r{f // 10}f{f % 10}", X, y, tr, te, rep, ms, site=site, cov=cov,
                      extra=dict(repeat=f // 10, fold=f % 10))


def exp_harm(models, quick=False):
    """E2: harmonization + representation ablation (10-fold, 2 repeats)."""
    if os.environ.get("BT_RESULTS", "").rstrip("\\/").endswith("w1"):
        log("[harm] skipped in worker w1 (re-balanced to a separate worker)"); return
    os.environ["BT_TABICL_NEST"] = "32"      # same TabICL ensemble size as the main benchmark
    X, y, site, cov, _ = load_abide()
    sink = Sink("harm")
    rsk = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=2026)
    for f, (tr, te) in enumerate(rsk.split(X, strat_labels(site, y))):
        log(f"[harm] split {f}")
        tag = f"r{f // 10}f{f % 10}"
        for rep, harm in [("plsauto", "combat"), ("pls5cf", None), ("pls3", None), ("pls5", None),
                          ("pls8", None), ("pls12", None), ("pca50", None), ("pca200", None)]:
            # wide PCA inputs are evaluated with the fast baselines only (TabICL cost grows with width)
            ms = [m for m in models if not (rep in ("pca50", "pca200") and (m.startswith("TabICL") or m == "TabM"))]
            run_split(sink, tag, X, y, tr, te, rep, ms, harm=harm, site=site, cov=cov,
                      extra=dict(repeat=f // 10, fold=f % 10))
        run_split(sink, tag, X, y, tr, te, "full", models, harm="combat", site=site, cov=cov,
                  extra=dict(repeat=f // 10, fold=f % 10))


def exp_fewshot(models, quick=False):
    """E3: label efficiency. Context sizes n in {10,20,50,100,200,400} (class-balanced),
    drawn from the training fold; test on the held-out fold. 5 outer folds x 5 draws."""
    X, y, site, cov, _ = load_abide()
    sink = Sink("fewshot")
    outer = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, strat_labels(site, y)))
    sizes = [10, 20, 50, 100, 200, 400]
    draws = 2 if quick else 5
    only = [int(f) for f in os.environ.get("BT_FOLDS", "").split(",") if f.strip()]
    for fo, (tr, te) in enumerate(outer):
        if only and fo not in only:
            continue
        for n in sizes:
            for d in range(draws):
                rng = np.random.default_rng(1000 * fo + 10 * n + d)
                ctx = np.concatenate([rng.choice(tr[y[tr] == c], n // 2, replace=False) for c in (0, 1)])
                log(f"[fewshot] fold {fo} n={n} draw {d}")
                run_split(sink, f"f{fo}n{n}d{d}", X, y, ctx, te, "plsauto" if n >= 50 else f"pls{max(1, min(3, n // 10))}",
                          models, seed=d, extra=dict(fold=fo, n_context=n, draw=d))


def exp_loso(models, quick=False):
    """E4: leave-one-site-out (17 sites), with and without label-free per-site standardization."""
    X, y, site, cov, _ = load_abide()
    sink = Sink("loso")
    Xs = site_standardize(X, site)
    for s in np.unique(site):
        tr, te = np.where(site != s)[0], np.where(site == s)[0]
        log(f"[loso] site {s} n_test={len(te)}")
        for harm, XX in [(None, X), ("sitez", Xs)]:
            for rep in ["plsauto", "full"]:
                run_split(sink, f"site={s}", XX, y, tr, te, rep, models, harm=harm,
                          extra=dict(site=s, n_asd=int(y[te].sum()), n_td=int((1 - y[te]).sum())))


def exp_loso100(models, quick=False):
    """E4b: leave-one-site-out with PCA-100 inputs (no harmonisation)."""
    X, y, site, cov, _ = load_abide()
    sink = Sink("loso")
    for s in np.unique(site):
        tr, te = np.where(site != s)[0], np.where(site == s)[0]
        log(f"[loso100] site {s} n_test={len(te)}")
        run_split(sink, f"site={s}", X, y, tr, te, "pca100", models,
                  extra=dict(site=s, n_asd=int(y[te].sum()), n_td=int((1 - y[te]).sum())))


def exp_fewshot_u(models, quick=False):
    """E3b: semi-supervised few-shot. PCA-100 is fitted on ALL training-fold participants WITHOUT
    labels (unlabelled connectomes are cheap); only n labelled participants form the context."""
    from braintab.compress import PCAReducer
    X, y, site, cov, _ = load_abide()
    sink = Sink("fewshot_u")
    outer = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, strat_labels(site, y)))
    only = [int(f) for f in os.environ.get("BT_FOLDS", "").split(",") if f.strip()]
    for fo, (tr, te) in enumerate(outer):
        if only and fo not in only:
            continue
        r = PCAReducer(100); r.fit_transform(X[tr]); Atr_all = r.transform(X)   # fitted on training fold, no labels
        sizes = [int(v) for v in os.environ.get("BT_FS_SIZES", "10,20,50,100,200,400").split(",")]
        for n in sizes:
            for d in range(2):
                rng = np.random.default_rng(1000 * fo + 10 * n + d)     # same draws as E3
                ctx = np.concatenate([rng.choice(tr[y[tr] == c], n // 2, replace=False) for c in (0, 1)])
                for m in models:
                    key = f"f{fo}n{n}d{d}|pca100u|None|{m}"
                    if sink.has(key):
                        continue
                    try:
                        p, sec = fit_predict(m, Atr_all[ctx], y[ctx], Atr_all[te], seed=d)
                    except Exception as e:
                        log("  !!", m, repr(e)[:200]); continue
                    met = all_metrics(y[te], p)
                    sink.add(key, dict(tag=f"f{fo}n{n}d{d}", rep="pca100u", harm=None, model=m, n_train=len(ctx), n_test=len(te),
                                       k=100, sec=sec, fold=fo, n_context=n, draw=d, **met), te, y[te], p)
                    log(f"[fewshot_u] f{fo} n={n} d{d} {m} acc={met['acc']:.3f} auc={met['auc']:.3f} ({sec:.1f}s)")


def exp_shortcut(models, quick=False):
    """E4c: site-shortcut test inside random CV. Same 10 folds as E1 repetition 1, but every edge is
    z-scored within its acquisition site (label-free) before PCA/PLS, removing site means/scales."""
    X, y, site, cov, _ = load_abide()
    Xs = site_standardize(X, site)
    sink = Sink("shortcut")
    rsk = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=2026)
    for f, (tr, te) in enumerate(rsk.split(X, strat_labels(site, y))):
        log(f"[shortcut] split {f}")
        for rep in ["pca100", "plsauto"]:
            run_split(sink, f"r0f{f}", Xs, y, tr, te, rep, models, harm="sitez", extra=dict(repeat=0, fold=f))


def exp_loso100z(models, quick=False):
    """E4d: leave-one-site-out with PCA-100 after label-free per-site standardisation."""
    X, y, site, cov, _ = load_abide()
    Xs = site_standardize(X, site)
    sink = Sink("loso")
    for s in np.unique(site):
        tr, te = np.where(site != s)[0], np.where(site == s)[0]
        log(f"[loso100z] site {s} n_test={len(te)}")
        run_split(sink, f"site={s}", Xs, y, tr, te, "pca100", models, harm="sitez",
                  extra=dict(site=s, n_asd=int(y[te].sum()), n_td=int((1 - y[te]).sum())))


def exp_groups(models, quick=False):
    """E5: anatomical feature groups (incident edges) vs size-matched random edge sets."""
    X, y, site, cov, _ = load_abide()
    masks = edge_masks(roi_lobes(os.path.join(ROOT, "data", "CC200_ROI_labels.csv")))
    sink = Sink("groups")
    rsk = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=2026)
    for f, (tr, te) in enumerate(rsk.split(X, strat_labels(site, y))):
        log(f"[groups] split {f}")
        for g, mk in masks.items():
            cols = np.where(mk)[0]
            run_split(sink, f"r0f{f}|{g}", X[:, cols], y, tr, te, "plsauto", models,
                      extra=dict(group=g, n_edges=len(cols), control=0, repeat=0, fold=f))
            if g != "Whole-brain":
                rng = np.random.default_rng(hash(g) % 2**32 + f)
                rc = rng.choice(X.shape[1], len(cols), replace=False)
                run_split(sink, f"r0f{f}|{g}-random", X[:, rc], y, tr, te, "plsauto", models,
                          extra=dict(group=g, n_edges=len(cols), control=1, repeat=0, fold=f))


def exp_perm(models, quick=False):
    """E6: label-permutation test (5-fold CV, whole pipeline incl. PLS re-fitted per permutation)."""
    X, y, site, cov, _ = load_abide()
    sink = Sink("perm")
    n_perm = int(os.environ.get("BT_NPERM", "100")); start = int(os.environ.get("BT_PERM_START", "-1"))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=99).split(X, strat_labels(site, y)))
    for b in range(start, 5 if quick else n_perm):
        yb = y if b < 0 else np.random.default_rng(b).permutation(y)
        for m in models:
            key = f"perm{b}|{m}"
            if sink.has(key):
                continue
            P = np.zeros(len(y))
            for tr, te in folds:
                A, B, _ = prepare(X[tr], X[te], yb[tr], "pls5")
                P[te], _ = fit_predict(m, A, yb[tr], B)
            met = all_metrics(yb, P)
            sink.add(key, dict(perm=b, model=m, **met))
            log(f"[perm] b={b} {m} acc={met['acc']:.3f} auc={met['auc']:.3f}")


def exp_samplesize(models, quick=False):
    """E7: effect of cohort size (subsamples of N=200/400/600 vs full 873), 5-fold CV, 5 draws."""
    X, y, site, cov, _ = load_abide()
    sink = Sink("samplesize")
    for N in [200, 400, 600, 873]:
        for d in range(2 if quick else 3):
            rng = np.random.default_rng(N + d)
            sub = np.sort(rng.choice(len(y), N, replace=False)) if N < len(y) else np.arange(len(y))
            for fo, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=d).split(sub, y[sub])):
                log(f"[samplesize] N={N} draw {d} fold {fo}")
                for rep in ["plsauto"]:
                    run_split(sink, f"N{N}d{d}f{fo}", X, y, sub[tr], sub[te], rep, models,
                              extra=dict(N=N, draw=d, fold=fo))


def exp_adhd(models, quick=False):
    """E8: ADHD-200 external stress test (nilearn 40-subject subset, Schaefer-200 FC)."""
    X, y, site = load_adhd()
    sink = Sink("adhd")
    rsk = RepeatedStratifiedKFold(n_splits=5, n_repeats=2 if quick else 10, random_state=2026)
    for f, (tr, te) in enumerate(rsk.split(X, y)):
        for rep in ["pls2", "pca10", "full"]:
            run_split(sink, f"r{f // 5}f{f % 5}", X, y, tr, te, rep, models, extra=dict(repeat=f // 5, fold=f % 5))
    for s in np.unique(site):
        tr, te = np.where(site != s)[0], np.where(site == s)[0]
        if len(np.unique(y[te])) < 2 or len(te) < 4:
            continue
        for rep in ["pls2", "full"]:
            run_split(sink, f"loso={s}", X, y, tr, te, rep, models, extra=dict(site=s))
    # permutation test on repeated CV mean accuracy/AUC (pls2)
    sinkp = Sink("adhd_perm")
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=11).split(X, y))
    for b in range(-1, 10 if quick else int(os.environ.get("BT_NPERM_ADHD", "50"))):
        yb = y if b < 0 else np.random.default_rng(b).permutation(y)
        for m in models:
            key = f"perm{b}|{m}"
            if sinkp.has(key):
                continue
            accs, aucs = [], []
            for tr, te in folds:
                A, B, _ = prepare(X[tr], X[te], yb[tr], "pls2")
                p, _ = fit_predict(m, A, yb[tr], B)
                mt = all_metrics(yb[te], p); accs.append(mt["acc"]); aucs.append(mt["auc"])
            sinkp.add(key, dict(perm=b, model=m, acc=np.mean(accs), auc=np.mean(aucs)))
            if b < 0 or b % 20 == 0:
                log(f"[adhd-perm] b={b} {m} acc={np.mean(accs):.3f} auc={np.mean(aucs):.3f}")


def exp_adhdfull(models, quick=False):
    """E8 (full ADHD-200): repeated 10-fold CV with the same representations as E1."""
    X, y, site, cov, _ = load_adhd_full()
    sink = Sink("adhdfull")
    rsk = RepeatedStratifiedKFold(n_splits=10, n_repeats=1 if quick else 3, random_state=2026)
    for f, (tr, te) in enumerate(rsk.split(X, strat_labels(site, y))):
        if not _split_ok(f):
            continue
        log(f"[adhdfull] split {f}")
        for rep in _reps(["plsauto", "pca100", "full"]):
            ms = [m for m in models if os.environ.get("BT_NOBUDGET") or
                  not (f >= 10 and m.startswith("TabICL") and rep == "pca100")]
            run_split(sink, f"r{f // 10}f{f % 10}", X, y, tr, te, rep, ms, site=site, cov=cov,
                      extra=dict(repeat=f // 10, fold=f % 10))


def exp_adhdfull_loso(models, quick=False):
    """E8b: leave-one-site-out on ADHD-200 (sites containing both classes), +/- per-site z-scoring."""
    X, y, site, cov, _ = load_adhd_full()
    Xs = site_standardize(X, site)
    sink = Sink("adhdfull_loso")
    for s in np.unique(site):
        tr, te = np.where(site != s)[0], np.where(site == s)[0]
        if len(np.unique(y[te])) < 2:
            continue
        log(f"[adhdfull_loso] site {s} n_test={len(te)}")
        for harm, XX in [(None, X), ("sitez", Xs)]:
            for rep in ["plsauto", "full"]:
                run_split(sink, f"site={s}", XX, y, tr, te, rep, models, harm=harm,
                          extra=dict(site=s, n_adhd=int(y[te].sum()), n_tdc=int((1 - y[te]).sum())))


def exp_adhdfull_perm(models, quick=False):
    """E8c: label-permutation test on full ADHD-200 (5-fold CV, PLS-5 re-fitted per permutation)."""
    X, y, site, cov, _ = load_adhd_full()
    sink = Sink("adhdfull_perm")
    n_perm = int(os.environ.get("BT_NPERM_ADHD", "100")); start = int(os.environ.get("BT_PERM_START", "-1"))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=99).split(X, strat_labels(site, y)))
    for b in range(start, 5 if quick else n_perm):
        yb = y if b < 0 else np.random.default_rng(b).permutation(y)
        for m in models:
            key = f"perm{b}|{m}"
            if sink.has(key):
                continue
            P = np.zeros(len(y))
            for tr, te in folds:
                A, B, _ = prepare(X[tr], X[te], yb[tr], "pls5")
                P[te], _ = fit_predict(m, A, yb[tr], B)
            met = all_metrics(yb, P)
            sink.add(key, dict(perm=b, model=m, **met))
            log(f"[adhdfull_perm] b={b} {m} acc={met['acc']:.3f} auc={met['auc']:.3f}")


def exp_adhdfull_fewshot(models, quick=False):
    """E8d: few-shot on full ADHD-200. (i) PLS fitted on the labelled context; (ii) semi-supervised
    PCA-100 fitted on all training-fold participants without labels. 5 outer folds x 2 draws."""
    from braintab.compress import PCAReducer
    X, y, site, cov, _ = load_adhd_full()
    sink = Sink("adhdfull_fewshot")
    outer = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, strat_labels(site, y)))
    only = [int(f) for f in os.environ.get("BT_FOLDS", "").split(",") if f.strip()]
    for fo, (tr, te) in enumerate(outer):
        if only and fo not in only:
            continue
        r = PCAReducer(100); r.fit_transform(X[tr]); Aall = r.transform(X)
        for n in [20, 50, 100, 200, 400]:
            if n // 2 > min((y[tr] == 0).sum(), (y[tr] == 1).sum()):
                continue
            for d in range(2):
                rng = np.random.default_rng(1000 * fo + 10 * n + d)
                ctx = np.concatenate([rng.choice(tr[y[tr] == c], n // 2, replace=False) for c in (0, 1)])
                log(f"[adhdfull_fewshot] fold {fo} n={n} draw {d}")
                run_split(sink, f"f{fo}n{n}d{d}", X, y, ctx, te, "plsauto" if n >= 50 else "pls2", models, seed=d,
                          extra=dict(fold=fo, n_context=n, draw=d))
                for m in models:
                    key = f"f{fo}n{n}d{d}|pca100u|None|{m}"
                    if sink.has(key):
                        continue
                    try:
                        p, sec = fit_predict(m, Aall[ctx], y[ctx], Aall[te], seed=d)
                    except Exception as e:
                        log("  !!", m, repr(e)[:200]); continue
                    met = all_metrics(y[te], p)
                    sink.add(key, dict(tag=f"f{fo}n{n}d{d}", rep="pca100u", harm=None, model=m, n_train=len(ctx),
                                       n_test=len(te), k=100, sec=sec, fold=fo, n_context=n, draw=d, **met), te, y[te], p)


def exp_adhdmixed(models, quick=False):
    """E8e (sensitivity): ADHD-200 restricted to the five sites that contributed both ADHD and control
    participants (Pittsburgh and WashU contributed controls only), 3 x 10-fold CV."""
    X, y, site, cov, _ = load_adhd_full()
    keep = np.isin(site, [s for s in np.unique(site) if len(np.unique(y[site == s])) == 2])
    X, y, site, cov = X[keep], y[keep], site[keep], cov[keep]
    sink = Sink("adhdmixed")
    rsk = RepeatedStratifiedKFold(n_splits=10, n_repeats=1 if quick else 3, random_state=2026)
    for f, (tr, te) in enumerate(rsk.split(X, strat_labels(site, y))):
        log(f"[adhdmixed] split {f} (N={len(y)})")
        for rep in ["plsauto", "full"]:
            run_split(sink, f"r{f // 10}f{f % 10}", X, y, tr, te, rep, models, site=site, cov=cov,
                      extra=dict(repeat=f // 10, fold=f % 10))


def exp_adhdmixed_perm(models, quick=False):
    """E8f: site-respecting permutation test on the five mixed-diagnosis ADHD-200 sites: labels are permuted
    WITHIN each site, so the null keeps every site's ADHD/TDC ratio and any site shortcut is preserved under H0."""
    X, y, site, cov, _ = load_adhd_full()
    keep = np.isin(site, [s for s in np.unique(site) if len(np.unique(y[site == s])) == 2])
    X, y, site = X[keep], y[keep], site[keep]
    sink = Sink("adhdmixed_perm")
    n_perm = int(os.environ.get("BT_NPERM_ADHD", "100")); start = int(os.environ.get("BT_PERM_START", "-1"))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=99).split(X, strat_labels(site, y)))
    for b in range(start, 3 if quick else n_perm):
        yb = y.copy()
        if b >= 0:
            rng = np.random.default_rng(10_000 + b)
            for s in np.unique(site):
                m = np.where(site == s)[0]; yb[m] = rng.permutation(y[m])
        for m in models:
            key = f"perm{b}|{m}"
            if sink.has(key):
                continue
            P = np.zeros(len(y))
            for tr, te in folds:
                A, B, _ = prepare(X[tr], X[te], yb[tr], "pls5")
                P[te], _ = fit_predict(m, A, yb[tr], B)
            met = all_metrics(yb, P)
            sink.add(key, dict(perm=b, model=m, **met))
            log(f"[adhdmixed_perm] b={b} {m} acc={met['acc']:.3f} auc={met['auc']:.3f}")


def _demo_run(sink, X, y, site, cov, models, prefix):
    """Comment 19: age+sex only, FC only (PLS-auto) and FC (PLS-auto) + age+sex, on the E1 splits."""
    from sklearn.preprocessing import StandardScaler
    rsk = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=2026)
    for f, (tr, te) in enumerate(rsk.split(X, strat_labels(site, y))):
        if not _split_ok(f):
            continue
        tag = f"r{f // 10}f{f % 10}"
        log(f"[{prefix}] split {f}")
        sc = StandardScaler().fit(cov[tr]); Ctr, Cte = sc.transform(cov[tr]), sc.transform(cov[te])
        need_fc = any(not sink.has(f"{tag}|plsauto+agesex|None|{m}") for m in models)
        if need_fc:
            A, B, info = prepare(X[tr], X[te], y[tr], "plsauto")
        for rep in _reps(["agesex", "plsauto+agesex"]):
            for m in models:
                key = f"{tag}|{rep}|None|{m}"
                if sink.has(key):
                    continue
                if rep == "agesex":
                    Atr, Ate, k = Ctr, Cte, 2
                else:
                    Atr, Ate, k = np.column_stack([A, Ctr]), np.column_stack([B, Cte]), info["k"]
                try:
                    p, sec = fit_predict(m, Atr, y[tr], Ate)
                except Exception as e:
                    log("  !!", m, repr(e)[:200]); continue
                met = all_metrics(y[te], p)
                sink.add(key, dict(tag=tag, rep=rep, harm=None, model=m, n_train=len(tr), n_test=len(te), k=k, sec=sec,
                                   repeat=f // 10, fold=f % 10, **met), te, y[te], p)
                log(f"  {tag} {rep:15s} {m:10s} auc={met['auc']:.3f} ({sec:.1f}s)")


def exp_demo(models, quick=False):
    X, y, site, cov, _ = load_abide()
    ok = ~np.isnan(cov).any(1)
    _demo_run(Sink("demo"), X, y, site, np.where(np.isnan(cov), np.nanmean(cov, 0), cov), models, "demo")


def exp_adhddemo(models, quick=False):
    X, y, site, cov, _ = load_adhd_full()
    _demo_run(Sink("adhddemo"), X, y, site, np.where(np.isnan(cov), np.nanmean(cov, 0), cov), models, "adhddemo")


def exp_adhdmixeddemo(models, quick=False):
    """Demographic control on the five mixed-diagnosis ADHD-200 sites (same splits as adhdmixed)."""
    X, y, site, cov, _ = load_adhd_full()
    keep = np.isin(site, [s for s in np.unique(site) if len(np.unique(y[site == s])) == 2])
    X, y, site, cov = X[keep], y[keep], site[keep], cov[keep]
    _demo_run(Sink("adhdmixeddemo"), X, y, site, np.where(np.isnan(cov), np.nanmean(cov, 0), cov), models, "adhdmixeddemo")


EXPS = dict(adhdmixeddemo=exp_adhdmixeddemo, demo=exp_demo, adhddemo=exp_adhddemo, adhdmixed_perm=exp_adhdmixed_perm, adhdmixed=exp_adhdmixed, adhdfull_fewshot=exp_adhdfull_fewshot, adhdfull=exp_adhdfull, adhdfull_loso=exp_adhdfull_loso, adhdfull_perm=exp_adhdfull_perm,
            main=exp_main, harm=exp_harm, fewshot=exp_fewshot, loso=exp_loso, groups=exp_groups,
            loso100=exp_loso100, fewshot_u=exp_fewshot_u, shortcut=exp_shortcut, loso100z=exp_loso100z,
            perm=exp_perm, samplesize=exp_samplesize, adhd=exp_adhd)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=list(EXPS) + ["all"])
    ap.add_argument("--models", default=",".join(COMPACT_MODELS))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    models = available(a.models.split(","))
    log("models:", models)
    for e in (EXPS if a.exp == "all" else [a.exp]):
        t0 = time.time(); EXPS[e](models, a.quick); log(f"== {e} finished in {(time.time() - t0) / 60:.1f} min")
