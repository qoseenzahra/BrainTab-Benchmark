"""Aggregate results from all workers, compute statistics, and draw figures.
Usage: python scripts/03_analyze.py <out_dir> <results_dir_1> [<results_dir_2> ...]"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

OUT = sys.argv[1]; DIRS = sys.argv[2:]
os.makedirs(OUT, exist_ok=True); FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
# Fixed categorical order (validated reference palette): colour follows the model, never its rank.
MODELS = ["LR-full", "LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]
COL = dict(zip(MODELS, ["#52514e", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#eb6834", "#6250d6"]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "axes.axisbelow": True, "axes.grid.axis": "y",
                     "savefig.dpi": 300, "legend.frameon": False})


def load(name):
    fr, pr = [], []
    for d in DIRS:
        f = os.path.join(d, f"{name}.csv")
        if os.path.exists(f):
            fr.append(pd.read_csv(f))
        p = os.path.join(d, f"{name}_preds.csv")
        if os.path.exists(p):
            pr.append(pd.read_csv(p))
    if not fr:
        return None, None
    R = pd.concat(fr).drop_duplicates("key", keep="last")
    P = pd.concat(pr).drop_duplicates(["key", "idx"], keep="last") if pr else None
    return R, P


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def corrected_ttest(a, b, k, r, n_tr, n_te):
    """Nadeau & Bengio (2003) corrected resampled t-test for r x k-fold CV."""
    d = np.asarray(a) - np.asarray(b); J = len(d)
    v = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1.0 / J + n_te / n_tr) * v) if v > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), J - 1)
    return d.mean(), t, p


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


def boot_ci(y, p, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(fn(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def fmt(m, s, pct=True):
    return f"{100*m:.1f} ± {100*s:.1f}" if pct else f"{m:.3f} ± {s:.3f}"


summary = {}
# ============================================================== E1 main
R, P = load("main")
if R is not None:
    R["model"] = R["model"].astype(str)
    mets = ["acc", "bacc", "auc", "sens", "spec", "f1", "brier", "ece"]
    g = R.groupby(["rep", "model"])
    tab = g[mets].agg(["mean", "std"]); tab["n_folds"] = g.size()
    tab.to_csv(os.path.join(OUT, "E1_main_fold_summary.csv"))
    # pooled (repeat-averaged) predictions -> bootstrap CIs
    rows = []
    P = P.merge(R[["key", "rep", "model", "repeat"]], on="key")
    for (rep, m), d in P.groupby(["rep", "model"]):
        a = d.groupby("idx").agg(y=("y", "first"), p=("p", "mean"))
        y, p = a.y.values, a.p.values
        auc = roc_auc_score(y, p); acc = ((p >= .5) == y).mean()
        ci_auc = boot_ci(y, p, roc_auc_score); ci_acc = boot_ci(y, p, lambda yy, pp: ((pp >= .5) == yy).mean())
        rows.append(dict(rep=rep, model=m, n=len(y), pooled_auc=auc, auc_lo=ci_auc[0], auc_hi=ci_auc[1],
                         pooled_acc=acc, acc_lo=ci_acc[0], acc_hi=ci_acc[1]))
    pooled = pd.DataFrame(rows); pooled.to_csv(os.path.join(OUT, "E1_main_pooled_ci.csv"), index=False)
    # paired corrected t-tests between configurations (model@representation), matched on split
    R["cfg"] = R["model"] + "@" + R["rep"]
    piv = {met: R.pivot_table(index="tag", columns="cfg", values=met) for met in ["acc", "auc", "bacc", "brier"]}
    best = R.groupby("cfg").auc.mean().idxmax(); summary["best_cfg_by_auc"] = best
    refs = [c for c in ["TabICL-v2@pca100", "TabICL-v1@pca100", best] if c in piv["auc"]]
    refs = list(dict.fromkeys(refs))
    n_tr = R.n_train.median(); n_te = R.n_test.median()
    tests = []
    for ref in refs:
        for other in piv["auc"].columns:
            if other == ref:
                continue
            ok = piv["auc"][[ref, other]].dropna().index
            if len(ok) < 8:
                continue
            row = dict(ref=ref, vs=other, n_splits=len(ok))
            for met in ["acc", "auc", "bacc", "brier"]:
                d_, t_, p_ = corrected_ttest(piv[met].loc[ok, ref], piv[met].loc[ok, other], 10, 1, n_tr, n_te)
                row.update({f"d_{met}": d_, f"t_{met}": t_, f"p_{met}": p_})
            tests.append(row)
    T = pd.DataFrame(tests)
    if len(T):
        for met in ["acc", "auc"]:
            T[f"p_{met}_holm"] = holm(T[f"p_{met}"].values)
        T.to_csv(os.path.join(OUT, "E1_paired_tests.csv"), index=False)
    # chosen PLS dimensionality
    ks = R[R.rep.isin(["plsauto", "plsautocf"])].drop_duplicates("tag")["k"].value_counts().to_dict()
    summary["plsauto_k_counts"] = {str(k): int(v) for k, v in ks.items()}

    # ---- Figure 2: representation x model (AUC, mean +- SD over folds)
    reps = [r for r in ["pca20", "pca100", "plsauto", "plsautocf"] if r in R.rep.unique()]
    lab = {"pca20": "PCA-20\n(legacy)", "pca100": "PCA-100", "plsauto": "PLS-auto", "plsautocf": "PLS-auto\n(cross-fit)"}
    models = [m for m in MODELS[1:] if m in R.model.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, met, name in [(axes[0], "acc", "Accuracy"), (axes[1], "auc", "ROC-AUC")]:
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            vals = [R[(R.rep == r) & (R.model == m)][met] for r in reps]
            mu = [v.mean() if len(v) else np.nan for v in vals]; sd = [v.std() if len(v) else 0 for v in vals]
            x = np.arange(len(reps)) + (i - (len(models) - 1) / 2) * w
            ax.bar(x, mu, w * 0.9, yerr=sd, color=COL[m], label=m, error_kw=dict(lw=0.6, capsize=0, ecolor="#52514e"))
        full = R[R.model == "LR-full"][met]
        if len(full):
            ax.axhline(full.mean(), color=COL["LR-full"], lw=1, ls="--", label="LR-full (19,900 edges)")
        ax.set_xticks(range(len(reps))); ax.set_xticklabels([lab[r] for r in reps], fontsize=7.5)
        ax.set_ylim(0.5 if met == "acc" else 0.5, 0.82); ax.set_ylabel(name)
        ax.axhline(0.5, color="#8a8984", lw=0.6)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig2_main_benchmark")

    # ---- Figure 6: ROC (PCA-100) + reliability diagrams
    def rel(ax, d, color, label, ls="-"):
        bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(d.p, bins) - 1, 0, 9)
        xs = [d.p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 15]
        ys = [d.y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 15]
        ax.plot(xs, ys, ls, marker="o", color=color, lw=1.2, ms=3, label=label)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    for m in MODELS:
        rep = "full" if m == "LR-full" else "pca100"
        d = P[(P.rep == rep) & (P.model == m)].groupby("idx").agg(y=("y", "first"), p=("p", "mean"))
        if len(d):
            fpr, tpr, _ = roc_curve(d.y, d.p)
            axes[0].plot(fpr, tpr, color=COL[m], lw=1.2, label=f"{m} ({roc_auc_score(d.y, d.p):.3f})")
    axes[0].plot([0, 1], [0, 1], color="#8a8984", lw=0.6); axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("(a) ROC, PCA-100 inputs", fontsize=8); axes[0].legend(fontsize=5.6, loc="lower right")
    shades = {"pca20": "#c9c3f0", "pca100": "#6250d6", "plsauto": "#eb6834", "plsautocf": "#1baf7a"}
    names = {"pca20": "PCA-20", "pca100": "PCA-100", "plsauto": "PLS-auto", "plsautocf": "PLS-auto (cross-fit)"}
    for rep in ["pca20", "pca100", "plsauto", "plsautocf"]:
        d = P[(P.rep == rep) & (P.model == "TabICL-v2")]
        if len(d):
            rel(axes[1], d, shades[rep], names[rep])
    axes[1].set_title("(b) TabICL-v2: effect of input", fontsize=8)
    for m in MODELS:
        rep = "full" if m == "LR-full" else "pca100"
        d = P[(P.rep == rep) & (P.model == m)]
        if len(d) and m != "TabM":
            rel(axes[2], d, COL[m], m)
    axes[2].set_title("(c) Models on PCA-100", fontsize=8)
    for ax in axes[1:]:
        ax.plot([0, 1], [0, 1], color="#8a8984", lw=0.6, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted P(ASD)"); ax.set_ylabel("Observed ASD fraction"); ax.legend(fontsize=5.6, loc="upper left")
    fig.tight_layout()
    save(fig, "fig6_roc_calibration")

# ============================================================== E2 harmonisation / ablation
R2, _ = load("harm")
if R2 is not None:
    t = R2.groupby(["rep", "harm", "model"], dropna=False)[["acc", "auc", "brier", "ece"]].agg(["mean", "std"])
    t.to_csv(os.path.join(OUT, "E2_ablation.csv"))

# ============================================================== E3 few-shot
R3, _ = load("fewshot")
R3u, _ = load("fewshot_u")
if R3 is not None:
    t = R3.groupby(["n_context", "model"])[["acc", "auc", "bacc"]].agg(["mean", "std", "count"])
    t.to_csv(os.path.join(OUT, "E3_fewshot.csv"))
    if R3u is not None:
        R3u.groupby(["n_context", "model"])[["acc", "auc", "bacc"]].agg(["mean", "std", "count"]).to_csv(os.path.join(OUT, "E3b_fewshot_semisup.csv"))
    panels = [(R3, "(a) PLS fitted on the labelled context only")] + ([(R3u, "(b) PCA-100 fitted on unlabelled training data")] if R3u is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.8), sharey=True, squeeze=False)
    for ax, (D, ttl) in zip(axes[0], panels):
        for m in [m for m in MODELS if m in D.model.unique()]:
            d = D[D.model == m].groupby("n_context")["auc"]
            mu, se = d.mean(), d.std() / np.sqrt(d.count())
            ax.plot(mu.index, mu.values, "-o", color=COL[m], lw=1.5, ms=4, label=m)
            ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=COL[m], alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_xticks([10, 20, 50, 100, 200, 400]); ax.set_xticklabels([10, 20, 50, 100, 200, 400])
        ax.axhline(0.5, color="#8a8984", lw=0.6); ax.set_xlabel("Labelled context size (class-balanced)"); ax.set_title(ttl, fontsize=8)
    axes[0][0].set_ylabel("ROC-AUC on held-out fold"); axes[0][-1].legend(fontsize=7, loc="lower right")
    save(fig, "fig3_fewshot")

# ============================================================== E4 LOSO
R4, _ = load("loso")
if R4 is not None:
    R4["harm"] = R4["harm"].fillna("none")
    t = R4.groupby(["harm", "rep", "model"])[["acc", "auc", "bacc"]].agg(["mean", "std", "count"])
    t.to_csv(os.path.join(OUT, "E4_loso_summary.csv"))
    R4.pivot_table(index=["site", "n_asd", "n_td"], columns=["harm", "model"], values="auc").to_csv(os.path.join(OUT, "E4_loso_per_site_auc.csv"))
    R4.pivot_table(index=["site", "n_asd", "n_td"], columns=["harm", "model"], values="acc").to_csv(os.path.join(OUT, "E4_loso_per_site_acc.csv"))
    sites = R4.groupby("site")[["n_asd", "n_td"]].first(); sites["n"] = sites.sum(1); sites = sites.sort_values("n")
    cfgs = [("LR-full", "full", "none"), ("XGBoost", "pca100", "none"), ("TabICL-v1", "pca100", "none"),
            ("TabICL-v2", "pca100", "none"), ("TabICL-v2", "plsauto", "none")]
    cfgs = [c for c in cfgs if len(R4[(R4.model == c[0]) & (R4.rep == c[1]) & (R4.harm == c[2])])]
    fig, ax = plt.subplots(figsize=(7.4, 3.4)); w = 0.8 / max(1, len(cfgs))
    for i, (m, rep, h) in enumerate(cfgs):
        d = R4[(R4.model == m) & (R4.rep == rep) & (R4.harm == h)].set_index("site").reindex(sites.index)
        lab = f"{m} ({'all edges' if rep == 'full' else rep.upper().replace('PLSAUTO', 'PLS-auto')})"
        ax.bar(np.arange(len(sites)) + (i - (len(cfgs) - 1) / 2) * w, d.auc, w * 0.9,
               color=COL[m], alpha=1.0 if rep != "plsauto" else 0.45, label=lab)
    ax.axhline(0.5, color="#8a8984", lw=0.8)
    nice = {"MAX_MUN": "MaxMun", "CALTECH": "Caltech", "LEUVEN": "Leuven", "STANFORD": "Stanford", "TRINITY": "Trinity", "YALE": "Yale", "OLIN": "Olin", "PITT": "Pitt"}
    ax.set_xticks(range(len(sites))); ax.set_xticklabels([f"{nice.get(s, s)} ({n})" for s, n in zip(sites.index, sites.n)], fontsize=6.5, rotation=40, ha="right")
    ax.set_ylabel("ROC-AUC on held-out site"); ax.set_ylim(0.3, 1.0)
    ax.legend(ncol=3, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    save(fig, "fig4_loso")

# ============================================================== E5 feature groups
R5, _ = load("groups")
if R5 is not None:
    t = R5.groupby(["group", "control", "model"])[["acc", "auc"]].agg(["mean", "std"])
    t.to_csv(os.path.join(OUT, "E5_groups.csv"))
    groups = ["Frontal", "Temporal", "Parietal", "Occipital", "Limbic", "Subcortical", "Cerebellum"]
    ms = [m for m in ["TabICL-v1", "TabICL-v2", "XGBoost", "LR"] if m in R5.model.unique()]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        real = R5[(R5.control == 0) & (R5.model == m)].groupby("group").auc.mean().reindex(groups)
        rnd = R5[(R5.control == 1) & (R5.model == m)].groupby("group").auc.mean().reindex(groups)
        x = np.arange(len(groups)) + (i - (len(ms) - 1) / 2) * w
        ax.bar(x, real, w * 0.9, color=COL[m], label=m)
        ax.scatter(x, rnd, marker="_", s=60, color="#0b0b0b", zorder=3, label="size-matched random edges" if i == 0 else None)
        wb = R5[(R5.group == "Whole-brain") & (R5.model == m)].auc.mean()
    ax.set_xticks(range(len(groups)))
    nedge = R5.groupby("group").n_edges.first()
    ax.set_xticklabels([f"{g}\n({nedge.get(g, 0):,} edges)" for g in groups], fontsize=7)
    ax.set_ylim(0.5, 0.82); ax.set_ylabel("ROC-AUC (5-fold CV)"); ax.legend(ncol=5, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_feature_groups")

# ============================================================== E6 permutation
R6, _ = load("perm")
if R6 is not None:
    rows = []
    for m, d in R6.groupby("model"):
        real = d[d.perm == -1]; null = d[d.perm >= 0]
        if not len(real) or not len(null):
            continue
        for met in ["acc", "auc"]:
            r = real[met].iloc[0]
            p = (1 + (null[met] >= r).sum()) / (len(null) + 1)
            rows.append(dict(model=m, metric=met, observed=r, null_mean=null[met].mean(), null_sd=null[met].std(),
                             n_perm=len(null), p_perm=p))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E6_permutation.csv"), index=False)

# ============================================================== E7 sample size
R7, _ = load("samplesize")
if R7 is not None:
    t = R7.groupby(["N", "rep", "model"])[["acc", "auc"]].agg(["mean", "std", "count"])
    t.to_csv(os.path.join(OUT, "E7_samplesize.csv"))

# ============================================================== E8 ADHD
R8, _ = load("adhd")
if R8 is not None:
    R8["kind"] = np.where(R8.tag.str.startswith("loso"), "loso", "cv")
    t = R8.groupby(["kind", "rep", "model"])[["acc", "auc", "bacc"]].agg(["mean", "std", "count"])
    t.to_csv(os.path.join(OUT, "E8_adhd.csv"))
R9, _ = load("adhd_perm")
if R9 is not None:
    rows = []
    for m, d in R9.groupby("model"):
        real = d[d.perm == -1]; null = d[d.perm >= 0]
        if len(real) and len(null):
            for met in ["acc", "auc"]:
                r = real[met].iloc[0]
                rows.append(dict(model=m, metric=met, observed=r, null_mean=null[met].mean(), null_sd=null[met].std(),
                                 n_perm=len(null), p_perm=(1 + (null[met] >= r).sum()) / (len(null) + 1)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E8_adhd_permutation.csv"), index=False)

json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("analysis written to", OUT)
