"""Revision analysis: tables, figures and numbers for the revised manuscript (supervisor comments, 29 Oct 2026).
Usage: python rev_analyze.py <results_all_dir> <rev_results_dir> <out_dir>
Primary model: TabICL (v1). TabICLv2 = sensitivity model. All paired tests use matched splits only."""
import sys, os, json, glob
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD, NEW, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FIG = os.path.join(OUT, "figs"); os.makedirs(FIG, exist_ok=True)
NUM = {}
LAB = {"TabICL-v1": "TabICL", "TabICL-v2": "TabICLv2"}
MODELS = ["LR-full", "LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]
COL = dict(zip(MODELS, ["#52514e", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#eb6834", "#6250d6"]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "axes.axisbelow": True, "axes.grid.axis": "y", "savefig.dpi": 300, "legend.frameon": False})
lab = lambda m: LAB.get(m, m)


def dirs(*names):
    out = []
    for n in names:
        out += sorted(glob.glob(os.path.join(OLD, n))) + sorted(glob.glob(os.path.join(NEW, n)))
    return out


def load(name, ds):
    fr, pr = [], []
    for d in ds:
        f = os.path.join(d, f"{name}.csv"); p = os.path.join(d, f"{name}_preds.csv")
        if os.path.exists(f): fr.append(pd.read_csv(f))
        if os.path.exists(p): pr.append(pd.read_csv(p))
    R = pd.concat(fr).drop_duplicates("key", keep="last") if fr else None
    P = pd.concat(pr).drop_duplicates(["key", "idx"], keep="last") if pr else None
    if R is not None and "harm" in R: R["harm"] = R["harm"].fillna("none").replace({"None": "none"})
    return R, P


def fauc(y, p):
    y = np.asarray(y); r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fbacc(y, p):
    y = np.asarray(y); h = np.asarray(p) >= .5
    return 0.5 * (h[y == 1].mean() + (~h[y == 0]).mean())


def boot(y, p, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p); n = len(y)
    return np.percentile([fn(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(B))], [2.5, 97.5])


def nb(a, b, n_tr, n_te):
    d = np.asarray(a) - np.asarray(b); J = len(d); v = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v) if v > 0 else np.inf
    return d.mean(), 2 * stats.t.sf(abs(t), J - 1)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


def f3(x): return f"{x:.3f}"
def pm(m, s, pct=False): return f"{100 * m:.1f} $\\pm$ {100 * s:.1f}" if pct else f"{m:.3f} $\\pm$ {s:.3f}"
def ptex(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, lines):
    open(os.path.join(OUT, name), "w").write("\n".join(lines) + "\n")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


# =========================================================== E1 main benchmark (ABIDE-I)
R, P = load("main", dirs("cloud", "pc_w1", "pc_w2b", "rev_w1"))
R = R[R.rep.isin(["pca20", "pca100", "plsauto", "plsautocf", "full"])]
P = P.merge(R[["key", "rep", "model"]], on="key")
g = R.groupby(["rep", "model"])
S = g[["acc", "bacc", "auc", "brier", "ece"]].agg(["mean", "std"]); S["J"] = g.size()
pooled = {}
for (rep, m), d in P.groupby(["rep", "model"]):
    a = d.groupby("idx").agg(y=("y", "first"), p=("p", "mean"))
    pooled[(rep, m)] = (fauc(a.y.values, a.p.values), *boot(a.y.values, a.p.values, fauc))
NUM["E1_J"] = {"|".join(k): int(v) for k, v in S["J"].items()}


def srow(rep, m, met="auc"): return S.loc[(rep, m), (met, "mean")]


order = [("full", ["LR-full"]), ("pca20", ["LR", "RF", "XGBoost", "TabICL-v1", "TabICL-v2"]),
         ("pca100", ["LR", "RF", "XGBoost", "TabICL-v1", "TabICL-v2"]),
         ("plsauto", ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"])]
repname = {"full": "All 19,900 edges", "pca20": "PCA-20", "pca100": "PCA-100", "plsauto": "PLS-auto", "plsautocf": "PLS-auto (cross-fit)"}
L = [r"\begin{table*}[!htbp]", r"\centering",
     r"\caption{E1 -- Main benchmark on ABIDE-I ($N=873$; the same 30 splits of 3$\times$10-fold cross-validation for every configuration; mean $\pm$ SD over splits). Pooled AUC: out-of-fold probabilities averaged over the three repeats, with 95\% bootstrap CI. TabICL is the primary model and TabICLv2 a sensitivity model. Best AUC per representation in bold (ties all bold). Cross-fitted PLS-auto and further metrics: Supplementary Table~S2.}",
     r"\label{tab:main}", r"\small", r"\begin{tabular}{llcccccc}", r"\toprule",
     r"Input & Model & $J$ & Accuracy (\%) & Bal.\ acc.\ (\%) & AUC & Pooled AUC [95\% CI] & ECE \\", r"\midrule"]
for rep, ms in order:
    best = max(srow(rep, m) for m in ms)
    for i, m in enumerate(ms):
        a = pm(S.loc[(rep, m), ("auc", "mean")], S.loc[(rep, m), ("auc", "std")])
        if rep != "full" and round(srow(rep, m), 3) == round(best, 3): a = r"\textbf{" + a + "}"
        pa = pooled[(rep, m)]
        L.append(f"{repname[rep] if i == 0 else ''} & {lab(m)} & {S.loc[(rep, m), 'J'].iloc[0] if hasattr(S.loc[(rep, m), 'J'], 'iloc') else S.loc[(rep, m), 'J']} & "
                 f"{pm(S.loc[(rep, m), ('acc', 'mean')], S.loc[(rep, m), ('acc', 'std')], True)} & {pm(S.loc[(rep, m), ('bacc', 'mean')], S.loc[(rep, m), ('bacc', 'std')], True)} & "
                 f"{a} & {pa[0]:.3f} [{pa[1]:.3f}, {pa[2]:.3f}] & {S.loc[(rep, m), ('ece', 'mean')]:.3f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table*}"]
write("tab_main.tex", L)
# supplementary full table (incl. cross-fit, Brier, sensitivity)
L = [r"\begin{table*}[!htbp]", r"\centering", r"\caption{Supplementary Table S2. E1 -- all configurations including cross-fitted PLS-auto (mean $\pm$ SD over the same 30 splits).}",
     r"\label{tab:s_main}", r"\small", r"\begin{tabular}{llccccc}", r"\toprule",
     r"Input & Model & $J$ & Accuracy (\%) & AUC & Brier & ECE \\", r"\midrule"]
for rep, ms in order + [("plsautocf", ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"])]:
    for i, m in enumerate(ms):
        L.append(f"{repname[rep] if i == 0 else ''} & {lab(m)} & {int(S.loc[(rep, m), 'J'])} & {pm(S.loc[(rep, m), ('acc', 'mean')], S.loc[(rep, m), ('acc', 'std')], True)} & "
                 f"{pm(S.loc[(rep, m), ('auc', 'mean')], S.loc[(rep, m), ('auc', 'std')])} & {S.loc[(rep, m), ('brier', 'mean')]:.3f} & {S.loc[(rep, m), ('ece', 'mean')]:.3f} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table*}"]
write("tab_s_main.tex", L)

# fold-matched paired tests
R["cfg"] = R.model + "@" + R.rep
piv = R.pivot_table(index="tag", columns="cfg", values="auc")
n_tr, n_te = R.n_train.median(), R.n_test.median()
pairs = [("TabICL-v1@plsauto", c) for c in ["LR-full@full", "LR@plsauto", "RF@plsauto", "XGBoost@plsauto", "TabM@plsauto", "TabICL-v2@plsauto"]] + \
        [("TabICL-v1@pca100", c) for c in ["LR@pca100", "RF@pca100", "XGBoost@pca100", "TabICL-v2@pca100"]] + \
        [("TabICL-v1@pca20", c) for c in ["LR@pca20", "RF@pca20", "XGBoost@pca20", "TabICL-v2@pca20"]] + \
        [("TabICL-v1@plsauto", "TabICL-v1@pca20"), ("TabICL-v1@plsauto", "TabICL-v1@pca100")]
T = []
for a, b in pairs:
    ok = piv[[a, b]].dropna().index
    d, p = nb(piv.loc[ok, a], piv.loc[ok, b], n_tr, n_te)
    T.append(dict(a=a, b=b, J=len(ok), d=d, p=p))
T = pd.DataFrame(T); T["p_holm"] = holm(T.p.values); T.to_csv(os.path.join(OUT, "E1_paired_tests.csv"), index=False)
nice = lambda c: f"{lab(c.split('@')[0])} ({repname[c.split('@')[1]]})"
L = [r"\begin{table}[!htbp]", r"\centering",
     r"\caption{Supplementary Table S1. Fold-matched comparisons on ABIDE-I (corrected resampled $t$-test \cite{nadeau2003inference}; $\Delta$AUC = first minus second configuration; $J$ = number of shared splits; Holm adjustment over all 16 comparisons). Only configurations evaluated on identical splits are compared.}",
     r"\label{tab:tests}", r"\small", r"\begin{tabular}{llcccc}", r"\toprule", r"Configuration & versus & $J$ & $\Delta$AUC & $p$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
for _, r in T.iterrows():
    L.append(f"{nice(r.a)} & {nice(r.b)} & {r.J} & {r.d:+.3f} & {ptex(r.p)} & {ptex(r.p_holm)} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("tab_tests.tex", L)
NUM["tests"] = T.round(4).to_dict("records")

pls = ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]
pca20 = ["LR", "RF", "XGBoost", "TabICL-v1", "TabICL-v2"]
gain = [round(srow("plsauto", m), 3) - round(srow("pca20", m), 3) for m in pca20]
gain100 = [round(srow("plsauto", m), 3) - round(srow("pca100", m), 3) for m in pca20]
NUM.update(GAIN100_LO=f3(min(gain100)), GAIN100_HI=f3(max(gain100)))
NUM.update(GAIN_LO=f3(min(gain)), GAIN_HI=f3(max(gain)),
           PCA20_LO=f3(min(srow("pca20", m) for m in pca20)), PCA20_HI=f3(max(srow("pca20", m) for m in pca20)),
           PLS_LO=f3(min(srow("plsauto", m) for m in pls)), PLS_HI=f3(max(srow("plsauto", m) for m in pls)),
           PLS_SPREAD=f3(max(srow("plsauto", m) for m in pls) - min(srow("plsauto", m) for m in pls)),
           T1_PLS_AUC=f3(srow("plsauto", "TabICL-v1")), T1_PLS_SD=f3(S.loc[("plsauto", "TabICL-v1"), ("auc", "std")]),
           T1_PLS_ACC=f"{100 * srow('plsauto', 'TabICL-v1', 'acc'):.1f}",
           T1_PLS_POOLED=f"{pooled[('plsauto', 'TabICL-v1')][0]:.3f}, 95\\% CI {pooled[('plsauto', 'TabICL-v1')][1]:.3f}--{pooled[('plsauto', 'TabICL-v1')][2]:.3f}",
           LRF_AUC=f3(srow("full", "LR-full")), LRF_ACC=f"{100 * srow('full', 'LR-full', 'acc'):.1f}",
           PLS_OTHERS=f"{min(srow('plsauto', m) for m in ['LR', 'RF', 'XGBoost', 'TabM']):.3f}--{max(srow('plsauto', m) for m in ['LR', 'RF', 'XGBoost', 'TabM']):.3f}",
           T2_PLS_AUC=f3(srow("plsauto", "TabICL-v2")),
           T1_PCA20_AUC=f3(srow("pca20", "TabICL-v1")), XGB_PCA20_AUC=f3(srow("pca20", "XGBoost")), RF_PCA20_AUC=f3(srow("pca20", "RF")),
           LR_PCA20_AUC=f3(srow("pca20", "LR")),
           T1_PCA100_AUC=f3(srow("pca100", "TabICL-v1")), XGB_PCA100_AUC=f3(srow("pca100", "XGBoost")), RF_PCA100_AUC=f3(srow("pca100", "RF")),
           LR_PCA100_AUC=f3(srow("pca100", "LR")),
           ECE_PLS_LO=f3(min(S.loc[("plsauto", m), ("ece", "mean")] for m in pls)), ECE_PLS_HI=f3(max(S.loc[("plsauto", m), ("ece", "mean")] for m in pls)),
           ECE_LRF=f3(S.loc[("full", "LR-full"), ("ece", "mean")]), ECE_T1_PLS=f3(S.loc[("plsauto", "TabICL-v1"), ("ece", "mean")]),
           ECE_T1_CF=f3(S.loc[("plsautocf", "TabICL-v1"), ("ece", "mean")]),
           CF_AUC_COST=f3(srow("plsauto", "TabICL-v1") - srow("plsautocf", "TabICL-v1")))
NUM["k_counts"] = R[R.rep == "plsauto"].drop_duplicates("tag").k.value_counts().to_dict()

# figure 2
reps = ["pca20", "pca100", "plsauto"]; models = ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, met, name in [(axes[0], "auc", "AUC"), (axes[1], "acc", "Accuracy")]:
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        mu = [S.loc[(r, m), (met, "mean")] if (r, m) in S.index else np.nan for r in reps]
        sd = [S.loc[(r, m), (met, "std")] if (r, m) in S.index else 0 for r in reps]
        x = np.arange(len(reps)) + (i - (len(models) - 1) / 2) * w
        ax.bar(x, mu, w * 0.9, yerr=sd, color=COL[m], label=lab(m) + (" (sensitivity)" if m == "TabICL-v2" else ""),
               error_kw=dict(lw=0.6, capsize=0, ecolor="#52514e"), alpha=0.55 if m == "TabICL-v2" else 1)
    ax.axhline(S.loc[("full", "LR-full"), (met, "mean")], color=COL["LR-full"], lw=1, ls="--", label="LR-full (all edges)")
    ax.set_xticks(range(3)); ax.set_xticklabels(["PCA-20", "PCA-100", "PLS-auto"]); ax.set_ylim(0.5, 0.82); ax.set_ylabel(name)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.13))
fig.tight_layout(w_pad=2.5); save(fig, "fig2_main_benchmark")

# calibration figure (supplementary S1)
def rel(ax, d, color, label, ls="-"):
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(d.p, bins) - 1, 0, 9)
    xs = [d.p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 15]; ys = [d.y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 15]
    ax.plot(xs, ys, ls, marker="o", color=color, lw=1.2, ms=3, label=label)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
for rep, c, nm in [("pca20", "#c9c3f0", "PCA-20"), ("pca100", "#6250d6", "PCA-100"), ("plsauto", "#eb6834", "PLS-auto"), ("plsautocf", "#1baf7a", "PLS-auto (cross-fit)")]:
    rel(axes[0], P[(P.rep == rep) & (P.model == "TabICL-v1")], c, nm)
for m in ["LR-full", "LR", "XGBoost", "TabM", "TabICL-v1"]:
    rep = "full" if m == "LR-full" else "plsauto"
    rel(axes[1], P[(P.rep == rep) & (P.model == m)], COL[m], lab(m))
axes[0].set_title("(a) TabICL: effect of the input", fontsize=8); axes[1].set_title("(b) Models on PLS-auto (LR-full: all edges)", fontsize=8)
for ax in axes:
    ax.plot([0, 1], [0, 1], color="#8a8984", lw=0.6, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(ASD)"); ax.set_ylabel("Observed ASD fraction"); ax.legend(fontsize=6, loc="upper left")
fig.tight_layout(); save(fig, "figS1_calibration")

# =========================================================== demographic control
RD, _ = load("demo", dirs("rev_w3", "rev_w4"))
RM = load("adhdmixeddemo", dirs("rev_w8"))[0]
RXm = load("adhdmixed", dirs("adhd/adhd_w*"))[0]
dem = {}
L = [r"\begin{table}[!htbp]", r"\centering",
     r"\caption{Demographic control (AUC, mean $\pm$ SD over the same 30 splits of 3$\times$10-fold CV). Age + sex: age and sex only; FC: PLS-auto connectivity scores only; FC + age + sex: both. ADHD-200: five mixed-diagnosis sites ($N=574$).}",
     r"\label{tab:demo}", r"\small", r"\begin{tabular}{lccc}", r"\toprule", r"Model & Age + sex & FC & FC + age + sex \\", r"\midrule",
     r"\multicolumn{4}{l}{\emph{ABIDE-I ($N=873$)}} \\"]
for m in ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]:
    if RD is None: break
    a = RD[(RD.rep == "agesex") & (RD.model == m)].auc; b = RD[(RD.rep == "plsauto+agesex") & (RD.model == m)].auc
    c = R[(R.rep == "plsauto") & (R.model == m)].auc
    dem[("abide", m)] = (a.mean(), c.mean(), b.mean(), len(a), len(b))
    L.append(f"{lab(m)} & {pm(a.mean(), a.std())} & {pm(c.mean(), c.std())} & {pm(b.mean(), b.std())} \\\\")
L.append(r"\midrule"); L.append(r"\multicolumn{4}{l}{\emph{ADHD-200, five mixed-diagnosis sites ($N=574$)}} \\")
for m in ["LR", "RF", "XGBoost", "TabICL-v1"]:
    if RM is None: break
    a = RM[(RM.rep == "agesex") & (RM.model == m)].auc; b = RM[(RM.rep == "plsauto+agesex") & (RM.model == m)].auc
    c = RXm[(RXm.rep == "plsauto") & (RXm.model == m)].auc
    dem[("adhdmixed", m)] = (a.mean(), c.mean(), b.mean(), len(a), len(b))
    L.append(f"{lab(m)} & {pm(a.mean(), a.std())} & {pm(c.mean(), c.std())} & {pm(b.mean(), b.std())} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("tab_demo.tex", L)
NUM["demo"] = {f"{k[0]}|{k[1]}": [round(v, 4) for v in vals] for k, vals in dem.items()}
ab = [v for k, v in dem.items() if k[0] == "abide"]; am = [v for k, v in dem.items() if k[0] == "adhdmixed"]
if ab:
    NUM["DEMO_AGESEX_RANGE"] = f"{min(v[0] for v in ab):.3f}--{max(v[0] for v in ab):.3f}"
    NUM["DEMO_DELTA_MAX"] = f"{max(abs(v[2] - v[1]) for v in ab):.3f}"
if am:
    NUM["AMD_AGESEX"] = f"{min(v[0] for v in am):.3f}--{max(v[0] for v in am):.3f}"
    NUM["AMD_DELTA"] = f"{min(v[2] - v[1] for v in am):+.3f} to {max(v[2] - v[1] for v in am):+.3f}"

# =========================================================== E2 few-shot (ABIDE-I)
F, _ = load("fewshot", dirs("cloud", "pc_w3", "rev_w2"))
Fu, _ = load("fewshot_u", dirs("cloud", "pc_w5", "rev_w2"))
sizes = [10, 20, 50, 100, 200, 400]
fsm = ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]
L = [r"\begin{table*}[!htbp]", r"\centering",
     r"\caption{E2 -- Few-shot label efficiency on ABIDE-I: AUC on fixed held-out folds (mean over 5 outer folds $\times$ 5 context draws; all models share the same test participants and context draws). Primary settings: $n\le200$; $n=400$ is an additional setting. Top: PLS fitted on the $n$ labeled participants. Bottom: PCA-100 fitted on all training-fold participants without labels. Best value per column within each block in bold (ties all bold).}",
     r"\label{tab:fewshot}", r"\small", r"\begin{tabular}{lcccccc}", r"\toprule",
     r"Model & $n=10$ & $n=20$ & $n=50$ & $n=100$ & $n=200$ & ($n=400$) \\", r"\midrule"]
fsnum = {}
for D, title, key in [(F, "Supervised PLS fitted on the labeled context", "sup"), (Fu, "Semi-supervised PCA-100 (unlabeled training data)", "semi")]:
    L.append(r"\multicolumn{7}{l}{\emph{" + title + r"}} \\")
    M = D.groupby(["model", "n_context"]).auc.mean().unstack()
    C = D.groupby(["model", "n_context"]).auc.size().unstack()
    fsnum[key] = M; fsnum[key + "_n"] = C
    for m in fsm:
        cells = []
        for n in sizes:
            v = M.loc[m, n] if (m in M.index and n in M.columns and not np.isnan(M.loc[m, n])) else None
            if v is None: cells.append("--"); continue
            s = f"{v:.3f}"
            if round(v, 3) == round(M[n].max(), 3): s = r"\textbf{" + s + "}"
            cells.append(s)
        L.append(f"{lab(m)} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table*}"]
write("tab_fewshot.tex", L)
Ms, Mu = fsnum["sup"], fsnum["semi"]
NUM["fewshot_counts_sup"] = fsnum["sup_n"].fillna(0).astype(int).to_dict()
NUM["fewshot_counts_semi"] = fsnum["semi_n"].fillna(0).astype(int).to_dict()
NUM.update(FS_SUP_200=f3(Ms[200].max()), FS_SUP_SPREAD=f3(max(Ms[n].max() - Ms[n].min() for n in [10, 20, 50, 100, 200])),
           FS_SEMI_T1_100=f3(Mu.loc["TabICL-v1", 100]), FS_SEMI_T1_200=f3(Mu.loc["TabICL-v1", 200]), FS_SEMI_T1_400=f3(Mu.loc["TabICL-v1", 400]),
           FS_SEMI_XGB_100=f3(Mu.loc["XGBoost", 100]), FS_SEMI_XGB_200=f3(Mu.loc["XGBoost", 200]))
if "TabM" in Mu.index:
    NUM.update(FS_SEMI_TABM_100=f3(Mu.loc["TabM", 100]), FS_SEMI_TABM_200=f3(Mu.loc["TabM", 200]))
best_sizes = [n for n in [10, 20, 50, 100, 200] if n in Mu.columns and not np.isnan(Mu.loc["TabICL-v1", n]) and round(Mu.loc["TabICL-v1", n], 3) >= round(Mu[n].max(), 3)]
NUM["FS_SEMI_BEST_SIZES"] = best_sizes
NUM["FS_SEMI_RANK"] = {int(n): Mu[n].rank(ascending=False).get("TabICL-v1") for n in Mu.columns}
# figure 3
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, D, ttl in [(axes[0], F, "(a) PLS fitted on the labeled context"), (axes[1], Fu, "(b) PCA-100 fitted on unlabeled training data")]:
    for m in [m for m in fsm if m in D.model.unique()]:
        d = D[D.model == m].groupby("n_context")["auc"]; mu, se = d.mean(), d.std() / np.sqrt(d.count())
        ax.plot(mu.index, mu.values, "-o", color=COL[m], lw=1.6 if m == "TabICL-v1" else 1.1, ms=3.5, label=lab(m),
                alpha=0.6 if m == "TabICL-v2" else 1)
        ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=COL[m], alpha=0.10, lw=0)
    ax.axvspan(300, 480, color="#f3f2ee", zorder=0); ax.text(380, 0.505, "additional", ha="center", fontsize=6, color="#52514e")
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(sizes); ax.set_xticklabels(sizes); ax.set_xlim(8, 480)
    ax.axhline(0.5, color="#8a8984", lw=0.6); ax.set_xlabel("Labeled context size (class-balanced)"); ax.set_title(ttl, fontsize=8)
axes[0].set_ylabel("AUC on held-out fold"); axes[1].legend(fontsize=6.5, loc="upper left")
fig.tight_layout(); save(fig, "fig3_fewshot")

# =========================================================== E3 LOSO (ABIDE-I) and ADHD LOSO with pooled estimates
def loso_table(name, ds, cohort):
    Rl, Pl = load(name, ds)
    Pl = Pl.copy(); Pl["site"] = Pl.key.str.split("|").str[0]; Pl["cfg"] = Pl.key.str.split("|", n=1).str[1]
    out = {}
    for cfg, gq in Pl.groupby("cfg"):
        rep, harm, m = cfg.split("|"); harm = "none" if harm == "None" else harm
        y, p = gq.y.values, gq.p.values
        ca = boot(y, p, fauc); cb = boot(y, p, fbacc)
        rs = Rl[(Rl.rep == rep) & (Rl.harm == harm) & (Rl.model == m)]
        out[(rep, harm, m)] = dict(mean=rs.auc.mean(), sd=rs.auc.std(), nsite=len(rs), pauc=fauc(y, p), lo=ca[0], hi=ca[1],
                                   pbacc=fbacc(y, p), blo=cb[0], bhi=cb[1])
    return Rl, out


Rl, LO = loso_table("loso", dirs("cloud", "pc_w2b", "pc_w3", "pc_w5", "pc_w7"), "ABIDE-I")
L = [r"\begin{table*}[!htbp]", r"\centering",
     r"\caption{E3 -- Site shift on ABIDE-I (17 sites). Mean site AUC: mean $\pm$ SD of the AUC over the held-out sites. Pooled: all participants' out-of-site predictions combined, with 95\% bootstrap CIs. Inductive LOSO uses no data of the held-out site; label-free target-site adaptation additionally $z$-scores every edge within each site, using the unlabeled features of the test site.}",
     r"\label{tab:loso}", r"\small", r"\begin{tabular}{llcccc}", r"\toprule",
     r"Input & Model & Mean site AUC & Pooled AUC [95\% CI] & Pooled bal.\ acc.\ (\%) [95\% CI] \\", r"\midrule"]
for harm, title in [("none", "Inductive LOSO"), ("sitez", "Label-free target-site adaptation (per-site $z$-scoring)")]:
    L.append(r"\multicolumn{5}{l}{\emph{" + title + r"}} \\")
    for rep, ms in [("full", ["LR-full"]), ("pca100", ["LR", "RF", "XGBoost", "TabICL-v1", "TabICL-v2"]), ("plsauto", ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"])]:
        for i, m in enumerate(ms):
            o = LO.get((rep, harm, m))
            if o is None: continue
            L.append(f"{repname[rep] if i == 0 else ''} & {lab(m)} & {pm(o['mean'], o['sd'])} & {o['pauc']:.3f} [{o['lo']:.3f}, {o['hi']:.3f}] & "
                     f"{100 * o['pbacc']:.1f} [{100 * o['blo']:.1f}, {100 * o['bhi']:.1f}] \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L = [l.replace(r"{llcccc}", r"{llccc}") for l in L] + [r"\end{tabular}", r"\end{table*}"]
write("tab_loso.tex", L)
NUM["LOSO"] = {"|".join(k): {kk: round(vv, 4) for kk, vv in v.items()} for k, v in LO.items()}
# figure 4
sites = Rl.groupby("site")[["n_asd", "n_td"]].first(); sites["n"] = sites.sum(axis=1); sites = sites.sort_values("n")
cfgs = [("LR-full", "full"), ("XGBoost", "pca100"), ("TabICL-v1", "pca100"), ("TabICL-v1", "plsauto")]
fig, ax = plt.subplots(figsize=(7.4, 3.3)); w = 0.8 / len(cfgs)
for i, (m, rep) in enumerate(cfgs):
    d = Rl[(Rl.model == m) & (Rl.rep == rep) & (Rl.harm == "none")].set_index("site").reindex(sites.index)
    ax.bar(np.arange(len(sites)) + (i - (len(cfgs) - 1) / 2) * w, d.auc, w * 0.9, color=COL[m], alpha=0.5 if rep == "plsauto" else 1,
           label=f"{lab(m)} ({'all edges' if rep == 'full' else repname[rep]})")
ax.axhline(0.5, color="#8a8984", lw=0.8)
nicen = {"MAX_MUN": "MaxMun", "CALTECH": "Caltech", "LEUVEN": "Leuven", "STANFORD": "Stanford", "TRINITY": "Trinity", "YALE": "Yale", "OLIN": "Olin", "PITT": "Pitt"}
ax.set_xticks(range(len(sites))); ax.set_xticklabels([f"{nicen.get(s, s)} ({n})" for s, n in zip(sites.index, sites.n)], fontsize=6.5, rotation=40, ha="right")
ax.set_ylabel("AUC on held-out site"); ax.set_ylim(0.3, 1.0); ax.legend(ncol=4, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); save(fig, "fig4_loso")

# =========================================================== E4 feature groups
G, _ = load("groups", dirs("cloud", "pc_w4"))
gm = ["LR", "XGBoost", "TabICL-v1", "TabICL-v2"]
def grp_table(groups, name, cap, label):
    L = [r"\begin{table}[!htbp]", r"\centering", cap, label, r"\small", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Group & Edges & LR & XGBoost & TabICL & TabICLv2 \\", r"\midrule"]
    ne = G.groupby("group").n_edges.first()
    for gname in groups:
        cells = []
        for m in gm:
            a = G[(G.group == gname) & (G.control == 0) & (G.model == m)].auc.mean()
            r = G[(G.group == gname) & (G.control == 1) & (G.model == m)].auc.mean()
            cells.append(f"{a:.3f}" if gname == "Whole-brain" or np.isnan(r) else f"{a:.3f} / {r:.3f}")
        L.append(f"{gname} & {int(ne.get(gname, 19900)):,} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]; write(name, L)
grp_table(["Whole-brain", "Frontal", "Temporal", "Limbic"], "tab_groups.tex",
          r"\caption{E4 -- Feature groups on ABIDE-I (edges with at least one endpoint in the group; PLS-auto; 5-fold CV). Cells: AUC of the anatomical set / a random edge set of the same size. All groups: Supplementary Table~S5.}", r"\label{tab:groups}")
grp_table(["Whole-brain", "Frontal", "Temporal", "Parietal", "Occipital", "Limbic", "Subcortical", "Cerebellum"], "tab_s_groups.tex",
          r"\caption{Supplementary Table S5. E4 -- all anatomical feature groups (AUC of the anatomical set / size-matched random edges; PLS-auto; 5-fold CV).}", r"\label{tab:s_groups}")
NUM["GRP_T1_TEMP"] = f3(G[(G.group == "Temporal") & (G.control == 0) & (G.model == "TabICL-v1")].auc.mean())
NUM["GRP_T1_TEMP_RND"] = f3(G[(G.group == "Temporal") & (G.control == 1) & (G.model == "TabICL-v1")].auc.mean())
NUM["GRP"] = {f"{gname}|{m}|{c}": round(G[(G.group == gname) & (G.control == c) & (G.model == m)].auc.mean(), 4)
              for gname in G.group.unique() for m in gm for c in [0, 1]}
groups4 = ["Whole-brain", "Frontal", "Temporal", "Limbic"]
fig, ax = plt.subplots(figsize=(6.0, 2.8)); w = 0.8 / len(gm); ne = G.groupby("group").n_edges.first()
for i, m in enumerate(gm):
    real = G[(G.control == 0) & (G.model == m)].groupby("group").auc.mean().reindex(groups4)
    rnd = G[(G.control == 1) & (G.model == m)].groupby("group").auc.mean().reindex(groups4)
    x = np.arange(len(groups4)) + (i - (len(gm) - 1) / 2) * w
    ax.bar(x, real, w * 0.9, color=COL[m], label=lab(m), alpha=0.55 if m == "TabICL-v2" else 1)
    ax.scatter(x, rnd, marker="_", s=70, color="#0b0b0b", zorder=3, label="random edges, same size" if i == 0 else None)
ax.set_xticks(range(len(groups4))); ax.set_xticklabels([f"{gname}\n({int(ne.get(gname, 19900)):,} edges)" for gname in groups4], fontsize=7)
ax.set_ylim(0.6, 0.8); ax.set_ylabel("AUC (5-fold CV)"); ax.legend(ncol=5, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(); save(fig, "fig5_feature_groups")

# =========================================================== permutation tests
def perm_summary(name, ds):
    Rp, _ = load(name, ds); out = {}
    for m, d in Rp.groupby("model"):
        obs = d[d.perm == -1]; null = d[d.perm >= 0].drop_duplicates("perm")
        if not len(obs) or not len(null): continue
        o = obs.auc.iloc[0]
        out[m] = dict(B=len(null), obs=o, null_mean=null.auc.mean(), null_sd=null.auc.std(),
                      p=(1 + (null.auc >= o).sum()) / (len(null) + 1))
    return Rp, out
RPa, PA = perm_summary("perm", dirs("cloud", "pc_w4", "rev_w1", "rev_w5", "rev_w6"))
RPg, PG = perm_summary("adhdfull_perm", dirs("adhd/adhd_w*", "rev_w2", "rev_w5", "rev_w6", "rev_w7"))
RPm, PM = perm_summary("adhdmixed_perm", dirs("adhd/adhd_w*", "rev_w2", "rev_w3", "rev_w4"))
NUM["PERM_ABIDE"] = PA; NUM["PERM_ADHD_GLOBAL"] = PG; NUM["PERM_ADHD_MIXED"] = PM
L = [r"\begin{table}[!htbp]", r"\centering",
     r"\caption{Supplementary Table S7. Permutation tests of the AUC (complete pipeline refitted under 5-fold CV for every permutation; ABIDE-I: PLS-auto; ADHD-200: PLS-5). Site-respecting: labels permuted within each of the five mixed-diagnosis ADHD-200 sites. $B$ = number of permutations; $p=(1+\#\{T_b\ge T_{\mathrm{obs}}\})/(B+1)$; TabICL and TabICLv2 used 8 ensemble members. Observed AUCs differ from the main tables because the permutation pipeline uses 5-fold CV, a fixed $k$ for ADHD-200 and 8 ensemble members. Permutation runs were stopped at a fixed compute budget and all completed permutations are reported; $^{\dagger}$minimum attainable $p$ for the given $B$.}",
     r"\label{tab:s_perm}", r"\small", r"\begin{tabular}{llccccc}", r"\toprule",
     r"Test & Model & $B$ & Observed AUC & Null mean $\pm$ SD & $p$ \\", r"\midrule"]
for title, D in [("ABIDE-I, global", PA), ("ADHD-200 (7 sites), global", PG), ("ADHD-200 (5 sites), site-respecting", PM)]:
    for i, m in enumerate([m for m in ["LR", "XGBoost", "TabICL-v1", "TabICL-v2"] if m in D]):
        o = D[m]
        L.append(f"{title if i == 0 else ''} & {lab(m)} & {o['B']} & {o['obs']:.3f} & {o['null_mean']:.3f} $\\pm$ {o['null_sd']:.3f} & {o['p']:.3f}{'$^{\\dagger}$' if abs(o['p'] - 1/(o['B']+1)) < 1e-12 else ''} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L = [l.replace("{llccccc}", "{llcccc}") for l in L] + [r"\end{tabular}", r"\end{table}"]
write("tab_s_perm.tex", L)

# =========================================================== E5 ADHD-200
meta_maj = 460 / 716
RA, PAd = load("adhdfull", dirs("adhd/adhd_w*", "rev_w1"))
RX, PX = load("adhdmixed", dirs("adhd/adhd_w*"))
def adhd_cv_table(Rq, Pq, fname, cap, label, permD=None, reps=None):
    Pq = Pq.merge(Rq[["key", "rep", "model"]], on="key")
    g = Rq.groupby(["rep", "model"]); St = g[["acc", "bacc", "auc", "ece"]].agg(["mean", "std"]); St["J"] = g.size()
    L = [r"\begin{table*}[!htbp]" if permD else r"\begin{table}[!htbp]", r"\centering", cap, label, r"\small",
         r"\begin{tabular}{llcccccc}" if permD else r"\begin{tabular}{llccccc}", r"\toprule",
         r"Input & Model & $J$ & Acc.\ (\%) & Bal.\ acc.\ (\%) & AUC & Pooled AUC [95\% CI]" + (r" & Perm.\ $p$ ($B$)" if permD else "") + r" \\", r"\midrule"]
    out = {}
    for rep, ms in reps:
        for i, m in enumerate(ms):
            if (rep, m) not in St.index: continue
            d = Pq[(Pq.rep == rep) & (Pq.model == m)].groupby("idx").agg(y=("y", "first"), p=("p", "mean"))
            pa = (fauc(d.y.values, d.p.values), *boot(d.y.values, d.p.values, fauc)); out[(rep, m)] = (St.loc[(rep, m), ("auc", "mean")], pa)
            extra = ""
            if permD is not None:
                extra = f" & {ptex(permD[m]['p'])} ({permD[m]['B']})" if (rep == "plsauto" and m in permD) else " & --"
            L.append(f"{repname.get(rep, rep).replace('19,900', '17,955') if i == 0 else ''} & {lab(m)} & {int(St.loc[(rep, m), 'J'])} & "
                     f"{pm(St.loc[(rep, m), ('acc', 'mean')], St.loc[(rep, m), ('acc', 'std')], True)} & {pm(St.loc[(rep, m), ('bacc', 'mean')], St.loc[(rep, m), ('bacc', 'std')], True)} & "
                     f"{pm(St.loc[(rep, m), ('auc', 'mean')], St.loc[(rep, m), ('auc', 'std')])} & {pa[0]:.3f} [{pa[1]:.3f}, {pa[2]:.3f}]{extra} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table*}" if permD else r"\end{table}"]
    write(fname, L); return St, out
StX, OX = adhd_cv_table(RX, PX, "tab_adhd_mixed.tex",
    r"\caption{E5a -- ADHD-200, five mixed-diagnosis sites ($N=574$; primary ADHD-200 analysis; 3$\times$10-fold CV, mean $\pm$ SD over the same 30 splits). Site-respecting permutation tests: Supplementary Table~S7.}",
    r"\label{tab:adhd_mixed}", None, [("full", ["LR-full"]), ("plsauto", ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"])])
StA, OA = adhd_cv_table(RA, PAd, "tab_adhd.tex",
    r"\caption{E5b -- ADHD-200, all seven sites ($N=716$, including the two control-only sites; 3$\times$10-fold CV; mean $\pm$ SD over splits). A constant `control' prediction reaches 64.2\% accuracy, so balanced accuracy and AUC are the informative metrics. Perm.\ $p$: global label permutation of the PLS-5 pipeline (number of permutations in brackets); because it also breaks the site--diagnosis association, it does not detect site confounding.}",
    r"\label{tab:adhd}", PG, [("full", ["LR-full"]), ("plsauto", ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]), ("pca100", ["LR", "RF", "XGBoost", "TabICL-v1", "TabICL-v2"])])
NUM["ADHD_MIXED"] = {f"{k[0]}|{k[1]}": [round(v[0], 4), [round(x, 4) for x in v[1]]] for k, v in OX.items()}
NUM["ADHD_ALL"] = {f"{k[0]}|{k[1]}": [round(v[0], 4), [round(x, 4) for x in v[1]]] for k, v in OA.items()}
NUM["ADHD_ALL_J"] = {"|".join(k): int(v) for k, v in StA["J"].items()}
NUM["AM_T1"] = f3(StX.loc[("plsauto", "TabICL-v1"), ("auc", "mean")])

# ADHD LOSO table with pooled
Rla, LA = loso_table("adhdfull_loso", dirs("adhd/adhd_w3"), "ADHD")
site_order = ["KKI", "NeuroIMAGE", "NYU", "OHSU", "Peking"]
L = [r"\begin{table*}[!htbp]", r"\centering",
     r"\caption{E5c -- Site shift on ADHD-200 (AUC on each held-out mixed-diagnosis site; models trained on all remaining participants, including the control-only sites). Mean: mean site AUC; pooled: all out-of-site predictions combined, with 95\% bootstrap CI. Adaptation: label-free per-site $z$-scoring.}",
     r"\label{tab:adhd_loso}", r"\small", r"\begin{tabular}{llcccccccc}", r"\toprule",
     r"Setting & Model & KKI & NeuroIMAGE & NYU & OHSU & Peking & Mean & Pooled AUC [95\% CI] & Pooled bal.\ acc.\ (\%) [95\% CI] \\", r"\midrule"]
_PL = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pooled_loso.csv"), keep_default_na=False); _PL = _PL[_PL.cohort.str.startswith("ADHD")]
def _pb(rep, harm, m):
    r = _PL[(_PL.rep == rep) & (_PL.harm == ("None" if harm == "none" else harm)) & (_PL.model == m)]
    return "--" if not len(r) else f"{100*r.bacc.iloc[0]:.1f} [{100*r.bacc_lo.iloc[0]:.1f}, {100*r.bacc_hi.iloc[0]:.1f}]"
for harm, title in [("none", "Inductive"), ("sitez", "Adaptation")]:
    for i, (rep, m) in enumerate([("full", "LR-full"), ("plsauto", "LR"), ("plsauto", "RF"), ("plsauto", "XGBoost"), ("plsauto", "TabM"), ("plsauto", "TabICL-v1"), ("plsauto", "TabICL-v2")]):
        d = Rla[(Rla.rep == rep) & (Rla.harm == harm) & (Rla.model == m)].set_index("site")
        o = LA[(rep, harm, m)]
        cells = [f"{d.auc.get(s, np.nan):.3f}" for s in site_order]
        L.append(f"{title if i == 0 else ''} & {lab(m)}{' (all edges)' if rep == 'full' else ''} & " + " & ".join(cells) + f" & {o['mean']:.3f} & {o['pauc']:.3f} [{o['lo']:.3f}, {o['hi']:.3f}] & {_pb(rep, harm, m)} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table*}"]
write("tab_adhd_loso.tex", L)
NUM["ADHD_LOSO"] = {"|".join(k): {kk: round(vv, 4) for kk, vv in v.items()} for k, v in LA.items()}

# ADHD few-shot
FA, _ = load("adhdfull_fewshot", dirs("adhd/adhd_w*"))
L = [r"\begin{table}[!htbp]", r"\centering",
     r"\caption{E5d -- Few-shot learning on ADHD-200 (all seven sites; AUC on fixed held-out folds, 5 outer folds $\times$ 2 shared context draws). Because these runs include the two control-only sites, only relative model comparisons are interpretable. $n=400$ is an additional setting. Best value per row in bold (ties all bold).}",
     r"\label{tab:adhd_fewshot}", r"\small", r"\begin{tabular}{llcccccc}", r"\toprule",
     r"Input & $n$ & LR & RF & XGBoost & TabM & TabICL & TabICLv2 \\", r"\midrule"]
FA["input"] = np.where(FA.rep.str.startswith("pca100"), "Semi-sup.\\ PCA-100", "Supervised PLS")
MA = FA.groupby(["input", "n_context", "model"]).auc.mean().unstack()
for inp in ["Supervised PLS", "Semi-sup.\\ PCA-100"]:
    for n in [20, 50, 100, 200, 400]:
        if (inp, n) not in MA.index: continue
        r = MA.loc[(inp, n)]; best = r.max()
        cells = [(r"\textbf{" + f"{r[m]:.3f}" + "}" if round(r[m], 3) == round(best, 3) else f"{r[m]:.3f}") if m in r and not np.isnan(r[m]) else "--"
                 for m in ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]]
        L.append(f"{inp} & {n if n != 400 else '(400)'} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table}"]
write("tab_adhd_fewshot.tex", L)

# ADHD results figure
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
a = axes[0, 0]; ms_ = ["LR-full", "LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]
for j, (O, nm, mk) in enumerate([(OX, "5 mixed sites (N = 574)", "o"), (OA, "7 sites (N = 716)", "s")]):
    for i, m in enumerate(ms_):
        k = ("full", m) if m == "LR-full" else ("plsauto", m)
        if k not in O: continue
        pa = O[k][1]
        a.errorbar(i + (j - 0.5) * 0.3, pa[0], yerr=[[pa[0] - pa[1]], [pa[2] - pa[0]]], fmt=mk, color=COL[m], ms=4, lw=1,
                   mfc="white" if j == 1 else COL[m], label=nm if i == 1 else None)
a.set_xticks(range(len(ms_))); a.set_xticklabels([lab(m) for m in ms_], rotation=30, ha="right", fontsize=7)
a.axhline(0.5, color="#8a8984", lw=0.6); a.set_ylabel("Pooled AUC [95% CI]"); a.set_ylim(0.5, 0.78)
a.set_title("(a) Cross-validation (PLS-auto; LR-full: all edges)", fontsize=8); a.legend(fontsize=6.3, loc="upper left")
a = axes[0, 1]
for m in ["LR", "XGBoost", "TabM", "TabICL-v1"]:
    for inp, ls in [("Supervised PLS", "-"), ("Semi-sup.\\ PCA-100", ":")]:
        d = FA[(FA.model == m) & (FA.input == inp)].groupby("n_context").auc.mean()
        if len(d): a.plot(d.index, d.values, ls, marker="o", ms=3, color=COL[m], label=lab(m) if ls == "-" else None)
a.set_xscale("log"); a.minorticks_off(); a.set_xticks([20, 50, 100, 200, 400]); a.set_xticklabels([20, 50, 100, 200, 400]); a.axhline(0.5, color="#8a8984", lw=0.6)
a.set_xlabel("Labeled context size"); a.set_ylabel("AUC"); a.set_title("(b) Few-shot (solid: PLS; dotted: semi-sup. PCA-100)", fontsize=8); a.legend(fontsize=6.3)
a = axes[1, 0]; w = 0.8 / 4
for i, m in enumerate(["LR", "XGBoost", "TabM", "TabICL-v1"]):
    d = Rla[(Rla.rep == "plsauto") & (Rla.harm == "none") & (Rla.model == m)].set_index("site").reindex(site_order)
    a.bar(np.arange(5) + (i - 1.5) * w, d.auc, w * 0.9, color=COL[m], label=lab(m))
a.axhline(0.5, color="#8a8984", lw=0.8); a.set_xticks(range(5)); a.set_xticklabels(site_order, fontsize=7); a.set_ylim(0.2, 0.8)
a.set_ylabel("AUC on held-out site"); a.set_title("(c) Inductive LOSO, mixed sites (PLS-auto)", fontsize=8); a.legend(fontsize=6.3, ncol=2)
a = axes[1, 1]
for m in ["LR", "TabICL-v1"]:
    for Rsrc, filled, ls in [(RPm, True, "-"), (RPg, False, "--")]:
        d = Rsrc[(Rsrc.model == m) & (Rsrc.perm >= 0)].drop_duplicates("perm").auc
        if len(d):
            a.hist(d, bins=30, density=True, color=COL[m], alpha=0.35 if filled else 1, histtype="stepfilled" if filled else "step", lw=1)
        o = Rsrc[(Rsrc.model == m) & (Rsrc.perm == -1)].auc
        if len(o): a.axvline(o.iloc[0], color=COL[m], ls=ls, lw=1.2, label=f"{lab(m)} {'site-respecting' if filled else 'global'}")
a.set_xlabel("AUC"); a.set_ylabel("density"); a.set_title("(d) Permutation nulls (lines: observed AUC)", fontsize=8); a.legend(fontsize=6)
fig.tight_layout(); save(fig, "fig_adhd_results")

# seven-site ADHD demographic control (supplementary)
RD7 = load("adhddemo", dirs("rev_w3"))[0]
if RD7 is not None and len(RD7):
    L = [r"\begin{table}[!htbp]", r"\centering", r"\caption{Supplementary Table S6. Demographic control on all seven ADHD-200 sites ($N=716$; AUC, mean $\pm$ SD over 3$\times$10-fold CV).}",
         r"\label{tab:s_demo}", r"\small", r"\begin{tabular}{lccc}", r"\toprule", r"Model & Age + sex & FC & FC + age + sex \\", r"\midrule"]
    for m in ["LR", "XGBoost", "TabICL-v1"]:
        a = RD7[(RD7.rep == "agesex") & (RD7.model == m)].auc; b = RD7[(RD7.rep == "plsauto+agesex") & (RD7.model == m)].auc
        c = RA[(RA.rep == "plsauto") & (RA.model == m)].auc
        if not len(a): continue
        L.append(f"{lab(m)} & {pm(a.mean(), a.std())} & {pm(c.mean(), c.std())} & {pm(b.mean(), b.std())} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]; write("tab_s_demo.tex", L)

NUM["PERM_T1_B"] = PA.get("TabICL-v1", {}).get("B")
json.dump(NUM, open(os.path.join(OUT, "numbers.json"), "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print("done")
