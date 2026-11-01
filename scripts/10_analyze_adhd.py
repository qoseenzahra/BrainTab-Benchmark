"""Aggregate the full ADHD-200 experiments (E8a-e), write tables, figure and a summary json.
Usage: python scripts/10_analyze_adhd.py <analysis_out_dir> <paper_dir> <results_dir_1> [...]"""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

OUT, PAPER, DIRS = sys.argv[1], sys.argv[2], sys.argv[3:]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
MODELS = ["LR-full", "LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]
COL = dict(zip(MODELS, ["#52514e", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#eb6834", "#6250d6"]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "axes.axisbelow": True, "axes.grid.axis": "y", "savefig.dpi": 300, "legend.frameon": False})
S = {}


def load(name):
    fr, pr = [], []
    for d in DIRS:
        f = os.path.join(d, f"{name}.csv"); p = os.path.join(d, f"{name}_preds.csv")
        if os.path.exists(f): fr.append(pd.read_csv(f))
        if os.path.exists(p): pr.append(pd.read_csv(p))
    R = pd.concat(fr).drop_duplicates("key", keep="first") if fr else None
    P = pd.concat(pr).drop_duplicates(["key", "idx"], keep="first") if pr else None
    return R, P


def nb_ttest(a, b, n_tr, n_te):
    d = np.asarray(a) - np.asarray(b); J = len(d); v = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v); return d.mean(), 2 * stats.t.sf(abs(t), J - 1)


def boot_auc(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2: v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


meta = pd.read_csv(os.path.join(ROOT, "data", "adhd_full_meta.csv"))
y_all = meta.y.values
# ------------------------------------------------------------- cohort table
g = meta.groupby("site").agg(N=("y", "size"), ADHD=("y", "sum"), age_m=("age", "mean"), age_s=("age", "std"),
                             male=("male", "mean"), vol=("n_tr", "median"))
g["TDC"] = g.N - g.ADHD
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{ADHD-200 cohort after quality control (Neuro Bureau Athena pipeline, CC200 parcellation with 190 non-empty parcels). Inclusion: Athena QC pass, known diagnosis (Brown excluded: diagnoses withheld), first resting-state run. Vol. = median number of fMRI volumes.}",
     r"\label{tab:cohort_adhd}", r"\small", r"\begin{tabular}{lrrrcrr}", r"\toprule",
     r"Site & $N$ & ADHD & TDC & Age (mean $\pm$ SD) & Female (\%) & Vol. \\", r"\midrule"]
for s, r in g.iterrows():
    L.append(f"{s} & {r.N} & {r.ADHD} & {r.TDC} & {r.age_m:.1f} $\\pm$ {r.age_s:.1f} & {100 * (1 - r.male):.0f} & {r.vol:.0f} \\\\")
L += [r"\midrule", f"Total & {len(meta)} & {int(meta.y.sum())} & {int((1 - meta.y).sum())} & {meta.age.mean():.1f} $\\pm$ {meta.age.std():.1f} & {100 * (1 - meta.male.mean()):.0f} & {meta.n_tr.median():.0f} \\\\",
      r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(PAPER, "tab_cohort_adhd.tex"), "w").write("\n".join(L) + "\n")
S["cohort"] = dict(N=len(meta), adhd=int(meta.y.sum()), tdc=int((1 - meta.y).sum()), sites=int(meta.site.nunique()),
                   age_mean=float(meta.age.mean()), age_sd=float(meta.age.std()), age_min=float(meta.age.min()), age_max=float(meta.age.max()),
                   female_pct=float(100 * (1 - meta.male.mean())), majority_acc=float(max(meta.y.mean(), 1 - meta.y.mean())),
                   vol_min=int(meta.n_tr.min()), vol_max=int(meta.n_tr.max()))

# ------------------------------------------------------------- E8a main CV
R, P = load("adhdfull")
R["model"] = R.model.astype(str)
rows = []
for (rep, m), d in R.groupby(["rep", "model"]):
    keys = set(d.key)
    pp = P[P.key.isin(keys)]
    pooled = pp.groupby("idx").agg(y=("y", "first"), p=("p", "mean"))       # mean over repeats per participant
    lo, hi = boot_auc(pooled.y.values, pooled.p.values)
    rows.append(dict(rep=rep, model=m, n_folds=len(d), acc=d.acc.mean(), acc_sd=d.acc.std(), bacc=d.bacc.mean(), bacc_sd=d.bacc.std(),
                     auc=d.auc.mean(), auc_sd=d.auc.std(), sens=d.sens.mean(), spec=d.spec.mean(), ece=d.ece.mean(), brier=d.brier.mean(),
                     pooled_auc=roc_auc_score(pooled.y, pooled.p), ci_lo=lo, ci_hi=hi))
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "E8_adhdfull_summary.csv"), index=False)
print(T.round(3).to_string())
# corrected paired tests vs LR-full on matched folds (plsauto, 30 folds)
tests = []
ref = R[R.model == "LR-full"].set_index("tag")
for (rep, m), d in R[R.rep.isin(["plsauto", "pca100"])].groupby(["rep", "model"]):
    d = d.set_index("tag"); common = d.index.intersection(ref.index)
    if len(common) < 5: continue
    for met in ["auc", "bacc"]:
        diff, p = nb_ttest(d.loc[common, met], ref.loc[common, met], 644, 72)
        tests.append(dict(rep=rep, model=m, metric=met, n=len(common), diff=diff, p=p))
Tt = pd.DataFrame(tests); Tt.to_csv(os.path.join(OUT, "E8_adhdfull_tests_vs_LRfull.csv"), index=False)

# ------------------------------------------------------------- E8c permutation
Rp, _ = load("adhdfull_perm")
perm = []
for m, d in Rp.groupby("model"):
    obs = d[d.perm == -1]; null = d[d.perm >= 0]
    if obs.empty or null.empty: continue
    o_auc, o_bacc = float(obs.auc.iloc[0]), float(obs.bacc.iloc[0])
    perm.append(dict(model=m, B=len(null), obs_auc=o_auc, obs_bacc=o_bacc, null_auc_mean=null.auc.mean(), null_auc_95=null.auc.quantile(0.95),
                     p_auc=(1 + (null.auc >= o_auc).sum()) / (1 + len(null)), p_bacc=(1 + (null.bacc >= o_bacc).sum()) / (1 + len(null))))
TP = pd.DataFrame(perm); TP.to_csv(os.path.join(OUT, "E8_adhdfull_permutation.csv"), index=False); print(TP.round(4).to_string())

# ------------------------------------------------------------- E8f within-site permutation (mixed sites)
Rq, _ = load("adhdmixed_perm"); MP = {}
if Rq is not None:
    for m, d in Rq.groupby("model"):
        obs = d[d.perm == -1]; null = d[d.perm >= 0]
        if obs.empty or null.empty: continue
        o = float(obs.auc.iloc[0])
        MP[m] = dict(B=int(len(null)), obs_auc=o, obs_bacc=float(obs.bacc.iloc[0]), null_auc_mean=float(null.auc.mean()), null_auc_sd=float(null.auc.std()),
                     null_auc_95=float(null.auc.quantile(0.95)), p_auc=float((1 + (null.auc >= o).sum()) / (1 + len(null))))
json.dump(MP, open(os.path.join(OUT, "E8_adhdmixed_perm.json"), "w"), indent=1); print("within-site perm", MP)

# ------------------------------------------------------------- E8b LOSO, E8e mixed sites
RL, _ = load("adhdfull_loso"); RL["harm"] = RL.harm.fillna("none")
RL.to_csv(os.path.join(OUT, "E8_adhdfull_loso_all.csv"), index=False)
RM, PM = load("adhdmixed")
mix = RM.groupby(["rep", "model"]).agg(n=("auc", "size"), acc=("acc", "mean"), bacc=("bacc", "mean"), auc=("auc", "mean"), auc_sd=("auc", "std")).reset_index()
mix.to_csv(os.path.join(OUT, "E8_adhdmixed_summary.csv"), index=False); print(mix.round(3).to_string())
S["mixed"] = dict(N=int(RM.n_train.iloc[0] + RM.n_test.iloc[0]))

# ------------------------------------------------------------- E8d few-shot
RF_, _ = load("adhdfull_fewshot"); RF_["rep2"] = np.where(RF_.rep == "pca100u", "semi", "pls")
fs = RF_.groupby(["rep2", "n_context", "model"]).agg(n=("auc", "size"), auc=("auc", "mean"), auc_sd=("auc", "std"), bacc=("bacc", "mean")).reset_index()
fs.to_csv(os.path.join(OUT, "E8_adhdfull_fewshot.csv"), index=False); print(fs.pivot_table(index=["rep2", "n_context"], columns="model", values="auc").round(3))

# ------------------------------------------------------------- tables
def g_(rep, m, col):
    r = T[(T.rep == rep) & (T.model == m)]
    return None if r.empty else r.iloc[0]


def pv(m):
    r = TP[TP.model == m]
    return "--" if r.empty else (f"{r.p_auc.iloc[0]:.3f}" if r.p_auc.iloc[0] >= 0.001 else "$<$0.001") + f" ({int(r.B.iloc[0])})"


L = [r"\begin{table}[t]", r"\centering",
     f"\\caption{{E8 -- full ADHD-200 cohort ($N={len(meta)}$, 7 sites; 3$\\times$10-fold stratified CV; mean $\\pm$ SD over folds). Pooled AUC: out-of-fold predictions averaged over repeats, with 95\\% bootstrap CI. Because only {100 * meta.y.mean():.0f}\\% of participants have ADHD, a constant `TDC' prediction reaches {100 * S['cohort']['majority_acc']:.1f}\\% accuracy; balanced accuracy (bACC) and AUC are therefore the primary metrics. Perm.\\ $p$: label-permutation test of the AUC (PLS-5 pipeline, 5-fold CV; number of permutations in brackets). TabICL on PCA-100 was evaluated on the first repeat (10 folds).}}",
     r"\label{tab:adhd}", r"\scriptsize", r"\begin{tabular}{llcccccc}", r"\toprule",
     r"Input & Model & Acc.\ (\%) & bACC (\%) & AUC & Pooled AUC [95\% CI] & ECE & Perm.\ $p$ \\", r"\midrule"]
for rep, lab in [("full", "All 17,955 edges"), ("plsauto", "PLS-auto"), ("pca100", "PCA-100")]:
    for m in MODELS:
        r = g_(rep, m, None)
        if r is None: continue
        L.append(f"{lab} & {m} & {100 * r.acc:.1f} $\\pm$ {100 * r.acc_sd:.1f} & {100 * r.bacc:.1f} $\\pm$ {100 * r.bacc_sd:.1f} & {r.auc:.3f} $\\pm$ {r.auc_sd:.3f} & {r.pooled_auc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] & {r.ece:.3f} & {pv(m) if rep == 'plsauto' else '--'} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table}"]
open(os.path.join(PAPER, "tab_adhd.tex"), "w").write("\n".join(L) + "\n")

# LOSO table: per-site AUC for each model (PLS-auto, raw vs site-z) + LR-full
sites = [s for s in ["KKI", "NeuroIMAGE", "NYU", "OHSU", "Peking"] if s in set(RL.site)]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{E8b -- leave-one-site-out on ADHD-200 (AUC on the held-out site). Only the five sites with both ADHD and control participants can be held out; models are trained on all remaining participants (including the control-only Pittsburgh and WashU samples). \emph{z}: label-free per-site $z$-scoring of the edges before compression.}",
     r"\label{tab:adhd_loso}", r"\scriptsize", r"\begin{tabular}{ll" + "c" * len(sites) + "c}", r"\toprule",
     "Input & Model & " + " & ".join(f"{s} ({int((meta.site == s).sum())})" for s in sites) + r" & Mean \\", r"\midrule"]
loso_rows = []
for harm, hl in [("none", ""), ("sitez", " + $z$")]:
    for rep, lab in [("full", "All edges"), ("plsauto", "PLS-auto")]:
        for m in MODELS:
            d = RL[(RL.harm == harm) & (RL.rep == rep) & (RL.model == m)].set_index("site")
            if d.empty: continue
            vals = [d.auc.get(s, np.nan) for s in sites]
            loso_rows.append(dict(harm=harm, rep=rep, model=m, mean_auc=np.nanmean(vals), mean_bacc=d.bacc.mean(), **{s: v for s, v in zip(sites, vals)}))
            L.append(f"{lab}{hl} & {m} & " + " & ".join(f"{v:.3f}" for v in vals) + f" & \\textbf{{{np.nanmean(vals):.3f}}} \\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table}"]
open(os.path.join(PAPER, "tab_adhd_loso.tex"), "w").write("\n".join(L) + "\n")
TL = pd.DataFrame(loso_rows); TL.to_csv(os.path.join(OUT, "E8_adhdfull_loso_summary.csv"), index=False); print(TL.round(3).to_string())

# few-shot table
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{E8d -- few-shot learning on ADHD-200: AUC (mean over 5 outer folds $\times$ 2 random context draws) as a function of the number of labelled participants $n$ (balanced classes). Supervised: PLS fitted on the $n$ labelled participants only (PLS-2 for $n=20$, PLS-auto otherwise). Semi-supervised: PCA-100 fitted without labels on all training-fold participants. Best value per row in bold.}",
     r"\label{tab:adhd_fewshot}", r"\scriptsize", r"\begin{tabular}{llcccccc}", r"\toprule",
     r"Input & $n$ & LR & RF & XGBoost & TabM & TabICL-v1 & TabICL-v2 \\", r"\midrule"]
for rep2, lab in [("pls", "Supervised PLS"), ("semi", "Semi-sup.\\ PCA-100")]:
    for n in sorted(fs.n_context.unique()):
        d = fs[(fs.rep2 == rep2) & (fs.n_context == n)].set_index("model")
        vals = {m: d.auc.get(m, np.nan) for m in ["LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]}
        best = np.nanmax(list(vals.values()))
        cells = ["--" if np.isnan(v) else (f"\\textbf{{{v:.3f}}}" if abs(v - best) < 1e-9 else f"{v:.3f}") for v in vals.values()]
        L.append(f"{lab} & {n} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table}"]
open(os.path.join(PAPER, "tab_adhd_fewshot.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw=dict(hspace=0.5, wspace=0.3))
a = ax[0, 0]; xs = []
for i, rep in enumerate(["plsauto", "pca100"]):
    for j, m in enumerate(MODELS):
        r = g_(rep, m, None)
        if r is None: continue
        x = i * 8 + j; xs.append(x)
        a.bar(x, r.pooled_auc, color=COL[m], width=0.8, label=m if i == 0 else None)
        a.plot([x, x], [r.ci_lo, r.ci_hi], color="#0b0b0b", lw=0.8)
rf = g_("full", "LR-full", None)
a.axhline(rf.pooled_auc, color=COL["LR-full"], ls="--", lw=0.9); a.text(15.2, rf.pooled_auc - 0.014, "LR-full", fontsize=6, color=COL["LR-full"], ha="right")
a.axhline(0.5, color="#8a8984", lw=0.6)
a.set_xticks([3, 11]); a.set_xticklabels(["PLS-auto", "PCA-100"]); a.set_ylim(0.45, 0.84); a.set_ylabel("pooled out-of-fold AUC")
a.set_title("(a) 3×10-fold CV, N = %d" % len(meta)); a.legend(fontsize=5.5, ncol=4, loc="upper center", columnspacing=0.8, handlelength=1.2)
a = ax[0, 1]
for m in ["LR", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]:
    for rep2, ls in [("pls", "-"), ("semi", ":")]:
        d = fs[(fs.rep2 == rep2) & (fs.model == m)].sort_values("n_context")
        if d.empty: continue
        a.plot(d.n_context, d.auc, ls=ls, marker="o", ms=2.5, color=COL[m], lw=1.1, label=m if rep2 == "pls" else None)
a.set_xscale("log"); a.minorticks_off(); a.set_xticks([20, 50, 100, 200, 400]); a.set_xticklabels([20, 50, 100, 200, 400]); a.axhline(0.5, color="#8a8984", lw=0.6)
a.set_xlabel("labelled participants in context (n)"); a.set_ylabel("AUC"); a.set_title("(b) few-shot (solid: PLS; dotted: semi-sup. PCA)")
a.legend(fontsize=5.5, loc="upper left")
a = ax[1, 0]
d = TL[(TL.rep == "plsauto")]
w = 0.12
for k, m in enumerate([m for m in MODELS if m in set(d.model)]):
    dd = d[(d.model == m) & (d.harm == "none")]
    if dd.empty: continue
    a.bar(np.arange(len(sites)) + (k - 2.5) * w, [dd[s].iloc[0] for s in sites], width=w, color=COL[m], label=m)
a.axhline(0.5, color="#8a8984", lw=0.6); a.set_xticks(range(len(sites))); a.set_xticklabels(sites, fontsize=6.5)
a.set_ylabel("AUC on held-out site"); a.set_ylim(0.3, 0.85); a.set_title("(c) leave-one-site-out (PLS-auto)")
a = ax[1, 1]
for (src_, ls_, lab_) in [(Rp, "global", "global"), (Rq, "within", "within-site")]:
    if src_ is None: continue
    for m, c in [("LR", COL["LR"]), ("TabICL-v2", COL["TabICL-v2"])]:
        d = src_[(src_.model == m) & (src_.perm >= 0)]
        if d.empty: continue
        a.hist(d.auc, bins=24 if m == "LR" else 10, color=c, alpha=0.45 if ls_ == "within" else 0.15, density=True,
               histtype="stepfilled" if ls_ == "within" else "step", lw=1, label=f"{m} {lab_} null (B={len(d)})")
        o = src_[(src_.model == m) & (src_.perm == -1)].auc
        if len(o): a.axvline(o.iloc[0], color=c, lw=1.4, ls="-" if ls_ == "within" else "--")
a.set_xlabel("AUC"); a.set_ylabel("density"); a.set_title("(d) permutation nulls; lines = observed\n(solid: within-site, mixed sites; dashed: global)", fontsize=7.5); a.legend(fontsize=5)
for e in ("png", "pdf"): fig.savefig(os.path.join(OUT, "figures", f"fig_adhd_results.{e}"), bbox_inches="tight")

# ------------------------------------------------------------- summary json
S["main"] = T.to_dict("records"); S["tests"] = Tt.to_dict("records"); S["perm"] = TP.to_dict("records")
S["loso"] = TL.to_dict("records"); S["mixed_summary"] = mix.to_dict("records"); S["fewshot"] = fs.to_dict("records")
json.dump(S, open(os.path.join(OUT, "E8_adhdfull_summary.json"), "w"), indent=1, default=float)
print("DONE")
