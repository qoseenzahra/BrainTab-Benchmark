"""Build LaTeX tables + a values.json of every number quoted in the manuscript.
Usage: python scripts/06_make_tables.py <analysis_dir> <paper_dir> <results_dir_1> [...]"""
import sys, os, json
import numpy as np, pandas as pd
ANA, PAPER, DIRS = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(PAPER, exist_ok=True)
REPN = {"full": "All 19,900 edges", "pca20": "PCA-20", "pca100": "PCA-100", "plsauto": "PLS-auto", "plsautocf": "PLS-auto (cross-fit)",
        "pls3": "PLS-3", "pls5": "PLS-5", "pls8": "PLS-8", "pls12": "PLS-12", "pls5cf": "PLS-5 (cross-fit)", "pca50": "PCA-50", "pca200": "PCA-200",
        "pls2": "PLS-2", "pca10": "PCA-10"}
MORDER = ["LR-full", "LR", "RF", "XGBoost", "TabM", "TabICL-v1", "TabICL-v2"]


def load(name):
    fr = [pd.read_csv(os.path.join(d, f"{name}.csv")) for d in DIRS if os.path.exists(os.path.join(d, f"{name}.csv"))]
    return pd.concat(fr).drop_duplicates("key", keep="last") if fr else None


def pm(x, s, pct=True, d=1):
    return (f"{100*x:.{d}f} $\\pm$ {100*s:.{d}f}" if pct else f"{x:.3f} $\\pm$ {s:.3f}")


def write(name, lines):
    txt = "\n".join(lines) + "\n"
    if name in ("tab_main", "tab_ablation", "tab_loso", "tab_groups"):   # keep wide tables inside the text block
        a = txt.index("\\begin{tabular}"); b = txt.index("\\end{tabular}") + len("\\end{tabular}")
        txt = txt[:a] + "\\resizebox{\\linewidth}{!}{%\n" + txt[a:b] + "}" + txt[b:]
    open(os.path.join(PAPER, name + ".tex"), "w").write(txt)


V = {}
# ---------------------------------------------------------------- E1
R = load("main")
R["model"] = R.model.astype(str)
g = R.groupby(["rep", "model"])
S = g[["acc", "bacc", "auc", "sens", "spec", "f1", "brier", "ece"]].agg(["mean", "std"]); S["n"] = g.size()
pooled = pd.read_csv(os.path.join(ANA, "E1_main_pooled_ci.csv")).set_index(["rep", "model"])
best_auc = S[("auc", "mean")].max(); best_acc = S[("acc", "mean")].max()
L = [r"\begin{table*}[t]", r"\centering", r"\caption{E1 -- Main benchmark on ABIDE-I ($N=" + str(int(R.n_train.iloc[0] + R.n_test.iloc[0])) + r"$), stratified 10-fold CV repeated 3 times. Fold-level mean $\pm$ SD (\%, except AUC, Brier and ECE). Pooled AUC: repetition-averaged out-of-fold probabilities with 95\% bootstrap CI. $J$ = number of train/test splits. Best value per column in bold.}", r"\label{tab:main}", r"\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
     r"\begin{tabular}{llrcccccccc}", r"\toprule", r"Input & Model & $J$ & Accuracy & Bal.\ acc. & Sensitivity & Specificity & AUC & Pooled AUC [95\% CI] & Brier & ECE \\", r"\midrule"]
for rep in ["full", "pca20", "pca100", "plsauto", "plsautocf"]:
    first = True
    for m in MORDER:
        if (rep, m) not in S.index:
            continue
        r = S.loc[(rep, m)]
        auc_s = f"{r[('auc','mean')]:.3f} $\\pm$ {r[('auc','std')]:.3f}"
        acc_s = pm(r[("acc", "mean")], r[("acc", "std")])
        if abs(r[("auc", "mean")] - best_auc) < 1e-12: auc_s = r"\textbf{" + auc_s + "}"
        if abs(r[("acc", "mean")] - best_acc) < 1e-12: acc_s = r"\textbf{" + acc_s + "}"
        pc = pooled.loc[(rep, m)] if (rep, m) in pooled.index else None
        pcs = f"{pc.pooled_auc:.3f} [{pc.auc_lo:.3f}, {pc.auc_hi:.3f}]" if pc is not None else "--"
        L.append(f"{REPN[rep] if first else ''} & {m} & {int(r['n'].iloc[0]) if hasattr(r['n'],'iloc') else int(r['n'])} & {acc_s} & {pm(r[('bacc','mean')], r[('bacc','std')])} & "
                 f"{pm(r[('sens','mean')], r[('sens','std')])} & {pm(r[('spec','mean')], r[('spec','std')])} & {auc_s} & {pcs} & {r[('brier','mean')]:.3f} & {r[('ece','mean')]:.3f} \\\\")
        first = False
        key = f"{m}@{rep}".replace("-", "").replace("@", "_")
        V[f"E1_{key}_acc"] = f"{100*r[('acc','mean')]:.1f}"; V[f"E1_{key}_accsd"] = f"{100*r[('acc','std')]:.1f}"
        V[f"E1_{key}_auc"] = f"{r[('auc','mean')]:.3f}"; V[f"E1_{key}_aucsd"] = f"{r[('auc','std')]:.3f}"
        V[f"E1_{key}_bacc"] = f"{100*r[('bacc','mean')]:.1f}"; V[f"E1_{key}_ece"] = f"{r[('ece','mean')]:.3f}"
        V[f"E1_{key}_brier"] = f"{r[('brier','mean')]:.3f}"; V[f"E1_{key}_sens"] = f"{100*r[('sens','mean')]:.1f}"; V[f"E1_{key}_spec"] = f"{100*r[('spec','mean')]:.1f}"
        V[f"E1_{key}_J"] = str(int(r['n'].iloc[0]) if hasattr(r['n'],'iloc') else int(r['n']))
        if pc is not None:
            V[f"E1_{key}_pauc"] = f"{pc.pooled_auc:.3f}"; V[f"E1_{key}_pauclo"] = f"{pc.auc_lo:.3f}"; V[f"E1_{key}_pauchi"] = f"{pc.auc_hi:.3f}"
            V[f"E1_{key}_pacc"] = f"{100*pc.pooled_acc:.1f}"
    L.append(r"\midrule" if rep != "plsautocf" else r"\bottomrule")
L += [r"\end{tabular}", r"\end{table*}"]
write("tab_main", L)
# paired tests
T = pd.read_csv(os.path.join(ANA, "E1_paired_tests.csv"))
T = T[T.ref == "TabICL-v2@pca100"].copy()
L = [r"\begin{table}[t]", r"\centering", r"\caption{E1 -- Fold-matched comparisons of TabICL-v2 on PCA-100 against every other configuration (corrected resampled $t$-test \cite{nadeau2003inference}; Holm-adjusted $p$). $\Delta$ = TabICL-v2@PCA-100 minus comparator.}", r"\label{tab:tests}", r"\scriptsize",
     r"\begin{tabular}{lrrrrr}", r"\toprule", r"Comparator & $J$ & $\Delta$AUC & $p_{\mathrm{Holm}}$ & $\Delta$Acc (pp) & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
def cfgname(c):
    m, r = c.split("@"); return f"{m} @ {REPN.get(r, r)}"
T["_o"] = T.vs.map(lambda c: (["full", "pca20", "pca100", "plsauto", "plsautocf"].index(c.split("@")[1]), MORDER.index(c.split("@")[0])))
for _, t in T.sort_values("_o").iterrows():
    fp = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    L.append(f"{cfgname(t.vs)} & {int(t.n_splits)} & {t.d_auc:+.3f} & {fp(t.p_auc_holm)} & {100*t.d_acc:+.1f} & {fp(t.p_acc_holm)} \\\\")
    V[f"T_{t.vs.replace('-', '').replace('@', '_')}_dauc"] = f"{t.d_auc:+.3f}"; V[f"T_{t.vs.replace('-', '').replace('@', '_')}_p"] = fp(t.p_auc_holm)
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("tab_tests", L)
summ = json.load(open(os.path.join(ANA, "summary.json")))
V["k_counts"] = ", ".join(f"$k={k}$: {v}" for k, v in sorted(summ.get("plsauto_k_counts", {}).items(), key=lambda kv: float(kv[0])))
json.dump(V, open(os.path.join(PAPER, "values_E1.json"), "w"), indent=1)
print(len(V), "values")

# ---------------------------------------------------------------- E2 ablations
R2 = load("harm")
if R2 is not None:
    R2["harm"] = R2.harm.fillna("none")
    a = R2.groupby(["rep", "harm", "model"])[["acc", "auc", "brier", "ece"]].agg(["mean", "std"])
    order = [("pls3", "none"), ("pls5", "none"), ("pls8", "none"), ("pls12", "none"), ("plsauto", "combat"), ("pls5cf", "none"), ("pca50", "none"), ("pca200", "none"), ("full", "combat")]
    ms = [m for m in MORDER if m in R2.model.unique()]
    L = [r"\begin{table*}[t]", r"\centering", r"\caption{E2 -- Ablations (one 10-fold repetition). Cells: AUC (mean over folds) / ECE. ComBat was fitted on training folds only (covariates age and sex). TabICL and TabM were not run on wide PCA inputs (cost).}", r"\label{tab:ablation}", r"\scriptsize",
         r"\begin{tabular}{l" + "c" * len(ms) + "}", r"\toprule", "Input & " + " & ".join(ms) + r" \\", r"\midrule"]
    for rep, h in order:
        cells = []
        for m in ms:
            if (rep, h, m) in a.index:
                r = a.loc[(rep, h, m)]; cells.append(f"{r[('auc','mean')]:.3f} / {r[('ece','mean')]:.3f}")
                V[f"E2_{rep}_{h}_{m.replace('-', '')}_auc"] = f"{r[('auc','mean')]:.3f}"; V[f"E2_{rep}_{h}_{m.replace('-', '')}_ece"] = f"{r[('ece','mean')]:.3f}"
            else:
                cells.append("--")
        lab = REPN.get(rep, rep) + (" + ComBat" if h == "combat" else "")
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write("tab_ablation", L)

# ---------------------------------------------------------------- E3 few-shot
R3 = load("fewshot"); R3u = load("fewshot_u")
if R3 is not None:
    ms = [m for m in MORDER if m in R3.model.unique()]
    a = R3.groupby(["n_context", "model"]).auc.agg(["mean", "std"])
    au = R3u.groupby(["n_context", "model"]).auc.agg(["mean", "std"]) if R3u is not None else None
    ns = sorted(R3.n_context.unique())
    L = [r"\begin{table}[t]", r"\centering", r"\caption{E3 -- Few-shot label efficiency: ROC-AUC (mean over 5 outer folds $\times$ context draws) as a function of the number $n$ of labelled context participants (class-balanced). Top: PLS fitted on the $n$ labelled participants only. Bottom: PCA-100 fitted on all training-fold participants \emph{without} labels (semi-supervised).}", r"\label{tab:fewshot}", r"\scriptsize",
         r"\begin{tabular}{l" + "c" * len(ns) + "}", r"\toprule", "Model & " + " & ".join(f"$n={n}$" for n in ns) + r" \\", r"\midrule",
         r"\multicolumn{" + str(len(ns) + 1) + r"}{l}{\emph{Supervised compression on the labelled context}} \\"]
    for m in ms:
        L.append(m + " & " + " & ".join(f"{a.loc[(n, m), 'mean']:.3f}" if (n, m) in a.index else "--" for n in ns) + r" \\")
        for n in ns:
            if (n, m) in a.index: V[f"E3_{m.replace('-', '')}_n{n}"] = f"{a.loc[(n, m), 'mean']:.3f}"
    if au is not None:
        L.append(r"\midrule"); L.append(r"\multicolumn{" + str(len(ns) + 1) + r"}{l}{\emph{Unsupervised PCA-100 on unlabelled training data}} \\")
        for m in [m for m in MORDER if m in R3u.model.unique()]:
            L.append(m + " & " + " & ".join(f"{au.loc[(n, m), 'mean']:.3f}" if (n, m) in au.index else "--" for n in ns) + r" \\")
            for n in ns:
                if (n, m) in au.index: V[f"E3u_{m.replace('-', '')}_n{n}"] = f"{au.loc[(n, m), 'mean']:.3f}"
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_fewshot", L)

# ---------------------------------------------------------------- E4 LOSO + shortcut
R4 = load("loso")
if R4 is not None:
    R4["harm"] = R4.harm.fillna("none")
    a = R4.groupby(["rep", "harm", "model"])[["acc", "bacc", "auc"]].agg(["mean", "std"])
    L = [r"\begin{table}[t]", r"\centering", r"\caption{E4 -- Leave-one-site-out (LOSO) generalisation over the 17 ABIDE-I sites: mean $\pm$ SD across held-out sites. Site-$z$ = label-free per-site standardisation.}", r"\label{tab:loso}", r"\scriptsize",
         r"\begin{tabular}{lllccc}", r"\toprule", r"Input & Harm. & Model & Accuracy (\%) & Bal.\ acc.\ (\%) & AUC \\", r"\midrule"]
    for rep in ["full", "pca100", "plsauto"]:
        for h in ["none", "sitez"]:
            for m in MORDER:
                if (rep, h, m) not in a.index:
                    continue
                r = a.loc[(rep, h, m)]
                L.append(f"{REPN[rep]} & {'site-$z$' if h == 'sitez' else 'none'} & {m} & {pm(r[('acc','mean')], r[('acc','std')])} & {pm(r[('bacc','mean')], r[('bacc','std')])} & {r[('auc','mean')]:.3f} $\\pm$ {r[('auc','std')]:.3f} \\\\")
                V[f"E4_{rep}_{h}_{m.replace('-', '')}_auc"] = f"{r[('auc','mean')]:.3f}"; V[f"E4_{rep}_{h}_{m.replace('-', '')}_acc"] = f"{100*r[('acc','mean')]:.1f}"
                V[f"E4_{rep}_{h}_{m.replace('-', '')}_aucsd"] = f"{r[('auc','std')]:.3f}"
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table}"]
    write("tab_loso", L)
    # per-site extremes for the best LOSO config
    for (rep, m) in [("plsauto", "TabICL-v2"), ("full", "LR-full"), ("pca100", "TabICL-v2")]:
        d = R4[(R4.rep == rep) & (R4.model == m) & (R4.harm == "none")]
        if len(d):
            V[f"E4_{rep}_{m.replace('-', '')}_minsite"] = f"{d.loc[d.auc.idxmin(), 'site'].title()} ({d.auc.min():.2f})"
            V[f"E4_{rep}_{m.replace('-', '')}_maxsite"] = f"{d.loc[d.auc.idxmax(), 'site'].title()} ({d.auc.max():.2f})"
SC = load("shortcut")
sj = json.load(open([os.path.join(d, "site_shortcut.json") for d in DIRS if os.path.exists(os.path.join(d, "site_shortcut.json"))][0]))
V["SC_prior_auc"] = f"{sj['site_prior_auc_mean']:.3f}"; V["SC_prior_acc"] = f"{100*sj['site_prior_acc_mean']:.1f}"
for rep, (mu, sd) in sj["site_decoding_bacc"].items():
    V[f"SC_dec_{rep}"] = f"{100*mu:.1f}"
if SC is not None:
    R1 = load("main"); R1 = R1[R1["repeat"] == 0]
    L = [r"\begin{table}[t]", r"\centering", r"\caption{Site-shortcut analysis (same ten folds, repetition 1 of E1). Site decoding: balanced accuracy of a 17-class logistic regression predicting acquisition site from each representation (chance 5.9\%). AUC before and after label-free per-site standardisation of every edge (site-$z$), which removes site-specific means and scales while keeping all labels unused.}", r"\label{tab:shortcut}", r"\scriptsize",
         r"\begin{tabular}{llccc}", r"\toprule", r"Input & Model & AUC (raw) & AUC (site-$z$) & $\Delta$ \\", r"\midrule"]
    for rep in ["pca100", "plsauto"]:
        for m in MORDER:
            raw = R1[(R1.rep == rep) & (R1.model == m)].set_index("tag").auc
            sz = SC[(SC.rep == rep) & (SC.model == m)].set_index("tag").auc
            idx = raw.index.intersection(sz.index)
            if len(idx) < 5:
                continue
            L.append(f"{REPN[rep]} & {m} & {raw[idx].mean():.3f} & {sz[idx].mean():.3f} & {sz[idx].mean() - raw[idx].mean():+.3f} \\\\")
            V[f"SC_{rep}_{m.replace('-', '')}_raw"] = f"{raw[idx].mean():.3f}"; V[f"SC_{rep}_{m.replace('-', '')}_sitez"] = f"{sz[idx].mean():.3f}"
            V[f"SC_{rep}_{m.replace('-', '')}_delta"] = f"{sz[idx].mean() - raw[idx].mean():+.3f}"
        L.append(r"\midrule")
    L.append(r"\multicolumn{5}{l}{Site decoding (balanced acc.): PCA-20 " + V['SC_dec_pca20'] + r"\%, PCA-100 " + V['SC_dec_pca100'] + r"\%, PLS-auto " + V['SC_dec_plsauto'] + r"\%; site-prior baseline AUC " + V['SC_prior_auc'] + r".} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_shortcut", L)

# ---------------------------------------------------------------- E5 groups
R5 = load("groups")
if R5 is not None:
    groups = ["Frontal", "Temporal", "Parietal", "Occipital", "Limbic", "Subcortical", "Cerebellum", "Whole-brain"]
    ms = [m for m in ["LR", "XGBoost", "TabICL-v1", "TabICL-v2"] if m in R5.model.unique()]
    a = R5.groupby(["group", "control", "model"]).auc.mean()
    ne = R5.groupby("group").n_edges.first()
    L = [r"\begin{table}[t]", r"\centering", r"\caption{E5 -- Anatomical feature groups (edges with at least one endpoint in the lobe; PLS-auto; 5-fold CV). Cells: AUC for the anatomical set / size-matched random edge set.}", r"\label{tab:groups}", r"\scriptsize",
         r"\begin{tabular}{lr" + "c" * len(ms) + "}", r"\toprule", "Group & Edges & " + " & ".join(ms) + r" \\", r"\midrule"]
    for gname in groups:
        cells = []
        for m in ms:
            real = a.get((gname, 0, m), np.nan); rnd = a.get((gname, 1, m), np.nan)
            cells.append(f"{real:.3f}" + ("" if np.isnan(rnd) else f" / {rnd:.3f}"))
            V[f"E5_{gname}_{m.replace('-', '')}"] = f"{real:.3f}"
            if not np.isnan(rnd): V[f"E5_{gname}_{m.replace('-', '')}_rnd"] = f"{rnd:.3f}"
        L.append(f"{gname} & {int(ne[gname]):,} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_groups", L)

# ---------------------------------------------------------------- E6 / E7 / E8
P6 = pd.read_csv(os.path.join(ANA, "E6_permutation.csv"))
for _, r in P6.iterrows():
    V[f"E6_{r.model.replace('-', '')}_{r.metric}_obs"] = f"{r.observed:.3f}"; V[f"E6_{r.model.replace('-', '')}_{r.metric}_null"] = f"{r.null_mean:.3f} $\\pm$ {r.null_sd:.3f}"
    V[f"E6_{r.model.replace('-', '')}_{r.metric}_p"] = f"{r.p_perm:.4f}".rstrip("0"); V[f"E6_{r.model.replace('-', '')}_nperm"] = str(int(r.n_perm))
R7 = load("samplesize")
if R7 is not None:
    a = R7.groupby(["N", "model"])[["acc", "auc"]].agg(["mean", "std"])
    ms = [m for m in MORDER if m in R7.model.unique()]
    L = [r"\begin{table}[t]", r"\centering", r"\caption{E7 -- Effect of cohort size (random subsamples, 5-fold CV, PLS-auto; 3 draws per size). Cells: AUC mean $\pm$ SD over folds and draws.}", r"\label{tab:samplesize}", r"\scriptsize",
         r"\begin{tabular}{r" + "c" * len(ms) + "}", r"\toprule", "$N$ & " + " & ".join(ms) + r" \\", r"\midrule"]
    for N in sorted(R7.N.unique()):
        L.append(f"{N} & " + " & ".join(f"{a.loc[(N, m), ('auc','mean')]:.3f} $\\pm$ {a.loc[(N, m), ('auc','std')]:.3f}" if (N, m) in a.index else "--" for m in ms) + r" \\")
        for m in ms:
            if (N, m) in a.index:
                V[f"E7_N{N}_{m.replace('-', '')}_auc"] = f"{a.loc[(N, m), ('auc','mean')]:.3f}"; V[f"E7_N{N}_{m.replace('-', '')}_aucsd"] = f"{a.loc[(N, m), ('auc','std')]:.3f}"
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_samplesize", L)
R8 = load("adhd")
if R8 is not None:
    R8 = R8[~R8.tag.str.startswith("loso")]
    a = R8.groupby(["rep", "model"])[["acc", "auc"]].agg(["mean", "std"])
    P8 = pd.read_csv(os.path.join(ANA, "E8_adhd_permutation.csv")).set_index(["model", "metric"])
    L = [r"\begin{table}[t]", r"\centering", r"\caption{E8 -- ADHD-200 stress test ($N=40$; 5-fold CV repeated 10 times). Permutation $p$: PLS-2 pipeline, 5-fold CV, accuracy statistic.}", r"\label{tab:adhd}", r"\scriptsize",
         r"\begin{tabular}{llccc}", r"\toprule", r"Input & Model & Accuracy (\%) & AUC & Perm.\ $p$ \\", r"\midrule"]
    for rep in ["full", "pca10", "pls2"]:
        for m in MORDER:
            if (rep, m) not in a.index:
                continue
            r = a.loc[(rep, m)]
            pp = f"{P8.loc[(m, 'acc'), 'p_perm']:.2f}" if rep == "pls2" and (m, "acc") in P8.index else "--"
            L.append(f"{REPN[rep]} & {m} & {pm(r[('acc','mean')], r[('acc','std')])} & {r[('auc','mean')]:.3f} $\\pm$ {r[('auc','std')]:.3f} & {pp} \\\\")
            V[f"E8_{rep}_{m.replace('-', '')}_acc"] = f"{100*r[('acc','mean')]:.1f}"; V[f"E8_{rep}_{m.replace('-', '')}_auc"] = f"{r[('auc','mean')]:.3f}"
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table}"]
    write("tab_adhd", L)
    V["E8_pmin"] = f"{P8.p_perm.min():.2f}"; V["E8_pmax"] = f"{P8.p_perm.max():.2f}"
json.dump(V, open(os.path.join(PAPER, "values.json"), "w"), indent=1)
print(len(V), "values total")
