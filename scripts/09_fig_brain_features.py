"""Brain-feature figures for both cohorts (ABIDE-I CC200 and ADHD-200 Athena CC200).

fig_adhd_brain      ADHD-200 parcellation, group-mean connectomes, ADHD-TDC effect sizes, top edges, node map
fig_network_effects lobe-by-lobe effect sizes, share of nominally significant edges, node-level disruption maps
fig_model_maps      what the PLS->LR model uses: Haufe activation patterns (edges, nodes) and split-half stability
fig_site_embedding  t-SNE of the connectomes coloured by site and by diagnosis, before and after per-site z-scoring

All group statistics are descriptive and computed after label-free per-site z-scoring. For ADHD-200 only the five
sites that contributed both ADHD and control participants enter group comparisons (Pittsburgh and WashU contributed
controls only, so site and diagnosis are confounded there).
Usage: python scripts/09_fig_brain_features.py <brain_image_dir> <out_dir>
"""
import os, sys, re, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import nibabel as nib
from scipy import stats
from nilearn import plotting, datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from braintab.groups import roi_lobes
from braintab.harmonize import site_standardize

IMG, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a3201f"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f6f5f1", "#f3b08f", "#e66767", "#a3201f", "#5c0f10"])
FCMAP = LinearSegmentedColormap.from_list("fc", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a3201f"])
LOBE_ORDER = ["Frontal", "Temporal", "Parietal", "Occipital", "Limbic", "Subcortical", "Cerebellum"]
LOBE_COL = {"Frontal": "#2a78d6", "Temporal": "#eb6834", "Parietal": "#1baf7a", "Occipital": "#eda100",
            "Limbic": "#e87ba4", "Subcortical": "#008300", "Cerebellum": "#6250d6", "Unassigned": "#b5b4ae"}
SUMMARY = {}


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p, dtype=float)
    r = p[o] * len(p) / np.arange(1, len(p) + 1); q[o] = np.minimum.accumulate(r[::-1])[::-1]
    return np.minimum(q, 1)


def cohend(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return (a.mean(0) - b.mean(0)) / np.where(sp > 0, sp, 1)


def to_mat(v, R):
    M = np.zeros((R, R)); i, j = np.triu_indices(R, 1); M[i, j] = v; M[j, i] = v
    return M


# ------------------------------------------------------------------ cohorts
lab_csv = os.path.join(ROOT, "data", "CC200_ROI_labels.csv")
df = pd.read_csv(lab_csv, skipinitialspace=True)
com200 = np.array([[float(v) for v in re.findall(r"-?[0-9.]+", s)] for s in df.iloc[:, 2].astype(str)])
lob200 = roi_lobes(lab_csv)

d = np.load(os.path.join(ROOT, "data", "abide_features.npz")); meta = pd.read_csv(os.path.join(ROOT, "data", "abide_meta.csv"))
qm = meta.qc_pass.values.astype(bool)
ABIDE = dict(name="ABIDE-I", dis="ASD", ctl="TD", X=d["fisher_z"][qm].astype(np.float32), y=meta.y.values[qm].astype(int),
             site=meta.site.values[qm].astype(str), com=com200, lobes=lob200, R=200)
del d

# ADHD-200: Athena CC200 (4 mm). 3dROIstats columns follow ascending parcel label (190 non-empty parcels).
atl = nib.load(os.path.join(IMG, "ADHD200_parcellate_200.nii.gz")); A = np.asarray(atl.dataobj).astype(int)
labs = np.unique(A[A > 0])
comA = np.array([nib.affines.apply_affine(atl.affine, np.argwhere(A == L).mean(0)) for L in labs])
nn = np.argmin(((comA[:, None, :] - com200[None]) ** 2).sum(-1), 1)          # nearest ABIDE CC200 parcel
lobA = lob200[nn]
SUMMARY["adhd_atlas"] = dict(n_parcels=int(len(labs)), median_nn_dist_mm=float(np.median(np.sqrt(((comA - com200[nn]) ** 2).sum(1)))))
d = np.load(os.path.join(ROOT, "data", "adhd_full_features.npz")); ma = pd.read_csv(os.path.join(ROOT, "data", "adhd_full_meta.csv"))
assert d["fisher_z"].shape[1] == len(labs) * (len(labs) - 1) // 2
ADHD = dict(name="ADHD-200", dis="ADHD", ctl="TDC", X=d["fisher_z"].astype(np.float32), y=ma.y.values.astype(int),
            site=ma.site.values.astype(str), com=comA, lobes=lobA, R=len(labs))
del d

for C in (ABIDE, ADHD):
    s, y = C["site"], C["y"]
    C["mixed"] = np.isin(s, [k for k in np.unique(s) if len(np.unique(y[s == k])) == 2])
    m = C["mixed"]; C["Xs"] = site_standardize(C["X"][m], s[m]).astype(np.float64); C["ym"] = y[m]
    a, b = C["Xs"][C["ym"] == 1], C["Xs"][C["ym"] == 0]
    C["t"], C["p"] = stats.ttest_ind(a, b, equal_var=False); C["d"] = cohend(a, b); C["q"] = bh(C["p"])
    C["order"] = np.argsort([LOBE_ORDER.index(l) if l in LOBE_ORDER else 99 for l in C["lobes"]], kind="stable")
    SUMMARY[C["name"]] = dict(n_group=int(m.sum()), n_dis=int((C["ym"] == 1).sum()), n_ctl=int((C["ym"] == 0).sum()),
                              n_fdr=int((C["q"] < 0.05).sum()), n_p001=int((C["p"] < 0.001).sum()),
                              n_edges=int(len(C["p"])), max_abs_d=float(np.abs(C["d"]).max()),
                              frac_neg_top200=float((C["d"][np.argsort(-np.abs(C["d"]))[:200]] < 0).mean()))
    print(C["name"], SUMMARY[C["name"]])


def lobe_ticks(ax, lobes, order):
    lo = lobes[order]; R = len(lo); pos = 0
    for L in LOBE_ORDER + ["Unassigned"]:
        n = int((lo == L).sum())
        if n == 0: continue
        ax.add_patch(plt.Rectangle((-R * 0.045, pos - 0.5), R * 0.035, n, color=LOBE_COL[L], clip_on=False, lw=0))
        ax.add_patch(plt.Rectangle((pos - 0.5, R - 0.5 + R * 0.01), n, R * 0.035, color=LOBE_COL[L], clip_on=False, lw=0))
        pos += n
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def matshow(ax, M, C, vmax, cmap, title):
    o = C["order"]; im = ax.imshow(M[np.ix_(o, o)], cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    lobe_ticks(ax, C["lobes"], o); ax.set_title(title, pad=8)
    return im


def node_colors(C):
    return [LOBE_COL.get(l, "#b5b4ae") for l in C["lobes"]]


lobe_legend = [Line2D([], [], marker="o", ls="", color=LOBE_COL[L], markersize=4, label=L) for L in LOBE_ORDER]

# ------------------------------------------------------------------ Fig: ADHD brain features
C = ADHD; R = C["R"]; tmpl = datasets.load_mni152_template(resolution=2)
fig = plt.figure(figsize=(7.2, 7.0))
gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.05, 0.95], hspace=0.30, wspace=0.28)
ax = fig.add_subplot(gs[0, 0:2])
plotting.plot_roi(atl, bg_img=tmpl, axes=ax, display_mode="ortho", cut_coords=(0, -18, 12), cmap="tab20",
                  draw_cross=False, annotate=False, black_bg=False, alpha=0.9, colorbar=False)
ax.set_title(f"(a) ADHD-200 Athena CC200 parcellation ({R} parcels, 4 mm MNI)", pad=2)
mt = C["X"][C["y"] == 0].mean(0); ma_ = C["X"][C["y"] == 1].mean(0)
ax = fig.add_subplot(gs[0, 2]); im = matshow(ax, to_mat(mt, R), C, 1.0, FCMAP, f"(b) mean FC, TDC (n={int((C['y']==0).sum())})")
ax = fig.add_subplot(gs[0, 3]); matshow(ax, to_mat(ma_, R), C, 1.0, FCMAP, f"(c) mean FC, ADHD (n={int(C['y'].sum())})")
cax = fig.add_axes([0.915, 0.70, 0.008, 0.16]); cb = fig.colorbar(im, cax=cax); cb.set_label("Fisher z", fontsize=6); cb.ax.tick_params(labelsize=5)
# row 2: effect-size matrix + top-40 edges
ax = fig.add_subplot(gs[1, 0]); vm = np.percentile(np.abs(C["d"]), 99.9)
im2 = matshow(ax, to_mat(C["d"], R), C, vm, DIV, "(d) Cohen's d, ADHD − TDC")
cax = fig.add_axes([0.13, 0.375, 0.13, 0.007]); cb = fig.colorbar(im2, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=5)
ax = fig.add_subplot(gs[1, 1:])
top = np.argsort(-np.abs(C["t"]))[:40]; M = np.zeros(len(C["t"])); M[top] = C["t"][top]; Mm = to_mat(M, R)
used = np.where(np.abs(Mm).sum(0) > 0)[0]
plotting.plot_connectome(Mm[np.ix_(used, used)], C["com"][used], node_color=[node_colors(C)[k] for k in used], node_size=9,
                         edge_cmap=DIV, edge_vmin=-6, edge_vmax=6, display_mode="lzr", axes=ax, colorbar=False,
                         edge_kwargs=dict(linewidth=0.9), annotate=False)
SUMMARY["ADHD-200"]["top40_lobes"] = pd.Series(np.concatenate([C["lobes"][np.triu_indices(R, 1)[0][top]], C["lobes"][np.triu_indices(R, 1)[1][top]]])).value_counts().to_dict()
SUMMARY["ADHD-200"]["top40_frac_pos"] = float((C["t"][top] > 0).mean()); SUMMARY["ADHD-200"]["max_abs_t"] = float(np.abs(C["t"]).max())
ax.set_title(f"(e) 40 edges with the largest ADHD–TDC difference (Welch t; red = ADHD > TDC)", pad=2)
# row 3: node maps
deg = (to_mat((C["p"] < 0.01).astype(float), R)).sum(0)
ax = fig.add_subplot(gs[2, 0:2])
plotting.plot_markers(deg, C["com"], node_size=(deg / deg.max()) ** 1.5 * 22 + 1, node_cmap=SEQ, node_vmin=0, node_vmax=np.percentile(deg, 99),
                      display_mode="lzr", axes=ax, colorbar=False, annotate=False, alpha=0.9)
ax.set_title("(f) edges per parcel with p < 0.01 (node size and colour)", pad=2)
ax = fig.add_subplot(gs[2, 2:])
lob_share = pd.DataFrame({"lobe": C["lobes"], "deg": deg}).groupby("lobe").deg.mean().reindex(LOBE_ORDER).dropna()
ax.barh(range(len(lob_share)), lob_share.values, color=[LOBE_COL[l] for l in lob_share.index], height=0.65)
ax.set_yticks(range(len(lob_share))); ax.set_yticklabels(lob_share.index); ax.invert_yaxis()
ax.axvline(0.01 * (R - 1), color=MUTED, lw=0.8, ls="--"); ax.set_xlabel("mean no. of p < 0.01 edges per parcel (dashed: chance)")
ax.yaxis.tick_right(); ax.spines["left"].set_visible(False); ax.set_title("(g) where ADHD–TDC differences concentrate", pad=4)
ax.spines["top"].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"fig_adhd_brain.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig); print("fig_adhd_brain done")

# ------------------------------------------------------------------ Fig: network-level effects (both cohorts)
fig = plt.figure(figsize=(7.2, 5.9)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.8, 1.15], hspace=0.5, wspace=0.75)
for r, C in enumerate((ABIDE, ADHD)):
    R = C["R"]; i, j = np.triu_indices(R, 1); li, lj = C["lobes"][i], C["lobes"][j]
    L = [l for l in LOBE_ORDER if (C["lobes"] == l).any()]
    B = np.full((len(L), len(L)), np.nan); S = np.zeros_like(B)
    for a, la in enumerate(L):
        for b, lb in enumerate(L):
            msk = ((li == la) & (lj == lb)) | ((li == lb) & (lj == la))
            if msk.any(): B[a, b] = C["d"][msk].mean(); S[a, b] = (C["p"][msk] < 0.05).mean() * 100
    ax = fig.add_subplot(gs[r, 0]); vm = np.nanmax(np.abs(B))
    im = ax.imshow(B, cmap=DIV, vmin=-vm, vmax=vm)
    ax.set_xticks(range(len(L))); ax.set_xticklabels([l[:4] for l in L], rotation=90); ax.set_yticks(range(len(L))); ax.set_yticklabels(L)
    for a in range(len(L)):
        for b in range(len(L)):
            ax.text(b, a, f"{S[a, b]:.0f}", ha="center", va="center", fontsize=4.6, color=INK if abs(B[a, b]) < 0.6 * vm else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=5)
    ax.set_title(f"({'ad'[r]}) {C['name']}: lobe × lobe mean d\n(numbers = % edges p < 0.05)", pad=3)
    SUMMARY[C["name"]]["lobe_block_min"] = dict(zip(["d", "pair"], [float(np.nanmin(B)), "/".join(np.array(L)[list(np.unravel_index(np.nanargmin(B), B.shape))])]))
    SUMMARY[C["name"]]["lobe_block_max"] = dict(zip(["d", "pair"], [float(np.nanmax(B)), "/".join(np.array(L)[list(np.unravel_index(np.nanargmax(B), B.shape))])]))
    # incident share of significant edges per lobe vs 5 % expected
    ax = fig.add_subplot(gs[r, 1])
    share = [((C["p"] < 0.05) & ((li == l) | (lj == l))).sum() / ((li == l) | (lj == l)).sum() * 100 for l in L]
    neg = [((C["p"] < 0.05) & (C["d"] < 0) & ((li == l) | (lj == l))).sum() / ((li == l) | (lj == l)).sum() * 100 for l in L]
    ax.barh(range(len(L)), share, color=[LOBE_COL[l] for l in L], height=0.65, alpha=0.35, label="all")
    ax.barh(range(len(L)), neg, color=[LOBE_COL[l] for l in L], height=0.65, label=f"{C['dis']} < {C['ctl']}")
    ax.axvline(5, color=MUTED, ls="--", lw=0.8); ax.set_yticks(range(len(L))); ax.set_yticklabels(L); ax.invert_yaxis()
    ax.set_xlabel("% incident edges"); ax.set_title(f"({'be'[r]}) edges p < 0.05 per lobe\n(dark: {C['dis']} < {C['ctl']}; dashed: chance)", pad=3)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    SUMMARY[C["name"]]["lobe_share_p05"] = dict(zip(L, map(float, share)))
    ax = fig.add_subplot(gs[r, 2])
    node_d = np.abs(to_mat(C["d"], R)).mean(0)
    plotting.plot_markers(node_d, C["com"], node_size=(node_d / node_d.max()) ** 2 * 28, node_cmap=SEQ, node_vmin=node_d.min(), node_vmax=node_d.max(),
                          display_mode="z", axes=ax, colorbar=True, annotate=False, alpha=0.9)
    top5 = np.argsort(-node_d)[:5]
    SUMMARY[C["name"]]["top_nodes"] = [dict(idx=int(k), lobe=str(C["lobes"][k]), mni=[round(float(v), 1) for v in C["com"][k]], mean_abs_d=float(node_d[k])) for k in top5]
    ax.set_title(f"({'cf'[r]}) {C['name']}: node mean |d|", pad=2)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"fig_network_effects.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig); print("fig_network_effects done")


# ------------------------------------------------------------------ Fig: model activation patterns (Haufe et al., 2014)
def haufe(X, y, k=5):
    Xc = X - X.mean(0)
    pls = PLSRegression(k, scale=False).fit(Xc, y - y.mean())
    sc = Xc @ pls.x_rotations_
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, max_iter=5000).fit((sc - sc.mean(0)) / sc.std(0), y)
    f = ((sc - sc.mean(0)) / sc.std(0)) @ lr.coef_.ravel()
    return Xc.T @ (f - f.mean()) / (len(y) - 1) / f.std()      # covariance of each edge with the decision value


fig = plt.figure(figsize=(7.2, 4.4)); gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 0.95, 0.85], hspace=0.45, wspace=0.32)
for r, C in enumerate((ABIDE, ADHD)):
    R = C["R"]; X, y = C["Xs"], C["ym"]
    P = haufe(X, y); rng = np.random.default_rng(0); rs = []
    for b in range(20):
        idx = rng.permutation(len(y)); h1, h2 = idx[: len(y) // 2], idx[len(y) // 2:]
        rs.append(np.corrcoef(haufe(X[h1], y[h1]), haufe(X[h2], y[h2]))[0, 1])
    r_t = np.corrcoef(P, C["t"])[0, 1]
    SUMMARY[C["name"]]["haufe"] = dict(split_half_r_mean=float(np.mean(rs)), split_half_r_sd=float(np.std(rs)), r_with_t=float(r_t),
                                       frac_neg_top100=float((P[np.argsort(-np.abs(P))[:100]] < 0).mean()))
    top = np.argsort(-np.abs(P))[:30]; M = np.zeros(len(P)); M[top] = P[top] / np.abs(P[top]).max(); Mm = to_mat(M, R)
    used = np.where(np.abs(Mm).sum(0) > 0)[0]
    ax = fig.add_subplot(gs[r, 0])
    plotting.plot_connectome(Mm[np.ix_(used, used)], C["com"][used], node_color=[node_colors(C)[k] for k in used], node_size=8,
                             edge_cmap=DIV, edge_vmin=-1, edge_vmax=1, display_mode="lzr", axes=ax, colorbar=False,
                             edge_kwargs=dict(linewidth=0.9), annotate=False)
    ax.set_title(f"({'ad'[r]}) {C['name']}: 30 strongest pattern edges (red = higher in {C['dis']})", pad=2)
    ax = fig.add_subplot(gs[r, 1]); node = np.abs(to_mat(P, R)).sum(0)
    plotting.plot_markers(node, C["com"], node_size=(node / node.max()) ** 2 * 30, node_cmap=SEQ, node_vmin=node.min(), node_vmax=node.max(),
                          display_mode="z", axes=ax, colorbar=False, annotate=False, alpha=0.9)
    SUMMARY[C["name"]]["haufe_top_nodes"] = [dict(lobe=str(C["lobes"][k]), mni=[round(float(v), 1) for v in C["com"][k]]) for k in np.argsort(-node)[:5]]
    ax.set_title(f"({'be'[r]}) node importance", pad=2)
    ax = fig.add_subplot(gs[r, 2]); sub = rng.choice(len(P), 4000, replace=False)
    ax.scatter(C["t"][sub], P[sub] / np.abs(P).max(), s=1.2, color="#2a78d6", alpha=0.35, lw=0)
    ax.axhline(0, color=GRID, lw=0.6); ax.axvline(0, color=GRID, lw=0.6)
    ax.set_xlabel("Welch t", labelpad=1); ax.set_ylabel("pattern (scaled)", labelpad=1)
    ax.set_title(f"({'cf'[r]}) vs t: r = {r_t:.2f}\nsplit-half r = {np.mean(rs):.2f}", pad=3)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    print(C["name"], SUMMARY[C["name"]]["haufe"])
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"fig_model_maps.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig); print("fig_model_maps done")

# ------------------------------------------------------------------ Fig: site vs diagnosis embedding
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9))
DXC = {0: "#8a8984", 1: "#e66767"}
for r, C in enumerate((ABIDE, ADHD)):
    X = C["X"].astype(np.float64); s, y = C["site"], C["y"]; Xz = site_standardize(X, s).astype(np.float64)
    sites = list(pd.Series(s).value_counts().index); cmap = plt.get_cmap("tab20")
    SUMMARY[C["name"]]["silhouette"] = {}
    for c, (lab, XX) in enumerate([("raw", X), ("per-site z", Xz)]):
        Zp = PCA(50, random_state=0).fit_transform(XX - XX.mean(0))
        E = TSNE(2, perplexity=30, init="pca", random_state=0, learning_rate="auto").fit_transform(Zp)
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        cv = StratifiedKFold(5, shuffle=True, random_state=0); lr = LogisticRegression(C=1.0, max_iter=5000)
        sil_site = float(cross_val_score(lr, Zp, s, cv=cv, scoring="balanced_accuracy").mean())
        sil_dx = float(cross_val_score(lr, Zp, y, cv=cv, scoring="roc_auc").mean())
        SUMMARY[C["name"]]["silhouette"][lab] = dict(site_bacc=sil_site, site_chance=1 / len(sites), diagnosis_auc=sil_dx)
        ax = axs[r, 2 * c]
        for k, si in enumerate(sites):
            m = s == si; ax.scatter(E[m, 0], E[m, 1], s=1.6, color=cmap(k % 20), lw=0, label=si)
        ax.set_title(f"{C['name']}, {lab}: by site\nsite decoding bACC {sil_site:.2f} (chance {1/len(sites):.2f})", pad=2, fontsize=6.5)
        ax = axs[r, 2 * c + 1]
        for v in (0, 1):
            m = y == v; ax.scatter(E[m, 0], E[m, 1], s=1.6, color=DXC[v], lw=0, alpha=0.7, label=[C["ctl"], C["dis"]][v])
        ax.set_title(f"{C['name']}, {lab}: by diagnosis\ndiagnosis decoding AUC {sil_dx:.2f}", pad=2, fontsize=6.5)
        if c == 1: ax.legend(fontsize=5, markerscale=3, frameon=False, loc="lower right", handletextpad=0.1)
    for ax in axs[r]: ax.set_xticks([]); ax.set_yticks([]); [sp.set_color(GRID) for sp in ax.spines.values()]
    print(C["name"], SUMMARY[C["name"]]["silhouette"])
fig.tight_layout(h_pad=0.8, w_pad=0.4)
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"fig_site_embedding.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
json.dump(SUMMARY, open(os.path.join(OUT, "brain_features_summary.json"), "w"), indent=1)
print("ALL DONE")
