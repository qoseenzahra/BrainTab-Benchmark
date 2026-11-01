"""Figures with real brain images from the two datasets.

Fig. 2  (fig_datasets):   raw and preprocessed MRI of one ABIDE-I and two ADHD-200 participants,
                          the CC200 parcellation, ROI time series and connectivity matrices.
Fig. 3  (fig_brain_maps): CC200 ROI centroids coloured by lobe (glass brain), ASD-TD group
                          difference in connectivity, and the 40 most group-different edges.
Usage: python scripts/07_fig_datasets.py <image_dir> <roi_dir> <out_dir>
  image_dir: brain_images/ (cc200_roi_atlas.nii.gz, ABIDE_*_func_mean.nii.gz, *_T1w_raw.nii.gz, small/*)
  roi_dir:   folder with ABIDE PCP *_rois_cc200.1D files
"""
import os, sys, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
import nibabel as nib
from nilearn import plotting, datasets, image
from braintab.groups import roi_lobes, LOBES

IMG, ROI, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})
INK, MUTED = "#0b0b0b", "#52514e"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a3201f"])
LOBE_COL = {"Frontal": "#2a78d6", "Temporal": "#eb6834", "Parietal": "#1baf7a", "Occipital": "#eda100",
            "Limbic": "#e87ba4", "Subcortical": "#008300", "Cerebellum": "#6250d6", "Unassigned": "#b5b4ae"}
SUBJ = "NYU_0051016"           # ABIDE-I example participant (ASD, 23 y, low head motion)


def panel_label(ax, s, x=None, y=None):
    t = ax.get_title()
    ax.set_title(f"({s}) {t}", pad=3, loc="center")


def load_labels():
    df = pd.read_csv(os.path.join(ROOT, "data", "CC200_ROI_labels.csv"), skipinitialspace=True)
    com = np.array([[float(v) for v in re.findall(r"-?[0-9.]+", s)] for s in df.iloc[:, 2].astype(str)])
    return com, roi_lobes(os.path.join(ROOT, "data", "CC200_ROI_labels.csv"))


def fc(ts):
    ts = ts[:, ts.std(0) > 0] if False else ts
    r = np.corrcoef(ts.T); np.fill_diagonal(r, 0)
    return np.arctanh(np.clip(np.nan_to_num(r), -0.999999, 0.999999))


com, lobes = load_labels()
order = np.argsort([list(LOBE_COL).index(l) for l in lobes], kind="stable")
tmpl = datasets.load_mni152_template(resolution=1)
atlas = nib.load(os.path.join(IMG, "cc200_roi_atlas.nii.gz"))
ts = np.loadtxt(os.path.join(ROI, f"{SUBJ}_rois_cc200.1D"))
d = np.load(os.path.join(ROOT, "data", "abide_features.npz"))
meta = pd.read_csv(os.path.join(ROOT, "data", "abide_meta.csv"))
m = meta.qc_pass.values; Z = d["fisher_z"][m].astype(np.float64); y = meta.y.values[m]
iu = np.triu_indices(200, 1)
def to_mat(v):
    M = np.zeros((200, 200)); M[iu] = v; return M + M.T

# ===================================================================== Figure 2: datasets
fig = plt.figure(figsize=(7.4, 8.9))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.12], hspace=0.45, wspace=0.42,
                      left=0.03, right=0.94, top=0.965, bottom=0.04)
fig.text(0.01, 0.985, "ABIDE-I  (autism; PCP C-PAC; N = 873 after QC, 17 sites)", fontsize=8, fontweight="bold", color=INK)
# (a) raw T1w
ax = fig.add_subplot(gs[0, 0])
T1 = nib.as_closest_canonical(nib.load(os.path.join(IMG, f"ABIDE_{SUBJ}_T1w_raw.nii.gz")))
T = np.asarray(T1.dataobj, dtype=np.float32); sl = T[T.shape[0] // 2 + 4].T
ax.imshow(sl, cmap="gray", origin="lower", vmin=np.percentile(sl, 1), vmax=np.percentile(sl, 99.5),
          aspect=T1.header.get_zooms()[2] / T1.header.get_zooms()[1]); ax.axis("off")
ax.set_title("raw T1-weighted MRI (native space)", pad=2); panel_label(ax, "a", 0.05, 1.0)
# (b) preprocessed fMRI mean
ax = fig.add_subplot(gs[0, 1])
plotting.plot_epi(os.path.join(IMG, f"ABIDE_{SUBJ}_func_mean.nii.gz"), display_mode="z", cut_coords=[8], axes=ax,
                  annotate=False, draw_cross=False, black_bg=False, cmap="gray", colorbar=False)
ax.set_title("preprocessed rs-fMRI mean (MNI)", pad=2); panel_label(ax, "b", 0.05, 1.0)
# (c) CC200 atlas
ax = fig.add_subplot(gs[0, 2])
rng = np.random.default_rng(3); cm = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20))[rng.permutation(20)])
plotting.plot_roi(atlas, bg_img=tmpl, display_mode="z", cut_coords=[8], axes=ax, cmap=cm, alpha=0.85,
                  annotate=False, draw_cross=False, black_bg=False, colorbar=False)
ax.set_title("CC200 parcellation: 200 ROIs", pad=2); panel_label(ax, "c", 0.05, 1.0)
# (d) ROI time series
ax = fig.add_subplot(gs[1, 0])
pick = [int(np.where(lobes == L)[0][0]) for L in ["Frontal", "Temporal", "Parietal", "Occipital", "Limbic", "Cerebellum"]]
tsz = (ts - ts.mean(0)) / (ts.std(0) + 1e-9); t = np.arange(len(ts)) * 2.0
for i, r in enumerate(pick):
    ax.plot(t, tsz[:, r] / 3 + i, lw=0.6, color=LOBE_COL[lobes[r]])
    ax.text(t[-1] + 6, i, f"ROI {r + 1}", fontsize=5.5, va="center", color=MUTED)
ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.set_xlim(0, t[-1] + 45)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.set_title(f"ROI BOLD time series ({len(ts)} vols)", pad=2); panel_label(ax, "d", -0.02, 1.0)
# (e) subject FC
ax = fig.add_subplot(gs[1, 1])
F = fc(ts)[np.ix_(order, order)]
im = ax.imshow(F, cmap=DIV, vmin=-1.2, vmax=1.2, interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
b = np.cumsum(np.unique(lobes[order], return_counts=True)[1][np.argsort(np.unique(lobes[order], return_index=True)[1])])
for v in b[:-1]: ax.axhline(v - 0.5, color="white", lw=0.4); ax.axvline(v - 0.5, color="white", lw=0.4)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=5.5)
ax.set_title("connectivity matrix (Fisher z)", pad=2); panel_label(ax, "e", -0.02, 1.0)
# (f) feature vector
ax = fig.add_subplot(gs[1, 2])
v = fc(ts)[iu]
ax.hist(v, bins=60, color="#2a78d6", edgecolor="white", linewidth=0.2)
ax.set_xlabel("edge value (Fisher z)"); ax.set_ylabel("number of edges")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.set_title("19,900 edges = one table row", pad=2); panel_label(ax, "f", -0.02, 1.0)

fig.text(0.01, 0.497, "ADHD-200  (ADHD; Athena pipeline; 8 sites)", fontsize=8, fontweight="bold", color=INK)
# (g) raw BOLD native KKI
ax = fig.add_subplot(gs[2, 0])
raw = image.index_img(os.path.join(IMG, "small", "ADHD_KKI_1018959_raw_3vols.nii.gz"), 1)
plotting.plot_epi(raw, display_mode="z", cut_coords=1, axes=ax, annotate=False, draw_cross=False,
                  black_bg=False, cmap="gray", colorbar=False)
ax.set_title("raw rs-fMRI, KKI (native space)", pad=2); panel_label(ax, "g", 0.05, 1.0)
# (h) preprocessed MNI NYU
ax = fig.add_subplot(gs[2, 1])
plotting.plot_epi(os.path.join(IMG, "small", "ADHD_NYU_0010042_mni_mean.nii.gz"), display_mode="z", cut_coords=[8], axes=ax,
                  annotate=False, draw_cross=False, black_bg=False, cmap="gray", colorbar=False)
ax.set_title("preprocessed rs-fMRI mean, NYU (MNI)", pad=2); panel_label(ax, "h", 0.05, 1.0)
# (i) voxel signals
ax = fig.add_subplot(gs[2, 2])
for k, (f, c, lab, tr) in enumerate([("ADHD_KKI_1018959_raw_voxts.npy", "#eb6834", "KKI raw voxel", 2.5),
                                     ("ADHD_NYU_0010042_mni_voxts.npy", "#6250d6", "NYU preprocessed voxel", 2.0)]):
    V = np.load(os.path.join(IMG, "small", f)); s = V[np.argmax(V.mean(1))]
    s = (s - s.mean()) / (s.std() + 1e-9)
    ax.plot(np.arange(len(s)) * tr, s / 3 + k * 1.6, lw=0.6, color=c)
    ax.text(0, k * 1.6 + 0.62, lab, fontsize=5.5, color=MUTED)
ax.set_yticks([]); ax.set_xlabel("time (s)")
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.set_title("single-voxel BOLD signals", pad=2); panel_label(ax, "i", -0.02, 1.0)

# (j) group mean FC ASD / TD / diff (ABIDE)
MA, MT = to_mat(Z[y == 1].mean(0)), to_mat(Z[y == 0].mean(0))
for j, (M, ttl, vmax, lab) in enumerate([(MA, "ABIDE-I mean FC, ASD (n = 405)", 0.9, "j"),
                                         (MT, "ABIDE-I mean FC, TD (n = 468)", 0.9, "k"),
                                         (MA - MT, "difference ASD − TD", 0.06, "l")]):
    ax = fig.add_subplot(gs[3, j])
    im = ax.imshow(M[np.ix_(order, order)], cmap=DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for v_ in b[:-1]: ax.axhline(v_ - 0.5, color="white", lw=0.4); ax.axvline(v_ - 0.5, color="white", lw=0.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=5.5)
    ax.set_title(ttl, pad=2); panel_label(ax, lab, -0.02, 1.0)
# lobe key for matrices
names = [L for L in LOBE_COL if (lobes == L).any()]
fig.legend([Line2D([0], [0], color=LOBE_COL[L], lw=3) for L in names], names, loc="lower center", ncol=len(names),
           frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0), title="ROI order in matrices / colours in (d): lobe of max-overlap AAL label",
           title_fontsize=6)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_datasets.{ext}"), dpi=300)
plt.close(fig)

# ===================================================================== Figure 3: brain maps
from scipy import stats
from braintab.harmonize import site_standardize
Zs = site_standardize(Z, meta.site.values[m].astype(str))       # remove site means/scales (label-free)
tval, pval = stats.ttest_ind(Zs[y == 1], Zs[y == 0], equal_var=False)
q = np.sort(pval)
fdr_n = int(np.sum(q <= 0.05 * np.arange(1, len(q) + 1) / len(q)))
top = np.argsort(-np.abs(tval))[:40]
A = np.zeros((200, 200)); A[iu[0][top], iu[1][top]] = tval[top]; A = A + A.T
fig = plt.figure(figsize=(7.4, 5.0))
ax = fig.add_axes([0.0, 0.52, 1.0, 0.44])
node_col = [LOBE_COL[l] for l in lobes]
plotting.plot_connectome(np.zeros((200, 200)), com, node_color=node_col, node_size=10, display_mode="lyrz",
                         axes=ax, annotate=False, black_bg=False, colorbar=False, node_kwargs={"edgecolors": "white", "linewidths": 0.3})
ax.set_title("CC200 ROI centroids (MNI) coloured by lobe — the anatomical feature groups of experiment E5", pad=1)
panel_label(ax, "a", 0.02, 0.98)
fig.legend([Line2D([0], [0], marker="o", color="w", markerfacecolor=LOBE_COL[L], markersize=5) for L in names], names,
           loc="center", ncol=len(names), frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.505))
ax = fig.add_axes([0.0, 0.02, 1.0, 0.44])
nodes = np.unique(np.concatenate([iu[0][top], iu[1][top]]))
ns = np.zeros(200); ns[nodes] = 14
plotting.plot_connectome(A, com, node_color=node_col, node_size=ns, edge_cmap=DIV, edge_vmin=-7, edge_vmax=7,
                         display_mode="lyrz", axes=ax, annotate=False, black_bg=False, colorbar=True,
                         edge_kwargs={"linewidth": 1.1})
ax.set_title(f"(b) 40 edges with the largest ASD–TD difference after per-site z-scoring (Welch t; blue = ASD < TD); "
             f"{fdr_n:,} edges survive FDR q < 0.05", pad=1)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_brain_maps.{ext}"), dpi=300)
summ = pd.DataFrame(dict(edge_i=iu[0][top] + 1, edge_j=iu[1][top] + 1, t=tval[top], p=pval[top],
                         lobe_i=lobes[iu[0][top]], lobe_j=lobes[iu[1][top]]))
summ.to_csv(os.path.join(OUT, "top40_group_edges.csv"), index=False)
lob_count = pd.Series(np.concatenate([summ.lobe_i, summ.lobe_j])).value_counts()
print("FDR edges:", fdr_n, "max |t|:", np.abs(tval).max().round(2)); print(lob_count.to_dict())
import json; json.dump(dict(fdr_edges=fdr_n, max_t=float(np.abs(tval).max()), top40_lobes=lob_count.to_dict(),
                            n_pos=int((tval[top] > 0).sum())), open(os.path.join(OUT, "brain_maps_summary.json"), "w"))
