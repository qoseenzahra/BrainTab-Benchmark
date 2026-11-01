"""Figure 1: BrainTab-Benchmark methodology, with real brain images from the datasets.
Usage: python scripts/04_fig1_pipeline.py <out_dir> <image_dir> <roi_dir> [cohorts.json]
cohorts.json (optional): {"adhd_n": .., "adhd_adhd": .., "adhd_tdc": .., "adhd_sites": ..}"""
import sys, os, json, io
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import nibabel as nib
from nilearn import plotting, datasets, image

out, IMG, ROI = sys.argv[1], sys.argv[2], sys.argv[3]
coh = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}
os.makedirs(out, exist_ok=True)
INK, MUTED = "#0b0b0b", "#52514e"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a3201f"])
SUBJ = "NYU_0051016"


def thumb(draw, size=(2.0, 2.0), dpi=220):
    f = plt.figure(figsize=size, facecolor="white")
    draw(f)
    buf = io.BytesIO(); f.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(f); buf.seek(0); return plt.imread(buf)


# ---------------------------------------------------------------- thumbnails from real data
vols = os.path.join(IMG, "small", "ABIDE_NYU_0051016_preproc_3vols.nii.gz")
def t_stack(f):
    for k in range(3):
        ax = f.add_axes([0.04 + 0.13 * k, 0.04 + 0.13 * (2 - k), 0.7, 0.7])
        v = np.asarray(nib.load(os.path.join(IMG, f"ABIDE_{SUBJ}_func_mean.nii.gz")).dataobj)[:, :, [26, 30, 36][k]].T
        ax.imshow(v, cmap="gray", origin="lower"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#8a8984"); s.set_linewidth(0.8)
def t_t1(f):
    ax = f.add_axes([0, 0, 1, 1])
    T1 = nib.as_closest_canonical(nib.load(os.path.join(IMG, f"ABIDE_{SUBJ}_T1w_raw.nii.gz")))
    T = np.asarray(T1.dataobj, dtype=np.float32); sl = T[T.shape[0] // 2 + 4].T
    ax.imshow(sl, cmap="gray", origin="lower", vmin=np.percentile(sl, 1), vmax=np.percentile(sl, 99.5),
              aspect=T1.header.get_zooms()[2] / T1.header.get_zooms()[1]); ax.axis("off")
def t_atlas(f):
    ax = f.add_axes([0, 0, 1, 1])
    rng = np.random.default_rng(3); cm = ListedColormap(plt.cm.tab20(np.linspace(0, 1, 20))[rng.permutation(20)])
    plotting.plot_roi(os.path.join(IMG, "cc200_roi_atlas.nii.gz"), bg_img=datasets.load_mni152_template(resolution=1),
                      display_mode="z", cut_coords=[8], axes=ax, cmap=cm, alpha=0.85, annotate=False,
                      draw_cross=False, black_bg=False, colorbar=False)
ts = np.loadtxt(os.path.join(ROI, f"{SUBJ}_rois_cc200.1D"))
def t_ts(f):
    ax = f.add_axes([0.02, 0.02, 0.96, 0.96]); z = (ts - ts.mean(0)) / (ts.std(0) + 1e-9)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    for i, r in enumerate([0, 3, 8, 30, 60, 120]):
        ax.plot(z[:120, r] / 3.2 + i, lw=0.9, color=cols[i])
    ax.axis("off")
def t_fc(f):
    ax = f.add_axes([0, 0, 1, 1]); r = np.corrcoef(ts.T); np.fill_diagonal(r, 0)
    ax.imshow(np.arctanh(np.clip(r, -0.99, 0.99)), cmap=DIV, vmin=-1.2, vmax=1.2, interpolation="nearest"); ax.axis("off")
def t_vec(f):
    ax = f.add_axes([0.02, 0.1, 0.96, 0.8]); r = np.corrcoef(ts.T); v = np.arctanh(np.clip(r[np.triu_indices(200, 1)], -.99, .99))
    ax.imshow(v[:600].reshape(6, 100), cmap=DIV, vmin=-1.2, vmax=1.2, aspect="auto", interpolation="nearest"); ax.axis("off")

TH = {"t1": thumb(t_t1), "stack": thumb(t_stack), "atlas": thumb(t_atlas), "ts": thumb(t_ts, (2.4, 1.8)),
      "fc": thumb(t_fc), "vec": thumb(t_vec, (2.4, 0.9))}

# ---------------------------------------------------------------- layout
fig, ax = plt.subplots(figsize=(7.4, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(-1, 80.5); ax.axis("off")
def box(x, y, w, h, title, body, fc, ec, fs=6.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    if title: ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.0, fontweight="bold", color=INK)
    if body: ax.text(x + w / 2, y + h - 5.4, body, ha="center", va="top", fontsize=fs, color=MUTED, linespacing=1.3)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=MUTED))
def put(img, x, y, w, h):
    ia = fig.add_axes([0, 0, 1, 1]); ia.axis("off")
    # map data coords to figure coords
    p0 = fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    p1 = fig.transFigure.inverted().transform(ax.transData.transform((x + w, y + h)))
    ia.set_position([p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]])
    ia.imshow(img, aspect="auto" if img.shape[1] / img.shape[0] > 2 else "equal")

gr, gr_e = "#f3f2ef", "#8a8984"; blue, blue_e = "#e8f1fb", "#2a78d6"; org, org_e = "#fdece4", "#eb6834"; vio, vio_e = "#ecebfa", "#6250d6"
ax.text(0.5, 80.0, "A   From resting-state fMRI to FC-derived phenotypes (T1w is used only for anatomical registration)", fontsize=7.4,
        fontweight="bold", color=INK, va="top")
stages = [("t1", "1  Raw MRI", "T1w: registration only\nrs-fMRI: model input"), ("stack", "2  Preprocessing", "C-PAC (ABIDE) /\nAthena (ADHD-200),\nMNI registration"),
          ("atlas", "3  Parcellation", "CC200 atlas: 200 ROIs\n(190 in ADHD-200)"), ("ts", "4  ROI time series", "mean BOLD per ROI\n(T × 200)"),
          ("fc", "5  Connectivity", "Pearson r → Fisher z\n200 × 200 matrix"), ("vec", "6  IDP vector", "upper triangle: 19,900\n(17,955) edges / subject")]
W = 15.4
for i, (k, t, b) in enumerate(stages):
    x = 0.6 + i * 16.6
    box(x, 45.6, W, 30.5, t, "", gr, gr_e)
    img = TH[k]; ar = img.shape[1] / img.shape[0]
    iw = W - 2.4; ih = min(15.5, iw / ar * (7.4 / 5.6) * (78 / 100) ** -1 * 0.78)
    put(img, x + 1.2, 53.1 + (15.5 - ih) / 2, iw, ih)
    ax.text(x + W / 2, 52.3, b, ha="center", va="top", fontsize=5.5, color=MUTED, linespacing=1.25)
    if i < 5: arrow(x + W + 0.15, 60.6, x + W + 1.15, 60.6)

ax.text(0.5, 43.6, "B   Leakage-controlled learning (everything fitted on the training fold only)", fontsize=7.4,
        fontweight="bold", color=INK, va="top")
adhd = (f"ADHD-200: N = {coh['adhd_n']}, {coh['adhd_sites']} sites\n{coh['adhd_adhd']} ADHD / {coh['adhd_tdc']} TDC"
        if coh else "ADHD-200 (external)")
box(0.6, 17.5, 18.5, 21.5, "QC analytic cohorts", f"ABIDE-I: N = 873, 17 sites\n405 ASD / 468 TD\n\n{adhd}", gr, gr_e, 5.6)
box(21.6, 17.5, 21, 21.5, "Train-fold preprocessing", "standardisation;\nPLS-auto: supervised\ncompression, inner 5-fold\nCV selects k ∈ {2,3,5,8,12}\n(or unsupervised PCA-k)", blue, blue_e, 5.6)
box(45.1, 17.5, 17.5, 21.5, "Compact table", "k latent connectome\nscores / subject\n(k ≪ 19,900)\ncontext = labelled\ntraining subjects", blue, blue_e, 5.6)
box(65.1, 17.5, 16.5, 21.5, "TabICL (v1)", "frozen; in-context\nprediction without\nparameter updates\n(TabICLv2: secondary)", org, org_e, 5.6)
box(84.1, 17.5, 15.3, 21.5, "Baselines", "LR, LR-full\n(all edges), RF,\nXGBoost, TabM", gr, gr_e, 5.6)
for x in [19.4, 42.9, 62.9]: arrow(x, 28.2, x + 2.0, 28.2)
ax.add_patch(FancyArrowPatch((81.9, 28.2), (83.9, 28.2), arrowstyle="<|-|>", mutation_scale=8, lw=0.9, color=MUTED))
items = [("E1", "main benchmark\n3×10-fold CV"), ("E2", "few-shot\nn = 10–200"), ("E3", "leave-one-\nsite-out"), ("E4", "feature\ngroups"),
         ("E5", "ADHD-200\nreplication"), ("S", "age + sex\ncontrol"), ("S", "permutation\ntests"), ("S", "ablations\n(Supplement)")]
box(0.6, 1.5, 98.8, 13.5, "", "", "#fcfcfb", vio_e)
ax.text(1.8, 14.2, "C   Evaluation (identical folds and context draws for every model; fold-matched tests, bootstrap CIs)",
        fontsize=6.6, fontweight="bold", color=INK, va="top")
for i, (k, t) in enumerate(items):
    x = 2.0 + i * 12.2
    ax.add_patch(FancyBboxPatch((x, 2.6), 10.8, 8.0, boxstyle="round,pad=0.2,rounding_size=0.8", fc=vio, ec="none"))
    ax.text(x + 5.4, 9.9, k, ha="center", va="top", fontsize=6.6, fontweight="bold", color=vio_e)
    ax.text(x + 5.4, 7.2, t, ha="center", va="top", fontsize=5.3, color=MUTED, linespacing=1.2)
fig.savefig(os.path.join(out, "fig1_pipeline.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(out, "fig1_pipeline.pdf"), bbox_inches="tight")
