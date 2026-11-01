"""Build connectivity features for the full ADHD-200 release (Neuro Bureau Athena pipeline, CC200 time courses).

Usage: python scripts/08_build_adhd_full.py <extracted_tc_dir> [<more_dirs> ...] --pheno <adhd200_preprocessed_phenotypics.tsv> [--prefix snwmrda]
Inclusion: Athena QC pass (QC_Athena == 1), known diagnosis (DX in 0,1,2,3; Brown 'pending' excluded),
first resting-state run of the first session, >= 50 volumes. Label: 1 = ADHD (any subtype), 0 = typically developing (TDC).
Writes data/adhd_full_features.npz (fisher_z: N x R(R-1)/2; R = 190 for the Athena CC200 parcellation) and data/adhd_full_meta.csv.
"""
import os, sys, re, argparse
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITES = {1: "Peking", 2: "Brown", 3: "KKI", 4: "NeuroIMAGE", 5: "NYU", 6: "OHSU", 7: "Pittsburgh", 8: "WashU"}
ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--pheno", required=True)
ap.add_argument("--prefix", default="snwmrda",
                help="Athena derivative: snwmrda = no band-pass filter (matches ABIDE nofilt), sfnwmrda = band-pass filtered")
a = ap.parse_args()

def read_tc(path):
    rows = []
    for line in open(path, errors="ignore"):
        tok = line.split()
        if not tok or not re.match(r"^-?[0-9.]", tok[-1]):
            continue
        num = []
        for t in tok:
            try: num.append(float(t))
            except ValueError: num = []          # drop leading non-numeric tokens (file name)
        rows.append(num)
    L = max(len(r) for r in rows); rows = [r for r in rows if len(r) == L]
    M = np.array(rows)
    if M.shape[1] == 202:                         # File, Sub-brick, 200 means  -> sub-brick index kept
        M = M[:, 2:]
    elif M.shape[1] == 201:
        M = M[:, 1:]
    return M

files = {}
for d in a.dirs:
    for root, _, fs in os.walk(d):
        for f in fs:
            if re.search(r"cc200", f, re.I) and f.lower().endswith((".1d", ".txt", ".csv")) and f.startswith(a.prefix):
                m = re.search(r"(\d{7})", f) or re.search(r"(\d{7})", root)
                if not m: continue
                sid = int(m.group(1)); run = re.search(r"rest_(\d)", f); ses = re.search(r"session_(\d)", f)
                rank = (int(ses.group(1)) if ses else 1, int(run.group(1)) if run else 1)
                if sid not in files or rank < files[sid][0]:
                    files[sid] = (rank, os.path.join(root, f))
print("time-course files:", len(files))
ph = pd.read_csv(a.pheno, sep="\t")
ph = ph.rename(columns={"ScanDir ID": "sid"})
ph["DXn"] = pd.to_numeric(ph.DX, errors="coerce")
keep = ph[(ph.QC_Athena == 1) & ph.DXn.isin([0, 1, 2, 3]) & ph.sid.isin(files)].copy()
print("phenotypic rows:", len(ph), "| QC pass + known DX + time series:", len(keep))
R_EXPECT = int(pd.Series([read_tc(files[s_][1]).shape[1] for s_ in list(keep.sid)[:20]]).mode()[0])
print("regions per time-course file:", R_EXPECT, "(Athena CC200 parcellation: 190 non-empty parcels)")
iu = np.triu_indices(R_EXPECT, 1)
X, rows = [], []
for _, r in keep.iterrows():
    ts = read_tc(files[r.sid][1])
    if ts.shape[1] != R_EXPECT or ts.shape[0] < 50:
        print("skip", r.sid, ts.shape); continue
    sd = ts.std(0); z = (ts - ts.mean(0)) / np.where(sd > 0, sd, 1)
    C = (z.T @ z) / len(z); C[:, sd == 0] = 0; C[sd == 0, :] = 0
    X.append(np.arctanh(np.clip(C[iu], -0.999999, 0.999999)).astype(np.float32))
    rows.append(dict(sid=r.sid, site=SITES[int(r.Site)], y=int(r.DXn > 0), dx=int(r.DXn), age=float(r.Age),
                     male=int(r.Gender == 1), n_tr=len(ts), n_zero_roi=int((sd == 0).sum()), file=os.path.basename(files[r.sid][1])))
X = np.stack(X); meta = pd.DataFrame(rows)
np.savez_compressed(os.path.join(ROOT, "data", "adhd_full_features.npz"), fisher_z=X, n_regions=R_EXPECT)
meta.to_csv(os.path.join(ROOT, "data", "adhd_full_meta.csv"), index=False)
print("N =", len(meta), "| ADHD", int(meta.y.sum()), "TDC", int((1 - meta.y).sum()))
print(meta.groupby("site").agg(n=("y", "size"), adhd=("y", "sum"), age=("age", "mean"), male=("male", "mean")).round(2))
print("checksum", float(np.abs(X[:, :100]).sum()).__round__(4))
