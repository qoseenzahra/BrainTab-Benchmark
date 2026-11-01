"""Data loaders for ABIDE-I (PCP, C-PAC, CC200) and the ADHD-200 nilearn subset."""
import os, glob
import numpy as np
import pandas as pd

QC_COLS = ["qc_rater_1", "qc_anat_rater_2", "qc_func_rater_2",
           "qc_anat_rater_3", "qc_func_rater_3"]


def load_abide_timeseries(roi_dir, pheno_csv, min_tr=50):
    """Return list of ROI time series (T x 200) and an aligned phenotype frame.

    Subjects are kept when (i) the phenotype row exists, (ii) the file has 200
    ROIs, (iii) at least `min_tr` volumes exist. ROIs with zero variance are
    recorded so that downstream code can handle them deterministically.
    """
    pheno = pd.read_csv(pheno_csv)
    pheno = pheno.set_index("FILE_ID")
    ts, rows = [], []
    for f in sorted(glob.glob(os.path.join(roi_dir, "*_rois_cc200.1D"))):
        fid = os.path.basename(f).replace("_rois_cc200.1D", "")
        if fid not in pheno.index:
            continue
        x = np.loadtxt(f, skiprows=1)
        if x.ndim != 2 or x.shape[1] != 200 or x.shape[0] < min_tr:
            continue
        ts.append(x.astype(np.float64))
        r = pheno.loc[fid].to_dict(); r["FILE_ID"] = fid; r["n_tr"] = x.shape[0]
        r["n_zero_roi"] = int((x.std(0) < 1e-8).sum())
        rows.append(r)
    meta = pd.DataFrame(rows).reset_index(drop=True)
    # labels: ABIDE DX_GROUP 1 = ASD, 2 = TD  ->  y = 1 for ASD
    meta["y"] = (meta["DX_GROUP"] == 1).astype(int)
    qc_ok = np.ones(len(meta), bool)
    for c in QC_COLS:
        qc_ok &= (meta[c].astype(str).str.lower() != "fail").values
    meta["qc_pass"] = qc_ok
    meta["site"] = meta["SITE_ID"].astype(str).str.replace(r"_\d$", "", regex=True)
    return ts, meta
