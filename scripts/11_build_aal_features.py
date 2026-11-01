"""AAL-atlas sensitivity analysis (revision): download PCP C-PAC (nofilt_noglobal) AAL ROI time series for the
QC-pass ABIDE-I participants and build data/abide_features_aal.npz, row-aligned to data/abide_meta.csv.
Run the benchmark on it with:  BT_ABIDE_FEATURES=abide_features_aal.npz python scripts/02_run_benchmark.py main

Usage: python scripts/11_build_aal_features.py <download_dir>"""
import os, sys, time, concurrent.futures as cf
import numpy as np, pandas as pd, requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from braintab.features import fisher_z_vectors
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
URL = ("https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/nofilt_noglobal/"
       "rois_aal/{}_rois_aal.1D")
meta = pd.read_csv(os.path.join(ROOT, "data", "abide_meta.csv"))
fids = meta.loc[meta.qc_pass, "FILE_ID"].astype(str).tolist()

def get(fid):
    dest = os.path.join(OUT, f"{fid}_rois_aal.1D")
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    for _ in range(4):
        try:
            r = requests.get(URL.format(fid), timeout=120)
            if r.status_code == 200 and len(r.content) > 1000:
                open(dest, "wb").write(r.content); return True
        except Exception:
            pass
        time.sleep(3)
    return False

with cf.ThreadPoolExecutor(8) as ex:
    ok = list(ex.map(get, fids))
assert all(ok), "some AAL files could not be downloaded"
ts = [np.loadtxt(os.path.join(OUT, f"{f}_rois_aal.1D"), skiprows=1).astype(np.float64) for f in fids]
fz = fisher_z_vectors(ts)
full = np.zeros((len(meta), fz.shape[1]), np.float32); full[meta.qc_pass.values] = fz
np.savez_compressed(os.path.join(ROOT, "data", "abide_features_aal.npz"), fisher_z=full)
print("saved", full.shape)
