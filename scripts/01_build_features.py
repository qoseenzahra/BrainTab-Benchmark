"""Step 1: build ABIDE connectivity features from CC200 ROI time series.
Outputs data/abide_features.npz and data/abide_meta.csv."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from braintab.data import load_abide_timeseries
from braintab.features import fisher_z_vectors, ledoit_wolf

ROI_DIR = sys.argv[1]; PHENO = sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else "data"
t0 = time.time()
ts, meta = load_abide_timeseries(ROI_DIR, PHENO)
print("subjects", len(ts), "QC-pass", int(meta.qc_pass.sum()), f"{time.time()-t0:.0f}s")
fz = fisher_z_vectors(ts)
covs = np.stack([ledoit_wolf(x) for x in ts]).astype(np.float32)
os.makedirs(OUT, exist_ok=True)
np.savez_compressed(os.path.join(OUT, "abide_features.npz"), fisher_z=fz, cov=covs)
meta.to_csv(os.path.join(OUT, "abide_meta.csv"), index=False)
print("done", fz.shape, covs.shape, f"{time.time()-t0:.0f}s")
