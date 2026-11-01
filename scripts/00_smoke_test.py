"""Quick environment + model smoke test on one ABIDE split (writes results/smoke.txt)."""
import sys, os, time, platform
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
out = open(os.path.join(ROOT, "results", "smoke.txt"), "w")
def w(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.write(s + "\n"); out.flush()
w("python", platform.python_version())
for mod in ["numpy", "sklearn", "xgboost", "torch", "tabicl", "tabm"]:
    try:
        m = __import__(mod); w(mod, getattr(m, "__version__", "?"))
    except Exception as e:
        w(mod, "IMPORT FAIL", repr(e)[:200])
import importlib.util
spec = importlib.util.spec_from_file_location("rb", os.path.join(ROOT, "scripts", "02_run_benchmark.py"))
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
X, y, site, cov, meta = rb.load_abide()
w("ABIDE", X.shape, int(y.sum()), "checksum", float(np.round(X[:, :100].astype(np.float64).sum(), 4)))
from sklearn.model_selection import StratifiedKFold
tr, te = next(iter(StratifiedKFold(10, shuffle=True, random_state=0).split(X, rb.strat_labels(site, y))))
A, B, info = rb.prepare(X[tr], X[te], y[tr], "plsauto")
w("plsauto k", info)
from braintab.metrics import all_metrics
for m in ["LR", "XGBoost", "TabICL-v1", "TabICL-v2", "TabM"]:
    try:
        t0 = time.time(); p, sec = rb.fit_predict(m, A, y[tr], B)
        w(m, "OK", {k: round(v, 3) for k, v in all_metrics(y[te], p).items()}, f"{sec:.1f}s")
    except Exception as e:
        import traceback; w(m, "FAIL", traceback.format_exc()[-1500:])
w("SMOKE_DONE")
