"""Save the pre-registered window features (mean of resid layers 14 to 19) for both probe sites from the activation
cache into one small file, results/probe_live_inert/features_window14_19.npz (about 20 MB), so controls.py can run
on any laptop without the 1.5 GB cache. Run from the repo root after capture.py."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
cache = ROOT / "interp/cache/behavior_v1_probe"
index = json.loads((cache / "index.json").read_text())
ids = sorted(index)
sites = ["memory_result_end", "entry0.occ0.last"]
X = np.full((len(sites), len(ids), 2880), np.nan, dtype=np.float32)
for k, i in enumerate(ids):
    r = np.load(cache / f"{i}.npz")["resid"].astype(np.float32)
    names = index[i]["position_names"]
    for s, site in enumerate(sites):
        if site in names:
            X[s, k] = r[names.index(site), 14:20].mean(0)
out = ROOT / "results/probe_live_inert/features_window14_19.npz"
np.savez_compressed(out, ids=np.array(ids), sites=np.array(sites), X=X.astype(np.float16))
print(len(ids), "requests saved to", out, "shape", X.shape)
