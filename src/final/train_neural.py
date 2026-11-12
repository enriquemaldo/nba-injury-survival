"""Neural part of the final evaluation, run as its own process by src.final.evaluate.

PyTorch and XGBoost each bring their own OpenMP threading library; on macOS,
using both in one process can freeze. So the final neural models are trained
here, in a separate process, and their held-out predictions are saved for
src.final.evaluate. Do not run this by hand.
"""
import argparse
import time as _time

import numpy as np
import pandas as pd

from src.final.evaluate import FIT_SEASONS, LABELS, MODEL_DIR, NEURAL, OUT, SEEDS, SETS
from src.models.neural import load, train_one

PRED = OUT / "neural_predictions.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    mask = np.isin(data["split"], list(SETS))
    out, info = {"rows": np.flatnonzero(mask)}, []
    for key, run in NEURAL.items():
        for s in range(SEEDS):
            t0 = _time.time()
            res, rows, eta, prob = train_one(data, run, s, args.device, log=lambda *a: None,
                                             train_seasons=FIT_SEASONS, eval_mask=mask, save_dir=MODEL_DIR)
            if not np.array_equal(rows, out["rows"]):
                raise RuntimeError("scored rows are not in landmark order")
            out[f"{key}_eta_s{s}"], out[f"{key}_prob_s{s}"] = eta, prob
            info.append({"model": key, "seed": s, "epochs": res["epochs"], "inner_c": res["inner_c"]})
            print(f"  {LABELS[key]} seed {s}: {res['epochs']} epochs ({_time.time() - t0:.0f}s)", flush=True)
    np.savez(PRED, **out)
    pd.DataFrame(info).to_csv(OUT / "neural_training.csv", index=False)


if __name__ == "__main__":
    main()
