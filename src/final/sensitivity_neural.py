"""Transformer variants for the sensitivity analysis, run as its own process by
src.final.sensitivity (PyTorch stays out of the main process). Do not run by hand.

Each variant is the final Transformer + DeepSurv (5 games) with ONE change, refit
with 3 seeds and scored on the held-out seasons. The final models are not touched:
variants are saved under data/processed/final/sensitivity_models/.
"""
import argparse
import time as _time

import numpy as np
import pandas as pd

from src.final.evaluate import FIT_SEASONS, OUT, SEEDS, SETS
from src.models import neural

PRED = OUT / "sensitivity_neural.npz"
ATYPICAL = ["2019-20", "2020-21"]
VARIANTS = {
    "tf5_excl_2019_21": {"train_seasons": [s for s in FIT_SEASONS if s not in ATYPICAL]},
    "tf5_small": {"config": {"d_model": 16, "n_heads": 2, "n_layers": 1, "ff": 32, "head_hidden": 16}},
    "tf5_large": {"config": {"d_model": 64, "n_heads": 4, "n_layers": 3, "ff": 128, "head_hidden": 64}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    data = neural.load()
    mask = np.isin(data["split"], list(SETS))
    out, info = {"rows": np.flatnonzero(mask)}, []
    base_config = dict(neural.CONFIG)
    for name, v in VARIANTS.items():
        neural.CONFIG.clear()
        neural.CONFIG.update(base_config, **v.get("config", {}))   # build_model reads CONFIG at call time
        save = OUT / "sensitivity_models" / name
        save.mkdir(parents=True, exist_ok=True)
        for s in range(args.seeds):
            t0 = _time.time()
            res, rows, eta, prob = neural.train_one(data, "tf_len5", s, args.device, log=lambda *a: None,
                                                    train_seasons=v.get("train_seasons", FIT_SEASONS),
                                                    eval_mask=mask, save_dir=save)
            if not np.array_equal(rows, out["rows"]):
                raise RuntimeError("scored rows are not in landmark order")
            out[f"{name}_eta_s{s}"], out[f"{name}_prob_s{s}"] = eta, prob
            info.append({"variant": name, "seed": s, "epochs": res["epochs"], "inner_c": res["inner_c"]})
            print(f"  {name} seed {s}: {res['epochs']} epochs ({_time.time() - t0:.0f}s)", flush=True)
    neural.CONFIG.clear()
    neural.CONFIG.update(base_config)
    np.savez(PRED, **out)
    pd.DataFrame(info).to_csv(OUT / "sensitivity_training.csv", index=False)


if __name__ == "__main__":
    main()
