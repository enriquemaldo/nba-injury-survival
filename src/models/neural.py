"""DeepSurv and Transformer + DeepSurv (proposal Baseline 4 and the proposed model).

Both are trained with the Cox partial likelihood (Breslow ties, risk set =
the minibatch), so they output a log-risk score like the Cox baseline.

    DeepSurv        MLP on the tabular (static) features.
    Transformer     The player's previous N games (sequence of per-game vectors +
                    days before the prediction) -> linear projection -> [CLS] token
                    + learned positions -> TransformerEncoder -> CLS output,
                    concatenated with static features -> MLP risk head.

Runs (python -m src.models.neural --runs ...):
    deepsurv          DeepSurv, all tabular features
    tf_full           Transformer, 20 games + all tabular features
    tf_seq            Transformer, 20 games + tabular features EXCEPT the aggregated
                      workload ones (tests sequence vs. hand-made aggregates)
    tf_len5, tf_len10 Transformer with 5 / 10 previous games (sequence-length study)
    ablations         DeepSurv without each feature family
Each run trains 3 seeds; results report mean +/- sd and a seed-averaged ensemble.

Training: training seasons only. 10% of training PLAYERS are held out for early
stopping (validation season is never used for training or stopping). Fixed
hyperparameters (no tuning on the validation season). The test season is not
touched.

    python -m src.models.neural --smoke          # 1-minute end-to-end check
    python -m src.models.neural                  # everything (~45-60 min on a laptop CPU)
    python -m src.models.neural --runs deepsurv tf_full --seeds 1
"""
import argparse
import json
import time as _time

import numpy as np
import pandas as pd

from src.config import CLASSIF_DAYS, PROCESSED, TRAIN_SEASONS
from src.models.metrics import auc_at, brier_at, c_index
from src.sequence.build import OUT as SEQ_PATH

OUT = PROCESSED / "neural"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(exist_ok=True)
H = CLASSIF_DAYS
FAMILIES = ["workload", "schedule", "travel", "history", "bio"]

CONFIG = {  # fixed in advance
    "batch_size": 4096, "lr": 1e-3, "weight_decay": 1e-4, "max_epochs": 30, "patience": 4,
    "deepsurv_hidden": (64, 32), "dropout": 0.2,
    "d_model": 32, "n_heads": 4, "n_layers": 2, "ff": 64, "tf_dropout": 0.1, "head_hidden": 32,
    "inner_frac": 0.10,
}
RUNS = {
    "deepsurv": dict(kind="deepsurv", n_seq=0, drop=[]),
    "tf_full": dict(kind="transformer", n_seq=20, drop=[]),
    "tf_seq": dict(kind="transformer", n_seq=20, drop=["workload"]),
    "tf_len5": dict(kind="transformer", n_seq=5, drop=[]),
    "tf_len10": dict(kind="transformer", n_seq=10, drop=[]),
    **{f"deepsurv_no_{f}": dict(kind="deepsurv", n_seq=0, drop=[f]) for f in FAMILIES},
}


# ===================================================== numpy helpers (tested) ==
def load(path=SEQ_PATH):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def static_columns(data, drop):
    keep = ~np.isin(data["static_family"], drop)
    return np.flatnonzero(keep)


def splits(data, seed, inner_frac=CONFIG["inner_frac"]):
    """Training rows (training seasons) split by PLAYER into fit / early-stopping sets."""
    train = np.isin(data["season"], TRAIN_SEASONS)
    players = np.unique(data["player_id"][train])
    rng = np.random.default_rng(seed)
    inner_players = rng.choice(players, int(len(players) * inner_frac), replace=False)
    inner = train & np.isin(data["player_id"], inner_players)
    fit = train & ~inner
    val = data["split"] == "val"
    return np.flatnonzero(fit), np.flatnonzero(inner), np.flatnonzero(val)


def breslow_cumhaz(eta, time, event, h):
    """Breslow baseline cumulative hazard at day h from training log-risks."""
    eta, time, event = np.asarray(eta, float), np.asarray(time, float), np.asarray(event, bool)
    shift = eta.max()
    w = np.exp(eta - shift)
    ut = np.unique(time[event & (time <= h)])
    H0 = 0.0
    for t in ut:
        H0 += (event & (time == t)).sum() / (w[time >= t].sum() * np.exp(shift))
    return H0


def evaluate(time, event, eta, prob, pid=None):
    r = {"c_index": c_index(time, event, eta), "auc_14d": auc_at(time, event, eta, H)}
    if prob is not None:
        r["brier_14d"] = brier_at(time, event, prob, H)
    return r


# ========================================================= torch parts =======
def _torch():
    import torch
    return torch


def cox_loss(eta, time, event):
    """Negative Breslow partial log-likelihood per event; risk set = the batch.
    Ties: the risk set of time t is every row with time >= t."""
    torch = _torch()
    order = torch.argsort(time, descending=True)
    eta, time, event = eta[order], time[order], event[order]
    lcse = torch.logcumsumexp(eta, dim=0)
    asc = torch.flip(time, dims=[0]).contiguous()
    n_ge = time.shape[0] - torch.searchsorted(asc, time.contiguous(), right=False)
    log_risk = lcse[n_ge - 1]
    ev = event > 0
    n_ev = ev.sum()
    if n_ev == 0:
        return (eta * 0).sum()
    return -(eta[ev] - log_risk[ev]).sum() / n_ev


def build_model(kind, n_static, n_step, n_seq):
    torch = _torch()
    nn = torch.nn
    c = CONFIG

    class DeepSurv(nn.Module):
        def __init__(self):
            super().__init__()
            layers, d = [], n_static
            for h in c["deepsurv_hidden"]:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(c["dropout"])]
                d = h
            layers += [nn.Linear(d, 1, bias=False)]          # a bias cancels in the Cox loss
            self.net = nn.Sequential(*layers)

        def forward(self, static, seq=None, pad=None, dt=None):
            return self.net(static).squeeze(-1)

    class TransformerSurv(nn.Module):
        def __init__(self):
            super().__init__()
            d = c["d_model"]
            self.inp = nn.Linear(n_step + 1, d)               # +1: log days before the prediction
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.pos = nn.Parameter(torch.randn(1, n_seq + 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, c["n_heads"], c["ff"], c["tf_dropout"], batch_first=True)
            self.enc = nn.TransformerEncoder(layer, c["n_layers"], enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Linear(d + n_static, c["head_hidden"]), nn.ReLU(),
                                      nn.Dropout(c["tf_dropout"]), nn.Linear(c["head_hidden"], 1, bias=False))

        def encode(self, seq, pad, dt):
            x = self.inp(torch.cat([seq, dt.unsqueeze(-1)], -1))
            b = x.shape[0]
            x = torch.cat([self.cls.expand(b, -1, -1), x], 1) + self.pos
            mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=pad.device), pad], 1)  # CLS never masked
            return self.enc(x, src_key_padding_mask=mask)[:, 0]

        def forward(self, static, seq, pad, dt):
            return self.head(torch.cat([self.encode(seq, pad, dt), static], -1)).squeeze(-1)

    return DeepSurv() if kind == "deepsurv" else TransformerSurv()


class Batcher:
    """Turns row indices into model inputs (gathering each row's previous games)."""

    def __init__(self, data, cols, n_seq, device):
        torch = _torch()
        self.torch, self.device, self.n_seq = torch, device, n_seq
        self.static = torch.from_numpy(data["static_X"][:, cols]).to(device)
        self.time = torch.from_numpy(data["time"].astype(np.float32)).to(device)
        self.event = torch.from_numpy(data["event"].astype(np.float32)).to(device)
        if n_seq:
            self.game_X = torch.from_numpy(data["game_X"]).to(device)
            self.seq_idx = torch.from_numpy(data["seq_idx"][:, -n_seq:].astype(np.int64)).to(device)
            self.seq_dt = torch.from_numpy(data["seq_dt"][:, -n_seq:]).to(device)

    def __call__(self, rows):
        rows = self.torch.as_tensor(rows, device=self.device)
        out = {"static": self.static[rows]}
        if self.n_seq:
            idx = self.seq_idx[rows]
            pad = idx < 0
            seq = self.game_X[idx.clamp(min=0)] * (~pad).unsqueeze(-1)
            out.update(seq=seq, pad=pad, dt=self.seq_dt[rows])
        return out


def predict(model, batcher, rows, bs=8192):
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            out.append(model(**batcher(rows[i:i + bs])).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_one(data, run, seed, device, smoke=False, log=print):
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = RUNS[run]
    cols = static_columns(data, spec["drop"])
    fit, inner, val = splits(data, seed)
    if smoke:
        rng = np.random.default_rng(seed)
        fit, inner, val = (rng.choice(a, min(len(a), n), replace=False) for a, n in ((fit, 20000), (inner, 4000), (val, 4000)))
    batcher = Batcher(data, cols, spec["n_seq"], device)
    model = build_model(spec["kind"], len(cols), data["game_X"].shape[1], spec["n_seq"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    t_in, e_in = data["time"][inner], data["event"][inner]
    best, best_state, bad = -1.0, None, 0
    epochs = 2 if smoke else CONFIG["max_epochs"]
    rng = np.random.default_rng(seed)
    t0 = _time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(fit)
        losses = []
        for i in range(0, len(perm), CONFIG["batch_size"]):
            b = perm[i:i + CONFIG["batch_size"]]
            x = batcher(b)
            loss = cox_loss(model(**x), batcher.time[torch.as_tensor(b, device=device)],
                            batcher.event[torch.as_tensor(b, device=device)])
            if not torch.isfinite(loss):
                raise RuntimeError(f"{run}: non-finite loss at epoch {ep}")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        c_in = c_index(t_in, e_in, predict(model, batcher, inner))
        log(f"    [{run} s{seed}] epoch {ep + 1:2d}  loss {np.mean(losses):.4f}  inner C {c_in:.4f}  "
            f"({_time.time() - t0:.0f}s)")
        if c_in > best + 1e-4:
            best, bad = c_in, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CONFIG["patience"]:
                break
    model.load_state_dict(best_state)
    eta_fit = predict(model, batcher, fit)
    H0 = breslow_cumhaz(eta_fit, data["time"][fit], data["event"][fit], H)
    eta_val = predict(model, batcher, val)
    prob = 1 - np.exp(-H0 * np.exp(eta_val))
    res = evaluate(data["time"][val], data["event"][val], eta_val, prob)
    res.update(run=run, seed=seed, epochs=ep + 1, inner_c=best, n_static=len(cols), n_seq=spec["n_seq"])
    if not smoke:
        torch.save({"state_dict": model.state_dict(), "run": run, "seed": seed, "cols": cols.tolist(),
                    "config": CONFIG, "spec": spec}, OUT / "models" / f"{run}_s{seed}.pt")
    return res, val, eta_val, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["deepsurv", "tf_full", "tf_seq", "tf_len5", "tf_len10", "ablations"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu", help="cpu (default) or mps (Apple GPU)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check (~1 min)")
    args = ap.parse_args()
    torch = _torch()
    runs = [r for a in args.runs for r in ([f"deepsurv_no_{f}" for f in FAMILIES] if a == "ablations" else [a])]
    if args.smoke:
        runs, args.seeds = ["deepsurv", "tf_full"], 1
    data = load()
    present = set(data["static_family"])
    runs = [r for r in runs if not (r.startswith("deepsurv_no_") and r.removeprefix("deepsurv_no_") not in present)]
    print(f"torch {torch.__version__} on {args.device}; runs: {runs}; seeds: {args.seeds}")

    results, preds = [], {}
    for run in runs:
        for seed in range(args.seeds):
            res, val, eta, prob = train_one(data, run, seed, args.device, smoke=args.smoke)
            results.append(res)
            preds[f"{run}_s{seed}"] = eta
            print(f"  {run} seed {seed}: val C {res['c_index']:.4f}  AUC14 {res['auc_14d']:.4f}  "
                  f"({res['epochs']} epochs)")
    if args.smoke:
        print("SMOKE TEST PASSED - the full run should work.")
        return
    res = pd.DataFrame(results)
    old = OUT / "neural_results.csv"
    if old.exists():  # keep results of runs not repeated now
        prev = pd.read_csv(old)
        res = pd.concat([prev[~prev.run.isin(res.run.unique())], res], ignore_index=True)
    res.to_csv(old, index=False)
    p_path = OUT / "val_predictions.csv"
    pr = pd.DataFrame({"PLAYER_ID": data["player_id"][val], "GAME_ID": data["game_id"][val]})
    if p_path.exists():
        prev = pd.read_csv(p_path)
        if len(prev) == len(pr) and (prev.GAME_ID.to_numpy() == pr.GAME_ID.to_numpy()).all():
            pr = prev.drop(columns=[c for c in prev.columns if c in preds])
    for k, v in preds.items():
        pr[k] = v
    pr.to_csv(p_path, index=False)
    json.dump(CONFIG, open(OUT / "config.json", "w"), indent=2)
    print(res.groupby("run")[["c_index", "auc_14d", "brier_14d"]].agg(["mean", "std"]).round(4).to_string())
    print(f"saved {old}, {p_path}")


if __name__ == "__main__":
    main()
