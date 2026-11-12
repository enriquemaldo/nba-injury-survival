"""SHAP explanations of the final models (proposal section 4G).

Run after the final evaluation (python -m src.final.evaluate). Explains every
prediction in the held-out seasons (2023-24 test, 2024-25 supplementary):

  Cox                exact SHAP values. The log-hazard is linear, so feature j
                     contributes beta_j * (x_j - mean_j), mean over the fit rows.
  Transformer +      expected gradients, the SHAP estimator for differentiable
  DeepSurv (5 games) models (shap.GradientExplainer; Erion et al., 2021):
                       phi = E_{x' ~ background, a ~ U(0,1)} [(x - x') * grad f(x' + a (x - x'))]
                     over the static features, the 5 previous games' per-game
                     features and how many days before the prediction each game was.
                     A player keeps his own padding (fewer than 5 previous games
                     stay missing). The model explained is the average of the 3
                     seeds' log-hazards, so contributions are in log-hazard units:
                     exp(phi) is the factor by which that input multiplies the hazard.

Contributions are summed into feature families and, for the sequence, into game
positions (1 = most recent game). An additivity check (sum of contributions vs.
model output minus its average) is reported.

Usage:  python -m src.explain.shap_values [--samples 32] [--device cpu] [--max-rows N]
Writes: data/processed/explain/ (shap_values.npz, shap_report.md, figures)
"""
import argparse
import pickle
import time as _time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.config import PROCESSED
from src.explain.labels import FAMILY_LABELS, seq_family, seq_label, static_family, static_label
from src.final.evaluate import FIT_SEASONS, MODEL_DIR, SETS
from src.models.baselines import _md
from src.survival.dataset import LANDMARKS

OUT = PROCESSED / "explain"
FINAL_RUN = "tf_len5"
N_BG = 256
MINUTE_FEATURES = ["prev_game_min", "min_7d", "min_14d", "min_28d", "ewma_min_short", "ewma_min_long"]
FAM_COL = {"history": "#c0392b", "workload": "#1f5fae", "workload_seq": "#4f93df", "schedule": "#e07b39",
           "travel": "#8a8984", "bio": "#6c4fb3", "context": "#b5b4ae", "other": "#d9d8d4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "figure.dpi": 150})


# ======================================================= Cox (numpy, exact) ==
def cox_shap(cox, fit_df, score_df):
    """Exact SHAP for a linear log-hazard. Returns phi [n, p], base, eta, names."""
    pre, coef = cox["pre"], np.asarray(cox["coef"])
    X_fit, names = pre.transform(fit_df)
    X, _ = pre.transform(score_df)
    mu = X_fit.mean(0)
    phi = (X - mu) * coef
    return phi, float(mu @ coef), X @ coef, list(names)


# ======================================================= Transformer (torch) ==
def expected_gradients(model, x, bg, pad, n_samples, generator):
    """phi[k] for k in static / seq / dt, same shapes as x[k]. `pad` is the explained
    rows' own padding mask; masked positions get zero attribution (no gradient)."""
    import torch
    keys = [k for k in ("static", "seq", "dt") if k in x]
    b, m = x["static"].shape[0], bg["static"].shape[0]
    phi = {k: torch.zeros_like(x[k]) for k in keys}
    for _ in range(n_samples):
        j = torch.randint(0, m, (b,), generator=generator).to(x["static"].device)
        a = torch.rand(b, generator=generator).to(x["static"].device)
        inp, diff = {}, {}
        for k in keys:
            base = bg[k][j]
            diff[k] = x[k] - base
            inp[k] = (base + a.view(-1, *[1] * (x[k].dim() - 1)) * diff[k]).detach().requires_grad_(True)
        out = model(static=inp["static"], seq=inp.get("seq"), pad=pad, dt=inp.get("dt"))
        grads = torch.autograd.grad(out.sum(), [inp[k] for k in keys])
        for k, g in zip(keys, grads):
            phi[k] += g * diff[k]
    return {k: (v / n_samples).detach() for k, v in phi.items()}


def load_final_transformers(data, device):
    import torch
    from src.models.neural import build_model
    models = []
    for path in sorted(MODEL_DIR.glob(f"{FINAL_RUN}_s*.pt")):
        ck = torch.load(path, map_location=device, weights_only=False)
        cols = np.asarray(ck["cols"])
        model = build_model(ck["spec"]["kind"], len(cols), data["game_X"].shape[1], ck["spec"]["n_seq"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        models.append((model, cols, ck["spec"]["n_seq"], ck["H0"]))
    if not models:
        raise SystemExit(f"no {FINAL_RUN} models in {MODEL_DIR} - run python -m src.final.evaluate first")
    return models


def transformer_shap(data, rows, bg_rows, device, n_samples=32, batch=1024, log=print):
    import torch
    from src.models.neural import Batcher
    models = load_final_transformers(data, device)
    n_seq = models[0][2]
    out = None
    for s, (model, cols, _, _) in enumerate(models):
        bat = Batcher(data, cols, n_seq, device)
        bg = bat(bg_rows)
        g = torch.Generator().manual_seed(1000 + s)
        parts = {"static": [], "seq": [], "dt": [], "eta": []}
        t0 = _time.time()
        for i in range(0, len(rows), batch):
            x = bat(rows[i:i + batch])
            phi = expected_gradients(model, x, bg, x["pad"], n_samples, g)
            with torch.no_grad():
                parts["eta"].append(model(**x).cpu().numpy())
            for k in ("static", "seq", "dt"):
                parts[k].append(phi[k].cpu().numpy())
        res = {k: np.concatenate(v) for k, v in parts.items()}
        log(f"  seed {s}: {len(rows):,} rows explained ({_time.time() - t0:.0f}s)")
        out = res if out is None else {k: out[k] + res[k] for k in out}
    n = len(models)
    return {k: v / n for k, v in out.items()}, models, n_seq


def additivity_check(data, models, n_seq, rows, bg_rows, phi_sum, eta, device, n_check=300, seed=0):
    """E[sum phi] = f(x) - E_x'[f(x') with x's padding]. Compares both sides on a sample of rows."""
    import torch
    from src.models.neural import Batcher
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), min(n_check, len(rows)), replace=False)
    ref = np.zeros(len(pick))
    for model, cols, _, _ in models:
        bat = Batcher(data, cols, n_seq, device)
        bg = bat(bg_rows)
        with torch.no_grad():
            for j, i in enumerate(pick):
                x = bat(rows[i:i + 1])
                nb = bg["static"].shape[0]
                ref[j] += model(static=bg["static"], seq=bg["seq"], pad=x["pad"].expand(nb, -1),
                                dt=bg["dt"]).mean().item()
    ref /= len(models)
    target = eta[pick] - ref
    got = phi_sum[pick]
    return {"rows_checked": len(pick), "mean_abs_gap": float(np.mean(np.abs(got - target))),
            "sd_of_output": float(np.std(eta)), "corr": float(np.corrcoef(got, target)[0, 1])}


# ======================================================= summaries / figures ==
def tf_tables(phi_static, phi_seq, static_names, seq_names, n_seq):
    """Global importance (mean |SHAP|) per input and per family."""
    rows = [{"input": static_label(nm), "name": nm, "family": static_family(nm),
             "mean_abs": float(np.abs(phi_static[:, j]).mean())} for j, nm in enumerate(static_names)]
    per_feat = phi_seq.sum(1)                                  # sum over game positions
    rows += [{"input": seq_label(nm, n_seq), "name": f"seq:{nm}", "family": seq_family(nm),
              "mean_abs": float(np.abs(per_feat[:, j]).mean())} for j, nm in enumerate(seq_names)]
    feat = pd.DataFrame(rows).sort_values("mean_abs", ascending=False).reset_index(drop=True)
    fam_phi = {}
    for j, nm in enumerate(static_names):
        f = static_family(nm)
        fam_phi[f] = fam_phi.get(f, 0) + phi_static[:, j]
    for j, nm in enumerate(seq_names):
        f = seq_family(nm)
        fam_phi[f] = fam_phi.get(f, 0) + per_feat[:, j]
    return feat, fam_phi


def family_table(fam_phi):
    d = pd.DataFrame({"family": list(fam_phi), "mean_abs": [float(np.abs(v).mean()) for v in fam_phi.values()]})
    d["share"] = d.mean_abs / d.mean_abs.sum()
    return d.sort_values("mean_abs", ascending=False).reset_index(drop=True)


def cox_family_phi(phi, names):
    fam = {}
    for j, nm in enumerate(names):
        f = static_family(nm)
        fam[f] = fam.get(f, 0) + phi[:, j]
    return fam


def fig_top(feat, path, n=20):
    d = feat.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 0.36 * len(d) + 1.4))
    ax.barh(range(len(d)), d.mean_abs, color=[FAM_COL.get(f, "#d9d8d4") for f in d.family], height=0.7)
    ax.set_yticks(range(len(d)), d.input)
    ax.set_xlabel("Mean |SHAP| (log-hazard units)")
    seen = list(dict.fromkeys(d.family[::-1]))
    ncol = 3
    ax.set_title("Transformer + DeepSurv (5 games): most influential inputs", loc="left", color=INK,
                 fontsize=11, pad=14 + 13 * -(-len(seen) // ncol))
    handles = [Patch(color=FAM_COL.get(f, "#d9d8d4"), label=FAMILY_LABELS.get(f, f)) for f in seen]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_families(tf_fam, cox_fam, path):
    order = list(tf_fam.family)
    order += [f for f in cox_fam.family if f not in order]
    order = order[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(order) + 1.5))
    y = np.arange(len(order))
    for off, (tab, col, lab) in zip((0.18, -0.18), ((tf_fam, "#1f5fae", "Transformer + DeepSurv (5 games)"),
                                                    (cox_fam, "#8a8984", "Cox"))):
        s = tab.set_index("family").share.reindex(order).fillna(0)
        ax.barh(y + off, s.to_numpy() * 100, height=0.34, color=col, label=lab)
    ax.set_yticks(y, [FAMILY_LABELS.get(f, f) for f in order])
    ax.set_xlabel("Share of total attribution (%; mean |SHAP| of the family's summed contributions)")
    ax.set_title("What drives the predictions? Feature families", loc="left", color=INK, fontsize=11, pad=24)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_positions(phi_seq, path):
    per_pos = np.abs(phi_seq.sum(2)).mean(0)[::-1]           # index 0 = most recent game
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(np.arange(1, len(per_pos) + 1), per_pos, color="#4f93df", width=0.65)
    ax.set_xticks(np.arange(1, len(per_pos) + 1), [f"{k} game{'s' if k > 1 else ''} ago"
                                                    for k in range(1, len(per_pos) + 1)])
    ax.set_ylabel("Mean |SHAP| of the game")
    ax.set_title("Which previous games matter? (Transformer)", loc="left", color=INK, fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def binned(x, y, bins=15):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    d["b"] = pd.qcut(d.x.rank(method="first"), bins, labels=False)
    g = d.groupby("b").agg(x=("x", "median"), y=("y", "mean"), lo=("y", lambda v: v.quantile(0.25)),
                           hi=("y", lambda v: v.quantile(0.75)))
    return g


def fig_dependence(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.0))
    for ax, (title, xlabel, series) in zip(np.atleast_1d(axes), panels):
        for lab, x, y, col in series:
            g = binned(x, y)
            ax.fill_between(g.x, g.lo, g.hi, color=col, alpha=0.15, lw=0)
            ax.plot(g.x, g.y, "o-", color=col, ms=4, lw=2, label=lab)
        ax.axhline(0, color=INK2, lw=1, ls="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SHAP contribution (log-hazard)")
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ================================================================== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=32, help="expected-gradients samples per row")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-rows", type=int, default=0, help="explain only a random subset (quick check)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = _time.time()
    from src.models.neural import load
    data = load()
    lm = pd.read_csv(LANDMARKS, parse_dates=["GAME_DATE"])
    rows = np.flatnonzero(lm.split.isin(list(SETS)).to_numpy())
    if args.max_rows and args.max_rows < len(rows):
        rows = np.sort(np.random.default_rng(0).choice(rows, args.max_rows, replace=False))
    fit_rows = np.flatnonzero(lm.SEASON.isin(FIT_SEASONS).to_numpy())
    bg_rows = np.sort(np.random.default_rng(0).choice(fit_rows, N_BG, replace=False))
    score = lm.iloc[rows].reset_index(drop=True)

    # ---- Cox --------------------------------------------------------------------
    with open(MODEL_DIR / "cox.pkl", "rb") as f:
        cox = pickle.load(f)
    c_phi, c_base, c_eta, c_names = cox_shap(cox, lm.iloc[fit_rows], score)
    assert np.allclose(c_phi.sum(1) + c_base, c_eta, atol=1e-8), "Cox SHAP does not add up"

    # ---- Transformer -------------------------------------------------------------
    print(f"explaining {len(rows):,} predictions; {args.samples} samples per row ...")
    tf, models, n_seq = transformer_shap(data, rows, bg_rows, args.device, args.samples)
    static_names = list(data["static_names"][models[0][1]])
    seq_names = list(data["game_feature_names"]) + ["days_before"]
    phi_seq = np.concatenate([tf["seq"], tf["dt"][..., None]], axis=2)        # [n, L, F+1]
    phi_sum = tf["static"].sum(1) + phi_seq.sum((1, 2))
    chk = additivity_check(data, models, n_seq, rows, bg_rows, phi_sum, tf["eta"], args.device)
    print(f"  additivity: mean |gap| {chk['mean_abs_gap']:.4f} (output sd {chk['sd_of_output']:.3f}), "
          f"corr {chk['corr']:.3f}")

    np.savez_compressed(OUT / "shap_values.npz", player_id=score.PLAYER_ID.to_numpy(), game_id=score.GAME_ID.to_numpy(),
                        tf_static=tf["static"].astype(np.float32), tf_seq=phi_seq.astype(np.float32),
                        tf_eta=tf["eta"], tf_static_names=np.array(static_names), tf_seq_names=np.array(seq_names),
                        n_seq=n_seq, cox_phi=c_phi.astype(np.float32), cox_base=c_base, cox_eta=c_eta,
                        cox_names=np.array(c_names))

    # ---- summaries ---------------------------------------------------------------
    feat, fam_phi = tf_tables(tf["static"], phi_seq, static_names, seq_names, n_seq)
    tf_fam = family_table(fam_phi)
    cox_fam = family_table(cox_family_phi(c_phi, c_names))
    cox_feat = pd.DataFrame({"input": [static_label(n) for n in c_names],
                             "mean_abs": np.abs(c_phi).mean(0)}).sort_values("mean_abs", ascending=False)
    pd.concat([tf_fam.assign(model="transformer"), cox_fam.assign(model="cox")]) \
        .to_csv(OUT / "shap_families.csv", index=False)
    feat.to_csv(OUT / "shap_top_inputs.csv", index=False)
    pd.Series(chk).to_json(OUT / "shap_check.json")
    fig_top(feat, OUT / "fig_shap_top.png")
    fig_families(tf_fam, cox_fam, OUT / "fig_shap_families.png")
    fig_positions(phi_seq, OUT / "fig_shap_positions.png")

    sidx = {n: j for j, n in enumerate(static_names)}
    cidx = {n: j for j, n in enumerate(c_names)}
    mins_tf = sum(tf["static"][:, sidx[n]] for n in MINUTE_FEATURES if n in sidx) + phi_seq[:, :, seq_names.index("MIN")].sum(1)
    mins_cox = sum(c_phi[:, cidx[n]] for n in MINUTE_FEATURES if n in cidx)
    hist_tf = fam_phi["history"]
    hist_cox = cox_family_phi(c_phi, c_names)["history"]
    fig_dependence([
        ("Minutes: summed contribution of all minutes inputs", "Recent minutes per game (~7 games)",
         [("Transformer", score.ewma_min_short.to_numpy(), mins_tf, "#1f5fae"),
          ("Cox", score.ewma_min_short.to_numpy(), mins_cox, "#8a8984")]),
        ("Injury history: summed contribution", "Prior injuries (any) in the data",
         [("Transformer", score.prior_any_injuries.to_numpy(), hist_tf, "#1f5fae"),
          ("Cox", score.prior_any_injuries.to_numpy(), hist_cox, "#8a8984")])],
        OUT / "fig_shap_dependence.png")

    def fmt(df, cols):
        d = df[cols].copy()
        if "share" in d:
            d["share"] = (d.share * 100).map(lambda v: f"{v:.0f}%")
        if "family" in d:
            d["family"] = d.family.map(lambda f: FAMILY_LABELS.get(f, f))
        return d

    md = ["# SHAP explanations of the final models\n",
          f"Explained: {len(rows):,} predictions in the held-out seasons (2023-24 test, 2024-25 supplementary). "
          "Transformer + DeepSurv (5 games): expected-gradients SHAP, averaged over the 3 seeds, with "
          f"{args.samples} samples per prediction and {N_BG} background predictions from the fit seasons. "
          "Cox: exact SHAP. Values are in log-hazard units (exp(SHAP) = hazard multiplier).\n",
          f"**Additivity check** ({chk['rows_checked']} predictions): the contributions add up to the model output "
          f"minus its background average with a mean absolute gap of {chk['mean_abs_gap']:.3f} log-hazard "
          f"(the output's SD is {chk['sd_of_output']:.3f}; correlation {chk['corr']:.3f}).\n",
          "## Feature families (share of total attribution)\n",
          _md(fmt(tf_fam, ["family", "mean_abs", "share"]).rename(columns={"mean_abs": "transformer_mean_abs",
                                                                          "share": "transformer_share"})
              .merge(fmt(cox_fam, ["family", "mean_abs", "share"]).rename(columns={"mean_abs": "cox_mean_abs",
                                                                                   "share": "cox_share"}),
                     on="family", how="outer").sort_values("transformer_mean_abs", ascending=False)), "",
          "![Families](fig_shap_families.png)\n",
          "## Most influential inputs (Transformer)\n", _md(fmt(feat.head(15), ["input", "family", "mean_abs"])), "",
          "![Top inputs](fig_shap_top.png)\n",
          "## Most influential inputs (Cox)\n", _md(cox_feat.head(10).reset_index(drop=True)), "",
          "## Previous games and dependence\n", "![Positions](fig_shap_positions.png)\n",
          "![Dependence](fig_shap_dependence.png)\n",
          "## Notes\n",
          "- SHAP explains what the model uses, not what causes injuries.",
          "- Correlated inputs share credit (e.g., minutes appear in several summary features and in the game "
          "sequence), so family totals are more reliable than single features.",
          "- Expected gradients are a sampling estimate; the additivity check above measures its error."]
    (OUT / "shap_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nsaved {OUT}  ({(_time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
