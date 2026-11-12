"""NBA non-contact injury risk dashboard (ECE 5904).

Shows the final Transformer + DeepSurv (5 games) model's 14-day injury risk for every
player-game in the held-out seasons, with the SHAP drivers of each prediction.

    python -m src.dashboard.prepare      # once, after the final evaluation and SHAP
    streamlit run app.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "dashboard"
FINAL = ROOT / "data" / "processed" / "final"
EXPLAIN = ROOT / "data" / "processed" / "explain"
RED, BLUE, GREY, INK2 = "#d9534f", "#3b82d6", "#8a8984", "#52514e"
TXT = "#9aa0a6"          # readable on both Streamlit's light and dark themes (figures are transparent)
SEASON_LABEL = {"2023-24": "2023-24 (test season)", "2024-25": "2024-25 (through Jan 12, 2025)"}

st.set_page_config(page_title="NBA Injury Risk", page_icon="🏀", layout="wide")


@st.cache_data
def load():
    games = pd.read_csv(DATA / "games.csv.gz", parse_dates=["GAME_DATE", "injury_date"])
    drv = pd.read_csv(DATA / "drivers.csv.gz", keep_default_na=False, na_values=[])
    drv["shap"] = drv.shap.astype(float)
    log = pd.read_csv(DATA / "gamelog.csv.gz", parse_dates=["GAME_DATE"])
    inj = pd.read_csv(DATA / "injuries.csv", parse_dates=["injury_date"])
    return games, drv, log, inj


def ordinal(n):
    n = int(round(n))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def pct(v):
    return f"{100 * v:.1f}%"


def _style(ax):
    fig = ax.figure
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TXT)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.grid(color=TXT, alpha=0.18, lw=0.8)
    ax.set_axisbelow(True)


def risk_chart(pg, inj, sel_row, avg):
    """Static figure: risk across the season, season average, injuries, selected game."""
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=150)
    x, y = pg.GAME_DATE.to_numpy(), 100 * pg.risk.to_numpy()
    ax.axhline(100 * avg, color=TXT, ls="--", lw=1, label="Season average")
    ax.plot(x, y, color=BLUE, lw=1.8, marker="o", ms=2.8, label="14-day risk before each game")
    for k, (_, r) in enumerate(inj.iterrows()):
        ax.axvline(r.injury_date, color=RED, lw=1.6, label="Lower-leg non-contact injury" if k == 0 else None)
    sel = pg.row.to_numpy() == sel_row
    if sel.any():
        ax.plot(x[sel], y[sel], "o", ms=10, mfc="none", mec=TXT, mew=1.6, label="Selected game")
    ax.set_ylim(0, max(y.max(), 100 * avg) * 1.15)
    ax.set_ylabel("14-day risk (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _style(ax)
    leg = ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    for t in leg.get_texts():
        t.set_color(TXT)
    fig.tight_layout()
    return fig


def driver_chart(d):
    """Static figure: the SHAP drivers of one prediction, largest first."""
    d = d.assign(name=np.where(d.value != "", d.label.str.replace(r" \((in|lb|km|h)\)$", "", regex=True)
                               + " (" + d.value + ")", d.label))
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(d) + 1.0), dpi=150)
    yy = np.arange(len(d))[::-1]
    cols = [RED if v > 0 else BLUE for v in d.shap]
    ax.barh(yy, d.shap, color=cols, height=0.6)
    lo, hi = min(d.shap.min(), 0.0), max(d.shap.max(), 0.0)
    pad = 0.2 * (hi - lo or 1.0)
    ax.set_xlim(lo - (pad if lo < 0 else 0), hi + pad)
    for yi, v in zip(yy, d.shap):
        ax.text(v + (0.01 if v >= 0 else -0.01) * (hi - lo or 1), yi, f"×{np.exp(v):.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=TXT)
    ax.axvline(0, color=TXT, lw=0.8)
    ax.set_yticks(yy, d.name)
    mult = [m for m in (0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4) if lo - pad <= np.log(m) <= hi + pad]
    ax.set_xticks(np.log(mult), [f"×{m:g}" for m in mult])
    ax.set_xlabel("Effect on injury risk (right of ×1 raises it, left lowers it)")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9, length=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- pages -----
def player_page(games, drv, log, inj):
    c1, c2, c3 = st.columns([1, 1, 2])
    season = c1.selectbox("Season", sorted(games.SEASON.unique()), format_func=lambda s: SEASON_LABEL.get(s, s))
    g = games[games.SEASON == season]
    team = c2.selectbox("Team", ["All teams"] + sorted(g.TEAM.dropna().unique()))
    pool = g if team == "All teams" else g[g.TEAM == team]
    names = pool.groupby("PLAYER_ID").PLAYER_NAME.first().sort_values()
    default = int(np.argmax(names.index == pool.groupby("PLAYER_ID").risk.mean().idxmax()))
    pid = c3.selectbox("Player", names.index, index=default, format_func=lambda p: names[p])
    pg = g[g.PLAYER_ID == pid].sort_values("GAME_DATE")
    pinj = inj[(inj.PLAYER_ID == pid) & inj.injury_date.between(pg.GAME_DATE.min(), pg.GAME_DATE.max()
                                                                  + pd.Timedelta(days=30))]
    labels = (pg.GAME_DATE.dt.strftime("%b %d, %Y") + " · " + pg.MATCHUP.fillna("")).tolist()
    pick = st.select_slider("Prediction before game", options=list(range(len(pg))), value=len(pg) - 1,
                            format_func=lambda i: labels[i])
    r = pg.iloc[pick]
    avg = g.risk.mean()

    fig = risk_chart(pg, pinj, r.row, avg)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Each point is the Transformer + DeepSurv model's risk of a lower-leg non-contact injury in the "
               "14 days after that game, using only information available before it.")

    st.markdown(f"**Transformer + DeepSurv (5 games) — prediction before {r.GAME_DATE:%b %d, %Y}**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("14-day injury risk", pct(r.risk), help="Main model: Transformer over the last 5 games + DeepSurv")
    m2.metric("vs. season average", f"{r.x_avg:.1f}×", help="This risk divided by the season's average risk")
    m3.metric("Percentile (season)", ordinal(r.percentile), help="Share of all player-games this season with lower risk")
    m4.metric("Cox model, for comparison", pct(r.cox_risk),
              help="Baseline Cox model on the summary features. On the test season both models ranked players "
                   "equally well (C-index 0.752 vs 0.753).")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Why this risk?")
        st.caption("SHAP explanation of the Transformer + DeepSurv prediction")
        d = drv[drv.row == r.row].sort_values("rank")
        fig = driver_chart(d)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("The inputs that moved this prediction most, relative to an average player-game. "
                   "×1.30 means the input multiplies the injury hazard by 1.3. "
                   "Associations learned from data, not causes.")
    with right:
        st.subheader("Previous games")
        prev = log[(log.PLAYER_ID == pid) & (log.GAME_DATE < r.GAME_DATE)].sort_values("GAME_DATE").tail(5)
        show = prev.assign(Date=prev.GAME_DATE.dt.strftime("%b %d"), Game=prev.MATCHUP,
                           Minutes=prev.MIN.round(0).astype("Int64"), Points=prev.PTS.astype("Int64"),
                           B2B=np.where(prev.back_to_back == 1, "yes", ""),
                           Travel_km=prev.travel_km.round(-1).astype("Int64"))
        st.dataframe(show[["Date", "Game", "Minutes", "Points", "B2B", "Travel_km"]].iloc[::-1],
                     hide_index=True, use_container_width=True)
        st.subheader("What happened next")
        if r.event == 1:
            note = pinj.loc[pinj.injury_date == r.injury_date, "notes"]
            txt = note.iloc[0] if len(note) and pd.notna(note.iloc[0]) else "target injury"
            n = int(r.time)
            when = ("in this game (injuries are dated to the last game before the injured-list move)" if n == 0
                    else f"{n} day{'s' if n != 1 else ''} later ({r.injury_date:%b %d})")
            st.error(f"Injured {when}: {txt}")
        else:
            n = int(r.time)
            if n == 0:
                st.info("Follow-up ended right after this game (another injury, the end of the season, "
                        "or the end of the injury data), so there is no outcome to show.")
            else:
                why = "" if n >= 30 else " (follow-up ended early: season over, another injury, or end of the data)"
                st.success(f"No target injury in the next {n} day{'s' if n != 1 else ''}{why}")


def team_page(games, drv):
    c1, c2, c3 = st.columns(3)
    season = c1.selectbox("Season ", sorted(games.SEASON.unique()), format_func=lambda s: SEASON_LABEL.get(s, s))
    g = games[games.SEASON == season]
    team = c2.selectbox("Team ", sorted(g.TEAM.dropna().unique()))
    tg = g[g.TEAM == team]
    dates = sorted(tg.GAME_DATE.dt.date.unique())
    day = c3.selectbox("Game date", dates[::-1], format_func=lambda d: f"{d:%b %d, %Y}")
    roster = tg[tg.GAME_DATE.dt.date == day].sort_values("risk", ascending=False)
    if roster.empty:
        st.info("No games.")
        return
    st.markdown(f"**{roster.MATCHUP.dropna().iloc[0] if roster.MATCHUP.notna().any() else team}** — "
                f"players who played, ranked by the Transformer + DeepSurv 14-day injury risk before the game")
    up = drv[drv.row.isin(roster.row) & (drv.shap > 0)].sort_values("shap", ascending=False) \
        .drop_duplicates("row").set_index("row")
    unitless = up.label.str.replace(r" \((in|lb|km|h)\)$", "", regex=True)
    main = [f"{unitless[i] + ' (' + up.value[i] + ')' if up.value[i] else up.label[i]}  ×{np.exp(up.shap[i]):.2f}"
            if i in up.index else "—" for i in roster.row]
    out = pd.DataFrame({
        "Player": roster.PLAYER_NAME.to_numpy(),
        "14-day risk": (100 * roster.risk).round(1).to_numpy(),
        "× season avg": roster.x_avg.to_numpy(),
        "Percentile": roster.percentile.round(0).astype(int).to_numpy(),
        "Top factor raising risk": main,
        "Injured within 14 days": np.where(roster.injured_14d == 1, "yes", "").tolist()})
    st.dataframe(out, hide_index=True, use_container_width=True,
                 column_config={"14-day risk": st.column_config.ProgressColumn(
                     "14-day risk (%)", min_value=0.0, max_value=float(max(out["14-day risk"].max(), 1.0)),
                     format="%.1f"),
                     "× season avg": st.column_config.NumberColumn("× season avg", format="%.1f×")})
    st.caption("Top factor raising risk: the input with the largest SHAP contribution pushing this player's risk "
               "up, with the factor by which it multiplies the hazard.")


def _img(col, path, caption):
    if path.exists():
        col.image(str(path), caption=caption, use_container_width=True)


def _results_table(t, s_):
    sub = t[t.set == s_]
    return pd.DataFrame({
        "Model": sub.model, "C-index": sub.c_index.map("{:.3f}".format), "95% CI": sub.c_95ci,
        "vs Cox": sub.c_minus_cox.map("{:+.3f}".format), "Paired 95% CI": sub.paired_95ci,
        "AUC (14 d)": sub.auc_14d.map("{:.3f}".format), "Brier (14 d)": sub.brier_14d.map("{:.4f}".format),
        "Predicted 14-d risk": sub.mean_pred_14d.map(lambda v: f"{100 * v:.1f}%"),
        "Observed": sub.observed_14d.map(lambda v: f"{100 * v:.1f}%")})


MAIN_MODEL = "Transformer + DeepSurv (5 games)"


def ranking_chart(test):
    """C-index of each model on the test season, main model highlighted."""
    d = test.sort_values("c_index")
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(d) + 1.0), dpi=150)
    y = np.arange(len(d))
    for yi, (m, r) in zip(y, d.iterrows()):
        lo_, hi_ = (float(v) for v in str(r.c_95ci).replace("–", " ").split())
        main = m == MAIN_MODEL
        col = BLUE if main else ("#b0b4ba" if m != "Cox" else GREY)
        ax.plot([lo_, hi_], [yi, yi], color=col, lw=3 if main else 2, solid_capstyle="round", alpha=0.9)
        ax.plot(r.c_index, yi, "o", color=col, ms=9 if main else 7)
        ax.text(hi_ + 0.004, yi, f"{r.c_index:.3f}", va="center", fontsize=9, color=TXT,
                fontweight="bold" if main else "normal")
    ax.set_yticks(y, list(d.index))
    for lab in ax.get_yticklabels():
        if lab.get_text() == MAIN_MODEL:
            lab.set_fontweight("bold")
    ax.set_xlim(0.6, 0.85)
    ax.set_xlabel("C-index, test season 2023-24 (95% CI)")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9, length=0)
    fig.tight_layout()
    return fig


def risk_groups(games, season, k=5):
    """Players split into k equal groups by predicted risk: mean predicted vs observed 14-day injury rate."""
    g = games[games.SEASON == season]
    known = (g.injured_14d == 1) | (g.time >= 14)          # 14-day outcome observed
    g = g[known].copy()
    g["group"] = pd.qcut(g.risk.rank(method="first"), k, labels=False)
    out = g.groupby("group").agg(predicted=("risk", "mean"), observed=("injured_14d", "mean"), n=("risk", "size"))
    return out.reset_index()


def groups_chart(rg):
    k = len(rg)
    names = [f"Q{i + 1}" for i in range(k)]
    names[0], names[-1] = names[0] + "\n(lowest)", names[-1] + "\n(highest)"
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    x = np.arange(k)
    ax.bar(x - 0.19, 100 * rg.predicted, 0.36, color=BLUE, label="Predicted")
    ax.bar(x + 0.19, 100 * rg.observed, 0.36, color="#b0b4ba", label="Observed")
    for xi, v in zip(x, rg.observed):
        ax.text(xi + 0.19, 100 * v + 0.2, f"{100 * v:.1f}%", ha="center", fontsize=8, color=TXT)
    ax.set_xticks(x, names)
    ax.set_ylabel("14-day injury rate (%)")
    ax.set_xlabel("Predicted-risk quintile (test season 2023-24)")
    _style(ax)
    ax.grid(axis="x", visible=False)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for tx in leg.get_texts():
        tx.set_color(TXT)
    fig.tight_layout()
    return fig


FAMILY_PLAIN = {"history": "Injury history", "workload_seq": "Minutes and usage, last 5 games",
                "workload": "Workload aggregates", "bio": "Player characteristics",
                "context": "Playoffs and season era", "travel": "Travel", "schedule": "Schedule and rest"}


def families_chart(fam):
    tf = fam[fam.model == "transformer"].sort_values("share")
    cx = fam[fam.model == "cox"].set_index("family").share
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(tf) + 1.0), dpi=150)
    y = np.arange(len(tf))
    ax.barh(y + 0.18, 100 * tf.share, 0.34, color=BLUE, label="Transformer + DeepSurv")
    ax.barh(y - 0.18, [100 * cx.get(f, 0) for f in tf.family], 0.34, color="#b0b4ba", label="Cox")
    for yi, v in zip(y, tf.share):
        ax.text(100 * v + 0.6, yi + 0.18, f"{100 * v:.0f}%", va="center", fontsize=8.5, color=TXT)
    ax.set_yticks(y, [FAMILY_PLAIN.get(f, f) for f in tf.family])
    ax.set_xlabel("Share of total attribution (%)")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9, length=0)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for tx in leg.get_texts():
        tx.set_color(TXT)
    fig.tight_layout()
    return fig


PLAIN_INPUT = {"Days since last injury (missing)": "No earlier injury on record",
               "Years since draft (missing)": "Draft year unknown",
               "Career minutes (in data)": "Career minutes (since 2016)",
               "Career games (in data)": "Career games (since 2016)",
               "Height (in)": "Height", "Weight (lb)": "Weight"}


def top_inputs_chart(top, n=10):
    d = top.head(n).iloc[::-1].assign(input=lambda x: x.input.replace(PLAIN_INPUT))
    total = top.mean_abs.sum()
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(d) + 1.0), dpi=150)
    cols = [RED if f == "history" else BLUE if f in ("workload", "workload_seq") else "#b0b4ba" for f in d.family]
    ax.barh(np.arange(len(d)), 100 * d.mean_abs / total, color=cols, height=0.65)
    ax.set_yticks(np.arange(len(d)), d.input)
    ax.set_xlabel("Share of total attribution (%)")
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9, length=0)
    fig.tight_layout()
    return fig


def _show(fig, where=st):
    where.pyplot(fig, use_container_width=True)
    plt.close(fig)


def model_page(games):
    f = FINAL / "test_results.csv"
    if not f.exists():
        st.info("Final results not found. Run: python -m src.final.evaluate")
        return
    t = pd.read_csv(f).fillna("")
    test = t[t.set == "test"].set_index("model")
    tf, cox = test.loc[MAIN_MODEL], test.loc["Cox"]
    lo, hi = (float(v) for v in str(tf.paired_95ci).replace(" to ", " ").split())
    verdict = ("higher discrimination than" if lo > 0 else "lower discrimination than" if hi < 0
               else "discrimination comparable to")

    r1, r2, r3 = st.tabs(["Performance", "Feature attribution", "Technical details"])
    with r1:
        st.markdown("### Held-out test performance (2023-24)")
        st.markdown(f"The Transformer + DeepSurv model showed **{verdict}** the Cox baseline on the held-out "
                    f"season (C-index {tf.c_index:.3f} vs. {cox.c_index:.3f}; difference {tf.c_minus_cox:+.3f}, "
                    f"paired 95% CI {tf.paired_95ci}).")
        st.caption("Models were trained on 2016-17 through 2022-23 and evaluated once on 2023-24. The comparison "
                   "with the Cox baseline was specified before the test season was evaluated.")
        k1, k2, k3 = st.columns(3)
        k1.metric("Transformer + DeepSurv (5 games)", f"{tf.c_index:.3f}", help=f"C-index, 95% CI {tf.c_95ci}")
        k2.metric("Cox baseline", f"{cox.c_index:.3f}", help=f"C-index, 95% CI {cox.c_95ci}")
        k3.metric("Difference", f"{tf.c_minus_cox:+.3f}", help=f"Paired 95% CI {tf.paired_95ci}")

        st.markdown("#### Discrimination by model")
        a, b = st.columns([3, 2], gap="large")
        _show(ranking_chart(test), a)
        b.markdown("The **C-index** is the probability that, for two player-games, the model assigns the higher "
                   "risk to the one followed sooner by an injury (0.5 = chance, 1.0 = perfect ranking).\n\n"
                   "All models fall near 0.75. Their 95% confidence intervals overlap substantially, so the "
                   "differences between models are not statistically reliable.")

        st.markdown("#### Observed injury rates by predicted-risk quintile")
        rg = risk_groups(games, "2023-24")
        a, b = st.columns([3, 2], gap="large")
        _show(groups_chart(rg), a)
        ratio = rg.observed.iloc[-1] / max(rg.observed.iloc[0], 1e-9)
        b.markdown(f"Player-games in the highest predicted-risk quintile had a 14-day injury rate of "
                   f"**{100 * rg.observed.iloc[-1]:.1f}%**, compared with **{100 * rg.observed.iloc[0]:.1f}%** "
                   f"in the lowest quintile (**{ratio:.0f}×** higher).\n\n"
                   + (f"Predicted and observed rates follow the same pattern across quintiles. "
                      if np.all(np.diff(rg.observed) > 0) else "")
                   + f"On average the model predicts {100 * tf.mean_pred_14d:.1f}% against an observed "
                   f"{100 * tf.observed_14d:.1f}%"
                   + (", a modest overestimate." if tf.mean_pred_14d > tf.observed_14d * 1.05 else
                      ", a modest underestimate." if tf.mean_pred_14d < tf.observed_14d * 0.95 else "."))

    with r2:
        st.markdown("### Feature attribution (SHAP)")
        st.markdown("SHAP decomposes each prediction into the contribution of every input. The charts below "
                    "summarize those contributions across all held-out predictions.")
        fam_f, top_f = EXPLAIN / "shap_families.csv", EXPLAIN / "shap_top_inputs.csv"
        if fam_f.exists():
            fam = pd.read_csv(fam_f)
            st.markdown("#### Contribution by feature group")
            a, b = st.columns([3, 2], gap="large")
            _show(families_chart(fam), a)
            sh = fam[fam.model == "transformer"].set_index("family").share
            b.markdown(f"- **Injury history** accounts for **{100 * sh.get('history', 0):.0f}%** of the "
                       f"Transformer's attribution, similar to the Cox model.\n"
                       f"- **Minutes and usage over the last 5 games**, the sequence input unique to the "
                       f"Transformer, account for **{100 * sh.get('workload_seq', 0):.0f}%**.\n"
                       f"- **Travel** and **schedule** contribute little "
                       f"({100 * sh.get('travel', 0):.0f}% and {100 * sh.get('schedule', 0):.0f}%).")
        if top_f.exists():
            st.markdown("#### Most influential inputs")
            a, b = st.columns([3, 2], gap="large")
            _show(top_inputs_chart(pd.read_csv(top_f)), a)
            b.markdown("Colors indicate the feature group: red for injury history, blue for workload, grey for "
                       "other inputs. Inputs labeled *Last 5 games* are derived from the game-by-game "
                       "sequence.\n\n"
                       "*Attributions describe what the model relies on; they do not establish causes of injury.*")

    with r3:
        st.caption("Full results tables and the figures from the project report.")
        for s_, title in (("test", "Test season 2023-24"), ("supplementary", "Supplementary 2024-25 (to Jan 12, 2025)")):
            if (t.set == s_).any():
                st.markdown(f"**{title}**")
                st.dataframe(_results_table(t, s_), hide_index=True, use_container_width=True)
        figs = [(FINAL / "fig_test_cindex.png", "C-index, both held-out seasons"),
                (FINAL / "fig_test_calibration.png", "Calibration (deciles)"),
                (FINAL / "fig_test_km.png", "Cumulative incidence by risk tertile"),
                (EXPLAIN / "fig_shap_positions.png", "SHAP by previous game"),
                (EXPLAIN / "fig_shap_dependence.png", "SHAP dependence: minutes and injury history"),
                (EXPLAIN / "fig_shap_top.png", "Top 20 inputs (mean |SHAP|)")]
        figs = [(fp, c) for fp, c in figs if fp.exists()]
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            for col, (img, cap) in zip(cols, figs[i:i + 2]):
                _img(col, img, cap)


def main():
    st.title("NBA non-contact injury risk")
    st.caption("Transformer + DeepSurv survival model (last 5 games + 42 workload, schedule, travel, injury-history "
               "and player features), evaluated on seasons it never saw. ECE 5904 research prototype — not "
               "medical advice; injury labels come from public injured-list transactions.")
    if not (DATA / "games.csv.gz").exists():
        st.error("Dashboard data not found. Run:  python -m src.dashboard.prepare")
        st.stop()
    games, drv, log, inj = load()
    t1, t2, t3 = st.tabs(["Player", "Team game night", "Model"])
    with t1:
        player_page(games, drv, log, inj)
    with t2:
        team_page(games, drv)
    with t3:
        model_page(games)


main()
