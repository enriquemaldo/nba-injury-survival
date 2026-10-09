"""Step 6: feasibility report for Update 1.

Usage:  python -m src.feasibility
Writes data/processed/feasibility_report.md (+ CSV tables) summarising:
  - events per season under each label tier (all episodes, and time-loss
    episodes = missed >= 1 team game)
  - events in the TRAINING seasons (2016-17..2021-22, Option B split)
  - site breakdown for the proxy tier
  - name-match rate and uncertain counts
"""
import pandas as pd

from src.config import (INTERIM, PROCESSED, RAW, SEASON_WINDOWS, SEASONS,
                        TRAIN_SEASONS, split_of)

TRAIN = TRAIN_SEASONS


def _md(df: pd.DataFrame) -> str:
    """Markdown table without the optional 'tabulate' dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def coverage(season: str, start: str, end: str) -> str:
    w0, w1 = (pd.Timestamp(x) for x in SEASON_WINDOWS[season])
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    if b < w0 or a > w1:
        return "none"
    if a <= w0 + pd.Timedelta(days=45) and b >= w1 - pd.Timedelta(days=20):
        return "full"
    return f"partial (to {min(b, w1):%Y-%m-%d})" if b < w1 else "partial"


def main():
    d, tag = INTERIM, ""
    ep = pd.read_csv(d / "injury_episodes.csv", parse_dates=["onset_date"])
    rows = pd.read_csv(d / "injury_rows_labeled.csv")
    gl = pd.read_csv(RAW / "gamelogs_all.csv", usecols=["SEASON", "PLAYER_ID"])

    ep["time_loss"] = ep.games_missed.fillna(0) >= 1
    src_info = pd.read_json(d / "label_source.json", typ="series")
    per = []
    for s in SEASONS:
        e = ep[ep.season == s]
        g = gl[gl.SEASON == s]
        per.append({
            "season": s,
            "split": split_of(s),
            "label_coverage": coverage(s, src_info.start, src_info.end),
            "players": g.PLAYER_ID.nunique(),
            "player_games": len(g),
            "strict": int(e.is_strict.sum()),
            "proxy": int(e.is_proxy.sum()),
            "broad": int(e.is_broad.sum()),
            "strict_tl": int((e.is_strict & e.time_loss).sum()),
            "proxy_tl": int((e.is_proxy & e.time_loss).sum()),
            "broad_tl": int((e.is_broad & e.time_loss).sum()),
            "uncertain": int((e.tier == "uncertain").sum()),
            # uncertain episodes whose only note is a reason-less IL placement
            "bare_il": int(((e.tier == "uncertain")
                            & e.notes.fillna("").str.fullmatch(r"(?i)placed on IL(\s*\(P\))?")).sum()),
            "other_injury": int((e.tier == "other_injury").sum()),
        })
    per = pd.DataFrame(per)
    total = per.drop(columns=["season", "split", "label_coverage"]).sum().to_frame().T
    total.insert(0, "label_coverage", ""); total.insert(0, "split", ""); total.insert(0, "season", "TOTAL")
    train = per[per.split == "train"].drop(columns=["season", "split", "label_coverage"]).sum().to_frame().T
    train.insert(0, "label_coverage", ""); train.insert(0, "split", ""); train.insert(0, "season", f"TRAIN ({len(TRAIN)} seasons)")
    # Player counts don't add across seasons; use unique players instead.
    train["players"] = gl[gl.SEASON.isin(TRAIN)].PLAYER_ID.nunique()
    total["players"] = gl.PLAYER_ID.nunique()
    per_out = pd.concat([per, train, total], ignore_index=True)
    per_out.to_csv(PROCESSED / f"feasibility_by_season{tag}.csv", index=False)

    sites = (ep[ep.is_proxy].groupby(["site"]).size().rename("proxy_episodes")
             .sort_values(ascending=False).reset_index())
    inj = rows[(rows.tier != "non_injury") & (rows.match_method != "not_player")]
    match = inj.match_method.fillna("unmatched").value_counts().rename("rows").reset_index()

    md = ["# Feasibility report (Update 1)\n",
          f"Label source: Kaggle 2016-2025 NBA injury data (Pro Sports Transactions injured-list "
          f"transactions), `{src_info.file}`, {src_info.start} to {src_info.end}.\n",
          "Tiers are nested (strict ⊂ proxy ⊂ broad). `_tl` = time-loss episodes "
          "(player missed ≥1 team game). `bare_il` = uncertain episodes whose only "
          "note is a reason-less 'placed on IL' (likely roster management; excluded).\n",
          "## Episodes per season\n", _md(per_out), "",
          "## Proxy-tier episodes by site\n", _md(sites), "",
          f"## Player-name matching\n\nMatch rate on injury rows: "
          f"**{inj.PLAYER_ID.notna().mean():.1%}**\n", _md(match), ""]
    md += ["## Event summary\n",
           f"- Training-season events, proxy tier (time-loss): **{int(train.proxy_tl.iloc[0])}**",
           f"- Training-season events, strict tier (time-loss): **{int(train.strict_tl.iloc[0])}**",
           f"- Training-season events, broad tier (time-loss): **{int(train.broad_tl.iloc[0])}**",
           f"- Validation (2022-23) proxy time-loss events: **{int(per[per.split == 'val'].proxy_tl.sum())}**",
           f"- Test (2023-24) proxy time-loss events: **{int(per[per.split == 'test'].proxy_tl.sum())}**",
           f"- Average proxy time-loss events per training season: "
           f"**{int(train.proxy_tl.iloc[0]) / len(TRAIN):.0f}**"]
    out = PROCESSED / f"feasibility_report{tag}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
