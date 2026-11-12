"""答え合わせと集計（Claude実装）。"""
import numpy as np
import pandas as pd


def settle_predictions(pred: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """1レース1行の予想に、1着艇・LogLoss・ブライアスコアを付ける。
    - 結果のないレース（中止など）と、フライング・出遅れ・欠場（返還）があったレースは、確率の評価から除く
      （予想は6艇が走る前提のため。Geminiレビューで修正）
    - 1着同着は1レース1行のまま、同着艇すべてを1着として扱う"""
    refund_races = set(results.loc[results["st_status"].isin(["F", "L", "K"]), "race_id"])
    winners = (
        results[results["finish_order"] == 1]
        .groupby("race_id")["boat_no"].apply(lambda s: sorted(int(x) for x in s)).rename("winners")
    )
    df = pred.merge(winners, left_on="race_id", right_index=True, how="inner")
    df = df[~df["race_id"].isin(refund_races)].reset_index(drop=True)
    P = df[[f"p{i}" for i in range(1, 7)]].to_numpy()
    Y = np.zeros_like(P)
    for i, ws in enumerate(df["winners"]):
        for w in ws:
            Y[i, w - 1] = 1
    df["winner"] = df["winners"].apply(lambda ws: ws[0])
    df["dead_heat"] = df["winners"].apply(len) > 1
    df["winners"] = df["winners"].apply(lambda ws: ",".join(map(str, ws)))
    pw = (P * Y).sum(axis=1)
    df["p_winner"] = pw
    df["logloss"] = -np.log(np.clip(pw, 1e-9, 1))
    df["brier"] = ((P - Y) ** 2).sum(axis=1)
    return df


def settle_recommendations(recs: pd.DataFrame, payouts: pd.DataFrame) -> pd.DataFrame:
    """推奨買い目に的中・払戻・収支を付ける。返還艇を含む目は元返し。賭け金は stake 列（円）"""
    if recs.empty:
        return recs
    pay = payouts[["race_id", "trifecta_comb", "trifecta_payout", "refund_boats", "is_aborted"]]
    df = recs.merge(pay, on="race_id", how="inner")

    def one(r):
        refund = set(str(r["refund_boats"]).split(",")) - {""}
        if r["is_aborted"] or (set(r["combo"].split("-")) & refund):
            return "refund", float(r["stake"])
        if r["combo"] in str(r["trifecta_comb"]).split(","):
            return "hit", float(r["trifecta_payout"]) * float(r["stake"]) / 100
        return "miss", 0.0

    res = df.apply(one, axis=1, result_type="expand")
    df["outcome"], df["payout"] = res[0], res[1]
    df["profit"] = df["payout"] - df["stake"]
    return df


def calibration_table(settled: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """全艇の(予想確率, 実際に1着か)を、件数が同じになるよう bins 区間に分けて比べる"""
    P = settled[[f"p{i}" for i in range(1, 7)]].to_numpy().ravel()
    Y = np.zeros((len(settled), 6))
    for i, ws in enumerate(settled["winners"].astype(str)):
        for w in ws.split(","):
            Y[i, int(w) - 1] = 1
    Y = Y.ravel()
    mask = P > 0
    df = pd.DataFrame({"p": P[mask], "y": Y[mask]})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    t = df.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return t


def ece(cal: pd.DataFrame) -> float:
    n = cal["n"].sum()
    return float((cal["n"] * (cal["pred"] - cal["actual"]).abs()).sum() / n) if n else float("nan")
