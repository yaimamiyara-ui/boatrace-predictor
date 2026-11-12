"""過去データでの「モデル対市場」の比較（Claude実装）。

過去の全オッズは残っていないが、払戻金から「当たった3連単のオッズ」は分かる。
そこで、当たり目に対して モデルが付けていた確率 と 市場が付けていた確率（= 払戻率0.75 ÷ オッズ）を比べる。
当たり目の確率が高いほど（LogLossが小さいほど）正確。市場に勝てない限り、期待値で選んでも回収率は上がらない。

注意：市場の確率は払戻率を0.75と仮定した近似。
モデルの3連単確率は、2着・3着の補正（model/ordering.py）を期間の前半で決めて適用したもの。全期間の数字は前半を含むので参考値。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import ordering
from storage.gdrive import GDRIVE_ROOT, read_category, read_remote_parquet, sync_category

PAYOUT_RATE = 0.75


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v0", help="backtest/{model}/settled.parquet を使う（例: v0, v1_5y）")
    a = ap.parse_args()

    s = read_remote_parquet(f"backtest/{a.model}/settled.parquet")
    if s.empty:
        print(f"[MKT] backtest/{a.model} がありません")
        return
    lo, hi = s["date"].min(), s["date"].max()
    sync_category("payouts", sorted({int(lo[:4]), int(hi[:4])}))
    pay = read_category("payouts", lo, hi, columns=["race_id", "date", "trifecta_comb", "trifecta_payout", "is_aborted"])
    m = s.merge(pay[["race_id", "trifecta_comb", "trifecta_payout"]], on="race_id", how="inner")
    m = m[(m["trifecta_payout"] > 0) & m["trifecta_comb"].str.fullmatch(r"[1-6]-[1-6]-[1-6]")].reset_index(drop=True)

    m = m.sort_values("date").reset_index(drop=True)
    P = m[[f"p{i}" for i in range(1, 7)]].to_numpy()
    idx = np.array([[int(x) - 1 for x in c.split("-")] for c in m["trifecta_comb"]])
    # 2着・3着の補正（l2, l3）は期間の前半で決め、後半で確かめる
    half = len(m) // 2
    l2, l3 = ordering.fit(P[:half], idx[:half])
    m["p_harville"] = ordering.winning_combo_prob(P, idx)
    m["p_model"] = ordering.winning_combo_prob(P, idx, l2, l3)
    m["second_half"] = np.arange(len(m)) >= half
    m["odds"] = m["trifecta_payout"] / 100.0
    m["p_market"] = np.clip(PAYOUT_RATE / m["odds"], 1e-9, 1)
    m["ll_model"] = -np.log(np.clip(m["p_model"], 1e-9, 1))
    m["ll_market"] = -np.log(m["p_market"])
    m["ev_at_final"] = m["p_model"] * m["odds"]

    day = m.groupby("date")[["ll_model", "ll_market"]].mean()
    d = day["ll_model"] - day["ll_market"]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"[MKT] model={a.model} period={lo}..{hi} races={len(m):,}")
    m["ll_harville"] = -np.log(np.clip(m["p_harville"], 1e-9, 1))
    h2 = m[m["second_half"]]
    print(f"[MKT] 2着・3着の補正 l2={l2} l3={l3}（前半 {m['date'].iloc[0]}〜{m['date'].iloc[half-1]} で決定）")
    print(f"[MKT] 後半 {h2['date'].iloc[0]}〜{h2['date'].iloc[-1]} {len(h2):,}レースの当たり目LogLoss  "
          f"Harville {h2['ll_harville'].mean():.4f} → 補正後 {h2['ll_model'].mean():.4f}  市場 {h2['ll_market'].mean():.4f}")
    import json, subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / f"{a.model}.json"
        f.write_text(json.dumps({"l2": l2, "l3": l3, "fit_races": int(half), "model": a.model}), encoding="utf-8")
        subprocess.run(["rclone", "copyto", str(f), f"{GDRIVE_ROOT}/data/models/ordering/{a.model}.json"], capture_output=True)
    print(f"[MKT] 3連単の当たり目のLogLoss  モデル {m['ll_model'].mean():.4f}  市場 {m['ll_market'].mean():.4f}  "
          f"差 {m['ll_model'].mean() - m['ll_market'].mean():+.4f}（マイナスならモデルが正確） 日次t={t:+.1f}")

    bins = [0, 10, 30, 100, 300, 1000, 1e9]
    labels = ["〜10倍", "10〜30倍", "30〜100倍", "100〜300倍", "300〜1000倍", "1000倍〜"]
    m["bucket"] = pd.cut(m["odds"], bins=bins, labels=labels, right=False)
    g = m.groupby("bucket", observed=True).agg(
        races=("odds", "size"), ll_model=("ll_model", "mean"), ll_market=("ll_market", "mean"),
        ratio=("ev_at_final", lambda x: float(np.exp(np.log(np.clip(x / PAYOUT_RATE, 1e-9, None)).mean()))),
    )
    g["diff"] = g["ll_model"] - g["ll_market"]
    print("[MKT] 当たり目のオッズ帯ごと（ratio＝モデル確率÷市場確率の幾何平均。1より小さいとモデルが低く見ている）")
    print(g.round(4).to_string())

    # 参考：モデルが「期待値1.15以上」と見ていた当たり目の割合（買い目に入っていたはずの的中）
    sel = m[(m["ev_at_final"] >= 1.15) & (m["p_model"] >= 0.01) & (m["odds"] <= 500)]
    print(f"[MKT] 参考：確定オッズで期待値1.15以上と判定されていた当たり目 {len(sel):,}件 / {len(m):,}件、"
          f"その払戻合計 {int(sel['trifecta_payout'].sum()):,}円（1点100円。外れ目の点数は過去オッズがないため不明）")


if __name__ == "__main__":
    main()
