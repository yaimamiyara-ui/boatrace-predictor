"""v1用の特徴量づくり（Claude実装）。

すべて「そのレースの前日までに分かっている情報」だけで作る:
  - 番組表の値（勝率・2連率・級別など）は、レース前に公表される値なのでそのまま使える
  - 選手の最近の成績（直近180日・30日、枠別は365日）は、成績データのうち日付が当日より前の行だけを集計
  - 場×枠の1着率は、その月の初日より前の直近3年で計算（v0と同じ）
"""
import numpy as np
import pandas as pd

from model import v0

GRADE = {"A1": 3, "A2": 2, "B1": 1, "B2": 0}
RESULT_COLS = ["race_id", "date", "venue_code", "boat_no", "racer_id", "course",
               "st_time", "st_status", "finish_order", "finish_status"]
PROGRAM_COLS = ["race_id", "date", "venue_code", "race_no", "boat_no", "racer_id", "age", "weight", "grade",
                "nat_win_rate", "nat_2ren_rate", "loc_win_rate", "loc_2ren_rate", "motor_2ren_rate", "boat_2ren_rate"]

# レース内で比べる特徴量（平均との差と順位を追加する）
RELATIVE = ["nat_win_rate", "nat_2ren_rate", "loc_win_rate", "motor_2ren_rate", "boat_2ren_rate",
            "r180_win", "r180_top3", "r180_st", "lane365_win", "grade_n"]

FEATURES = (
    ["boat_no", "venue", "race_no", "age", "weight", "grade_n",
     "nat_win_rate", "nat_2ren_rate", "loc_win_rate", "loc_2ren_rate", "motor_2ren_rate", "boat_2ren_rate",
     "cw",
     "r180_n", "r180_win", "r180_top2", "r180_top3", "r180_st", "r180_f",
     "r30_n", "r30_win", "r30_top3",
     "lane365_n", "lane365_win", "lane365_top2",
     "inner_st_diff", "inner_win_diff", "boat1_nat_win", "boat1_r180_st"]
    + [f"{c}_diff" for c in RELATIVE] + [f"{c}_rank" for c in RELATIVE]
)
CATEGORICAL = ["venue"]


def _cum_tables(results: pd.DataFrame):
    """選手ごと・選手×枠ごとの、日付順の累計表（その日までの合計）"""
    r = results.copy()
    r["d"] = pd.to_datetime(r["date"])
    fo = r["finish_order"]
    st_ok = r["st_time"].notna() & (r["st_status"] == "NORMAL")
    r["n"] = 1.0
    r["w1"] = (fo == 1).astype(float)
    r["w2"] = (fo <= 2).astype(float)
    r["w3"] = (fo <= 3).astype(float)
    r["st_n"] = st_ok.astype(float)
    r["st_sum"] = r["st_time"].where(st_ok, 0.0).astype(float)
    r["f"] = (r["st_status"] == "F").astype(float)

    cols = ["n", "w1", "w2", "w3", "st_n", "st_sum", "f"]
    racer = r.groupby(["racer_id", "d"], as_index=False)[cols].sum().sort_values(["racer_id", "d"])
    racer[cols] = racer.groupby("racer_id")[cols].cumsum()

    lcols = ["n", "w1", "w2"]
    lane = r.groupby(["racer_id", "boat_no", "d"], as_index=False)[lcols].sum().sort_values(["racer_id", "boat_no", "d"])
    lane[lcols] = lane.groupby(["racer_id", "boat_no"])[lcols].cumsum()
    return racer.sort_values("d"), lane.sort_values("d")


def _window(left: pd.DataFrame, cum: pd.DataFrame, by: list[str], days: int, cols: list[str]) -> pd.DataFrame:
    """left の各行について、[当日-days, 当日) の合計を返す（当日は含めない）"""
    L = left[["_row", "d"] + by].sort_values("d")
    hi = pd.merge_asof(L, cum, on="d", by=by, allow_exact_matches=False)
    L2 = L.assign(d=L["d"] - pd.Timedelta(days=days)).sort_values("d")
    lo = pd.merge_asof(L2, cum, on="d", by=by, allow_exact_matches=False)
    hi = hi.set_index("_row")[cols].fillna(0.0)
    lo = lo.set_index("_row")[cols].fillna(0.0)
    return (hi - lo).reindex(left["_row"])


def _course_rate(prog: pd.DataFrame, results: pd.DataFrame, col: str = "boat_no") -> np.ndarray:
    """場×枠（col="course" なら場×進入コース）の1着率（その月の初日より前の直近3年）"""
    out = np.full(len(prog), np.nan)
    months = prog["date"].str[:7]
    for ym in sorted(months.unique()):
        rates = v0.course_win_rates(results, f"{ym}-01")
        idx = np.where(months.to_numpy() == ym)[0]
        sub = prog.iloc[idx]
        out[idx] = [rates.get(v, rates["_all"])[int(b) - 1] if b == b and 1 <= b <= 6 else np.nan
                    for v, b in zip(sub["venue_code"], sub[col])]
    return out


def build(program: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """番組表（1行=1艇）に特徴量を付けて返す。6艇そろったレースだけを、race_id・艇番順に並べる。
    results は、番組表の最も古い日付より少なくとも3年前からの成績を渡すこと。"""
    p = program[PROGRAM_COLS].copy()
    cnt = p.groupby("race_id")["boat_no"].transform("nunique")
    size = p.groupby("race_id")["boat_no"].transform("size")
    p = p[(cnt == 6) & (size == 6)].sort_values(["race_id", "boat_no"]).reset_index(drop=True)
    p["_row"] = np.arange(len(p))
    p["d"] = pd.to_datetime(p["date"])
    p["venue"] = p["venue_code"].astype(int)
    p["grade_n"] = p["grade"].map(GRADE).astype(float)

    racer_cum, lane_cum = _cum_tables(results)
    cols = ["n", "w1", "w2", "w3", "st_n", "st_sum", "f"]
    w180 = _window(p, racer_cum, ["racer_id"], 180, cols)
    w30 = _window(p, racer_cum, ["racer_id"], 30, cols)
    wl = _window(p, lane_cum, ["racer_id", "boat_no"], 365, ["n", "w1", "w2"])

    # 出走が少ない選手の率が極端な値にならないよう、平均的な値へ寄せる（10走分。Geminiレビューで追加）
    PRIOR_N, ST_PRIOR_N, ST_MEAN = 10.0, 5.0, 0.17

    def shrink(num, den, prior, k=PRIOR_N):
        return (num.to_numpy() + k * np.asarray(prior, dtype=float)) / (den.to_numpy() + k)

    p["cw"] = _course_rate(p, results)
    p["r180_n"] = w180["n"].to_numpy()
    p["r180_win"] = shrink(w180["w1"], w180["n"], 1 / 6)
    p["r180_top2"] = shrink(w180["w2"], w180["n"], 2 / 6)
    p["r180_top3"] = shrink(w180["w3"], w180["n"], 3 / 6)
    p["r180_st"] = shrink(w180["st_sum"], w180["st_n"], ST_MEAN, ST_PRIOR_N)
    p["r180_f"] = w180["f"].to_numpy()
    p["r30_n"] = w30["n"].to_numpy()
    p["r30_win"] = shrink(w30["w1"], w30["n"], p["r180_win"])
    p["r30_top3"] = shrink(w30["w3"], w30["n"], p["r180_top3"])
    p["lane365_n"] = wl["n"].to_numpy()
    p["lane365_win"] = shrink(wl["w1"], wl["n"], p["cw"])
    p["lane365_top2"] = shrink(wl["w2"], wl["n"], p["r180_top2"])

    g = p.groupby("race_id", sort=False)
    for c in RELATIVE:
        p[f"{c}_diff"] = p[c] - g[c].transform("mean")
        # 順位は「良いほうが1位」。平均STだけは小さいほど良い
        p[f"{c}_rank"] = g[c].rank(ascending=(c == "r180_st"), method="average", na_option="bottom")

    # 1つ内側の艇との差と、1号艇の強さ（展開は内隣と1号艇に左右されるため。Geminiレビューで追加）
    p["inner_st_diff"] = p["r180_st"] - g["r180_st"].shift(1).fillna(p["r180_st"])
    p["inner_win_diff"] = p["nat_win_rate"] - g["nat_win_rate"].shift(1).fillna(p["nat_win_rate"])
    p["boat1_nat_win"] = g["nat_win_rate"].transform("first")
    p["boat1_r180_st"] = g["r180_st"].transform("first")

    p[FEATURES] = p[FEATURES].astype("float32")
    p["venue"] = p["venue"].astype("int32")
    return p.drop(columns=["_row", "d"])


# ---- 直前情報ありモデル（v1p）用の追加特徴量 ----
PRE_COLS = ["race_id", "boat_no", "exhibit_time", "course", "wind_speed", "wave_height"]
PRE_FEATURES = ["exhibit_time", "exhibit_diff", "exhibit_rank", "course", "course_shift", "cw_course",
                "wind_speed", "wave_height"]


def add_pre(feat: pd.DataFrame, pre: pd.DataFrame, results: pd.DataFrame | None = None, rates: dict | None = None) -> pd.DataFrame:
    """直前情報（展示タイム・進入コース・風速・波高）の特徴量を足す。
    pre は PRE_COLS を持つ表。学習時は成績ファイルの値（展示タイム・本番の進入・レース時の風と波）を使い、
    本番では締切前の直前情報（展示タイム・スタート展示の進入・その時点の風と波）を渡す。"""
    out = feat.merge(pre[PRE_COLS].drop_duplicates(["race_id", "boat_no"]), on=["race_id", "boat_no"], how="left")
    out = out.sort_values(["race_id", "boat_no"]).reset_index(drop=True)
    g = out.groupby("race_id", sort=False)
    out["exhibit_diff"] = out["exhibit_time"] - g["exhibit_time"].transform("mean")
    out["exhibit_rank"] = g["exhibit_time"].rank(ascending=True, method="average", na_option="bottom")
    out["course_shift"] = out["course"] - out["boat_no"]
    if rates is not None:  # 本番用：朝に保存した 場×コースの1着率 を使う
        out["cw_course"] = [rates.get(v, rates["_all"])[int(c) - 1] if c == c and 1 <= c <= 6 else np.nan
                            for v, c in zip(out["venue_code"], out["course"])]
    else:
        out["cw_course"] = _course_rate(out, results, col="course")
    out[PRE_FEATURES] = out[PRE_FEATURES].astype("float32")
    return out


def labels(feat: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """学習・評価用のラベル y を付ける。1着がいないレース、返還（F・L・欠場）があったレースは除く。
    同着は 1/同着数 ずつ。戻り値は race_id・艇番順のまま、6行ずつそろっている。"""
    refund = set(results.loc[results["st_status"].isin(["F", "L", "K"]), "race_id"])
    win = results[results["finish_order"] == 1][["race_id", "boat_no"]].assign(y=1.0)
    k = win.groupby("race_id")["y"].transform("size")
    win["y"] = 1.0 / k
    ok = set(win["race_id"]) - refund
    out = feat[feat["race_id"].isin(ok)].merge(win, on=["race_id", "boat_no"], how="left")
    out["y"] = out["y"].fillna(0.0)
    return out.sort_values(["race_id", "boat_no"]).reset_index(drop=True)
