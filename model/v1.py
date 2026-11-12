"""学習モデル v1（Claude実装）。

LightGBMで各艇の点数を出し、同じレースの6艇でソフトマックスをかけて1着確率にする（Group-Softmax）。
学習後、検証データで温度Tを決めて確率を補正する（温度スケーリング）。
データは必ず race_id・艇番順で、1レース6行そろっていること。
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from model.features import CATEGORICAL, FEATURES

MODEL_VER = "v1"


def softmax6(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(-1, 6) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _objective(preds, dtrain):
    y = dtrain.get_label().reshape(-1, 6)
    p = softmax6(preds)
    grad = (p - y).ravel()
    hess = np.maximum(p * (1 - p), 1e-6).ravel()
    return grad, hess


def race_logloss(p: np.ndarray, y: np.ndarray) -> float:
    """1レースあたりの対数損失。同着は y を 1/同着数 ずつ持つ"""
    y = y.reshape(-1, 6)
    return float(-(y * np.log(np.clip(p, 1e-9, 1))).sum(axis=1).mean())


def _metric(preds, dvalid):
    return "race_logloss", race_logloss(softmax6(preds), dvalid.get_label()), False


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame, rounds: int = 3000):
    params = {
        "objective": _objective,
        "metric": "None",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "lambda_l2": 5.0,
        "max_cat_to_onehot": 32,
        "verbose": -1,
        "seed": 42,
    }
    dtr = lgb.Dataset(train_df[FEATURES], label=train_df["y"], categorical_feature=CATEGORICAL, free_raw_data=False)
    dva = lgb.Dataset(valid_df[FEATURES], label=valid_df["y"], categorical_feature=CATEGORICAL, reference=dtr)
    booster = lgb.train(
        params, dtr, num_boost_round=rounds, valid_sets=[dva], valid_names=["valid"], feval=_metric,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return booster


def raw_scores(booster, df: pd.DataFrame) -> np.ndarray:
    return booster.predict(df[FEATURES], num_iteration=booster.best_iteration or None, raw_score=True)


def fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    grid = np.round(np.arange(0.60, 1.81, 0.02), 2)
    losses = [race_logloss(softmax6(scores, t), y) for t in grid]
    return float(grid[int(np.argmin(losses))])


def predict_frame(booster, temperature: float, feat: pd.DataFrame, data_window: str) -> pd.DataFrame:
    """特徴量（race_id・艇番順、6行ずつ）から、1レース1行の予想（p1〜p6）を作る"""
    p = softmax6(raw_scores(booster, feat), temperature)
    head = feat.iloc[::6][["race_id", "date", "venue_code", "race_no"]].reset_index(drop=True)
    out = head.assign(model_ver=MODEL_VER, data_window=data_window, has_beforeinfo=False, n_boats=6)
    out["race_no"] = out["race_no"].astype(int)
    for i in range(6):
        out[f"p{i+1}"] = p[:, i]
    return out


def save(booster, temperature: float, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"), num_iteration=booster.best_iteration or None)
    (out_dir / "meta.json").write_text(
        json.dumps({"temperature": temperature, "features": FEATURES, **meta}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load(model_dir: Path):
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    return booster, float(meta["temperature"]), meta
