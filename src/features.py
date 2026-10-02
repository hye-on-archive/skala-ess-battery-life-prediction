"""중간 파일 → 셀 단위 피처 테이블.

모든 피처는 사이클 100까지의 기록만 사용한다(FEATURE_MAX_CYCLE).
라벨 정제는 원논문 공식 코드 규칙(config.py)을 따르며, 제공 라벨도 함께 남겨 민감도 분석에 쓴다.

사용법
    python src/features.py            # data/processed/features.csv 생성
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from config import (CONTINUED_CELLS, FEATURE_MAX_CYCLE, NOISY_B3_CELLS,
                    NOT_FINISHED_CELLS, PROCESSED_DIR)

# 피처셋: DAY 1 전략의 기본 → 확장 A → 확장 B, 그리고 정책 정보를 더한 보조 실험
FEATURE_SETS = {
    "F0_variance": ["dq_logvar"],
    "F1_capacity": ["dq_logvar", "qd_mean", "qd_slope"],
    "F2_full": ["dq_logvar", "qd_mean", "qd_slope", "ir_change", "tavg_median"],
    "F3_full_policy": ["dq_logvar", "qd_mean", "qd_slope", "ir_change", "tavg_median",
                       "avg_crate_0_80", "switch_soc"],
}
# 예측에 절대 넣지 않는 열(미래 정보·식별자)
NON_FEATURE_COLS = ["cell_uid", "batch", "policy", "structure", "cycle_life", "cycle_life_provided",
                    "in_model", "exclude_reason", "label_corrected", "n_cycles_recorded",
                    "last_cycle", "end_QD"]


def parse_policy(policy: str) -> dict:
    """'5.4C(40%)-3.6C' → 1단계 C-rate, 전환 SOC, 2단계 C-rate, 0→80% 평균 C-rate."""
    m = re.match(r"([\d.]+)C\((\d+)%\)-([\d.]+)C", str(policy))
    if not m:
        return {"c_rate_1": np.nan, "switch_soc": np.nan, "c_rate_2": np.nan, "avg_crate_0_80": np.nan}
    c1, q, c2 = float(m[1]), float(m[2]), float(m[3])
    hours = q / 100 / c1 + (80 - q) / 100 / c2           # 0→80% 충전에 걸리는 이론 시간(h)
    return {"c_rate_1": c1, "switch_soc": q, "c_rate_2": c2, "avg_crate_0_80": 0.8 / hours}


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 5 else np.nan


def cell_features(early: pd.DataFrame, q10: np.ndarray, q100: np.ndarray) -> dict:
    """한 셀의 사이클 2~100 요약값과 Q10·Q100 곡선으로 피처를 만든다."""
    dq = q100 - q10
    var = float(np.var(dq, ddof=0))
    # 0 이하·1.5Ah 초과 용량은 측정 이상 후보라 용량 피처에서만 제외(원본은 보존)
    qd_ok = early[(early.QD > 0) & (early.QD < 1.5)]
    ir_ok = early[early.IR > 0]
    ir_start = ir_ok[ir_ok.cycle.between(2, 10)].IR
    ir_end = ir_ok[ir_ok.cycle.between(91, 100)].IR
    slope_part = qd_ok[qd_ok.cycle >= 10]
    return {
        "dq_logvar": np.log10(max(var, 1e-16)),
        "dq_logabsmin": np.log10(abs(dq.min())),
        "dq_mean": float(dq.mean()),
        "dq_skew": float(skew(dq)),
        "dq_kurt": float(kurtosis(dq)),
        "qd_mean": float(qd_ok.QD.mean()),
        "qd_slope": _slope(slope_part.cycle.to_numpy(float), slope_part.QD.to_numpy(float)) * 1000,  # mAh/cycle
        "ir_change": (ir_end.mean() - ir_start.mean()) * 1000 if len(ir_start) and len(ir_end) else np.nan,  # mΩ
        "tavg_median": float(early.Tavg.median()),
        "chargetime_median": float(early.loc[early.chargetime > 0, "chargetime"].median()),
    }


def curate_labels(cells: pd.DataFrame) -> pd.DataFrame:
    """원논문 규칙으로 수명 라벨을 정제한다. 제공 라벨은 cycle_life_provided 에 그대로 남긴다."""
    out = cells.copy()
    out["cycle_life"] = out.cycle_life_provided
    out["label_corrected"] = out.cell_uid.isin(CONTINUED_CELLS)
    for uid, add in CONTINUED_CELLS.items():
        out.loc[out.cell_uid == uid, "cycle_life"] += add
    reason = pd.Series("", index=out.index)
    reason[out.cycle_life_provided.isna()] = "no_label"
    reason[out.cell_uid.isin(NOT_FINISHED_CELLS)] = "not_reached_eol"
    reason[out.cell_uid.isin(NOISY_B3_CELLS)] = "noisy_channel"
    out["exclude_reason"] = reason
    out.loc[reason != "", "cycle_life"] = np.nan
    out["in_model"] = reason == ""
    return out


def build_features(processed_dir=PROCESSED_DIR) -> pd.DataFrame:
    meta = pd.read_csv(processed_dir / "cells_meta.csv")
    summary = pd.read_csv(processed_dir / "summary.csv.gz")
    curves = np.load(processed_dir / "qdlin.npz")
    early_all = summary[(summary.cycle >= 2) & (summary.cycle <= FEATURE_MAX_CYCLE)]
    rows = []
    for uid, early in early_all.groupby("cell_uid"):
        row = {"cell_uid": uid}
        row.update(cell_features(early, curves[f"{uid}_q10"], curves[f"{uid}_q100"]))
        rows.append(row)
    feats = meta.merge(pd.DataFrame(rows), on="cell_uid", how="left")
    feats = feats.join(feats.policy.apply(parse_policy).apply(pd.Series))
    feats["structure"] = np.where(feats.policy.str.contains("newstructure"), "new", "standard")
    return curate_labels(feats)


if __name__ == "__main__":
    df = build_features()
    df.to_csv(PROCESSED_DIR / "features.csv", index=False)
    print(df.groupby("batch").agg(cells=("cell_uid", "size"), in_model=("in_model", "sum")))
    print("saved", PROCESSED_DIR / "features.csv")
