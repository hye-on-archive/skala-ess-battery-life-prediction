"""학습·모델 선택·평가 전체 파이프라인.

1) Batch 1(정제 41셀)을 충전 정책 단위로 Hold-out 5정책 / Development 나머지로 나눈다.
2) Development에서 정책 단위 GroupKFold 5-fold × 5회로 모든 후보를 비교한다.
3) 1-SE 규칙으로 최종 모델을 고른다(최저 CV MAPE + 표준오차 안에서 피처 수·복잡도가 가장 작은 것).
4) Development 전체로 재학습한 고정 모델로 Hold-out, Batch 2, Batch 3를 한 번씩 평가한다.
5) 라벨 민감도, 부트스트랩 신뢰구간, 오류 분석 결과와 그림을 results/ 에 저장한다.

사용법
    python src/train.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

import evaluate as ev
from config import (CONTINUED_CELLS, FIG_DIR, NOISY_B3_CELLS, NOT_FINISHED_CELLS, RESULTS_DIR,
                    SEED, TARGET_MAPE)
from features import FEATURE_SETS, build_features
from models import COMPLEXITY, candidates
from split import holdout_policies, repeated_group_folds


def fit_predict(cand, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    cols = FEATURE_SETS[cand["feature_set"]]
    model = cand["make"]()
    model.fit(train[cols], np.log10(train.cycle_life))
    return 10 ** model.predict(test[cols])


def cross_validate(cand, dev: pd.DataFrame, splits) -> np.ndarray:
    scores = []
    for _, _, tr, va in splits:
        p = fit_predict(cand, dev.iloc[tr], dev.iloc[va])
        y = dev.iloc[va].cycle_life.to_numpy()
        scores.append(np.mean(np.abs(p - y) / y) * 100)
    return np.array(scores)


def _cv_by_index(i: int, dev: pd.DataFrame, splits) -> np.ndarray:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # 병렬 작업자에서도 src 모듈을 찾도록
    from models import candidates as _c
    return cross_validate(_c()[i], dev, splits)


def select(cv: pd.DataFrame) -> pd.Series:
    """1-SE 규칙: 최저 평균 MAPE + 그 표준오차 이내에서 (피처 수, 복잡도, 평균) 순으로 가장 단순한 후보."""
    sel = cv[cv.selectable & (cv.family != "M0")]
    best = sel.loc[sel.cv_mean.idxmin()]
    pool = sel[sel.cv_mean <= best.cv_mean + best.cv_se].copy()
    pool["complexity"] = pool.family.map(COMPLEXITY)
    return pool.sort_values(["n_features", "complexity", "cv_mean"]).iloc[0]


def label_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """민감도 분석용 라벨 버전. main = 원논문 규칙."""
    d = df.copy()
    if variant == "provided_labels":      # 제공 라벨을 그대로(B1 46, B3 44)
        d["cycle_life"] = d.cycle_life_provided
        d["in_model"] = d.cycle_life.notna()
    elif variant == "exclude_10":         # 이어측정·미도달 10셀을 모두 제외(B1 36)
        drop = list(CONTINUED_CELLS) + NOT_FINISHED_CELLS
        d.loc[d.cell_uid.isin(drop), "in_model"] = False
    return d


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = build_features()
    df.to_csv(RESULTS_DIR.parent / "data" / "processed" / "features.csv", index=False)

    b1 = df[(df.batch == "Batch 1") & df.in_model].reset_index(drop=True)
    ho_pols = holdout_policies(b1)
    dev = b1[~b1.policy.isin(ho_pols)].reset_index(drop=True)
    ho = b1[b1.policy.isin(ho_pols)].reset_index(drop=True)
    b2 = df[(df.batch == "Batch 2") & df.in_model].reset_index(drop=True)
    b3 = df[(df.batch == "Batch 3") & df.in_model].reset_index(drop=True)
    splits = repeated_group_folds(dev)
    split_info = {"holdout_policies": ho_pols, "n_dev_cells": len(dev), "n_dev_policies": int(dev.policy.nunique()),
                  "n_holdout_cells": len(ho), "holdout_cells": ho.cell_uid.tolist(),
                  "n_test_b2": len(b2), "n_test_b3": len(b3), "n_cv_folds": len(splits), "seed": SEED}
    print(json.dumps(split_info, ensure_ascii=False, indent=1))

    # ---- 2) 후보 비교 ----
    cands = candidates()
    rows, fold_rows = [], []
    # 후보마다 독립이므로 CPU 코어 수만큼 병렬 실행(결과는 순서·시드가 같아 직렬 실행과 동일)
    all_scores = Parallel(n_jobs=-1)(delayed(_cv_by_index)(i, dev, splits) for i in range(len(cands)))
    for c, s in zip(cands, all_scores):
        rows.append({k: c[k] for k in ["id", "family", "model", "feature_set", "params", "n_features", "selectable"]}
                    | {"cv_mean": s.mean(), "cv_std": s.std(ddof=1), "cv_se": s.std(ddof=1) / np.sqrt(len(s))})
        fold_rows += [{"id": c["id"], "repeat": r, "fold": k, "mape": v} for (r, k, _, _), v in zip(splits, s)]
    cv = pd.DataFrame(rows).sort_values("cv_mean")
    cv.round(4).to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    pd.DataFrame(fold_rows).round(4).to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    print(f"CV done: {len(cands)} candidates × {len(splits)} folds ({time.time() - t0:.0f}s)")

    # ---- 3) 선택 ----
    chosen = select(cv)
    cand = next(c for c in cands if c["id"] == chosen.id)
    best_overall = cv[cv.selectable & (cv.family != "M0")].iloc[0]
    print("selected:", chosen.id, chosen.params, round(chosen.cv_mean, 2))

    # ---- 4) 고정 모델 평가 ----
    preds = []
    for name, d in [("Valid", ho), ("Batch 2", b2), ("Batch 3", b3)]:
        p = fit_predict(cand, dev, d)
        preds.append(d[["cell_uid", "batch", "policy", "structure", "dq_logvar", "cycle_life"]]
                     .rename(columns={"cycle_life": "y_true"}).assign(set=name, y_pred=p))
    pred = pd.concat(preds, ignore_index=True)
    pred["pct_err"] = (pred.y_pred - pred.y_true) / pred.y_true * 100
    pred["life_in_train_range"] = pred.y_true.between(dev.cycle_life.min(), dev.cycle_life.max())
    pred["feature_in_train_range"] = pred.dq_logvar.between(dev.dq_logvar.min(), dev.dq_logvar.max())
    pred.round(3).to_csv(RESULTS_DIR / "predictions.csv", index=False)

    m = {s: ev.metrics(g.y_true, g.y_pred) for s, g in pred.groupby("set")}
    table = ev.performance_table(chosen.cv_mean, chosen.cv_std, m["Valid"]["MAPE"], m["Batch 2"]["MAPE"], m["Batch 3"]["MAPE"])
    table.to_csv(RESULTS_DIR / "model_performance.csv", index=False)
    print(table.to_string(index=False))

    # 보조 지표: MAE, 과대예측, 부트스트랩 CI, 그룹별
    detail = []
    for s, g in pred.groupby("set"):
        lo, hi = ev.bootstrap_mape(g.y_true, g.y_pred)
        detail.append({"set": s, "group": "all"} | m[s] | {"MAPE_CI95_low": lo, "MAPE_CI95_high": hi})
    for st, g in pred[pred.set == "Batch 2"].groupby("structure"):
        detail.append({"set": "Batch 2", "group": f"structure={st}"} | ev.metrics(g.y_true, g.y_pred))
    for flag, g in pred[pred.set == "Batch 2"].groupby("life_in_train_range"):
        detail.append({"set": "Batch 2", "group": f"life_in_train_range={flag}"} | ev.metrics(g.y_true, g.y_pred))
    for row in detail:  # 순위 보존 여부(Spearman): 절대 수준이 틀려도 셀 간 순서는 맞는지
        g = pred[pred.set == row["set"]]
        if row["group"].startswith("structure="):
            g = g[g.structure == row["group"].split("=")[1]]
        elif row["group"].startswith("life_in"):
            g = g[g.life_in_train_range == (row["group"].endswith("True"))]
        row["spearman"] = spearmanr(g.y_true, g.y_pred)[0]
        row["log10_offset"] = float(np.mean(np.log10(g.y_pred) - np.log10(g.y_true)))
    pd.DataFrame(detail).round(3).to_csv(RESULTS_DIR / "metrics_detail.csv", index=False)

    # ---- 사후 분석(탐색, 테스트 정답을 본 뒤 수행): 소수 셀로 배치 오프셋 보정 ----
    # 새 배치에서 k개 셀의 실제 수명만 알면 log 오프셋을 추정해 나머지 셀 예측을 보정할 수 있는지 확인
    rng = np.random.default_rng(SEED)
    recal = []
    for s in ["Batch 2", "Batch 3"]:
        g = pred[pred.set == s].reset_index(drop=True)
        lr = np.log10(g.y_pred) - np.log10(g.y_true)
        recal.append({"set": s, "k": 0, "MAPE_mean": ev.metrics(g.y_true, g.y_pred)["MAPE"], "MAPE_p10": np.nan, "MAPE_p90": np.nan})
        for k in [1, 2, 3, 5]:
            vals = []
            for _ in range(500):
                idx = rng.choice(len(g), k, replace=False)
                rest = np.setdiff1d(np.arange(len(g)), idx)
                adj = g.y_pred.to_numpy()[rest] / 10 ** lr[idx].mean()
                vals.append(np.mean(np.abs(adj - g.y_true.to_numpy()[rest]) / g.y_true.to_numpy()[rest]) * 100)
            recal.append({"set": s, "k": k, "MAPE_mean": np.mean(vals), "MAPE_p10": np.percentile(vals, 10), "MAPE_p90": np.percentile(vals, 90)})
    recal = pd.DataFrame(recal)
    recal.round(2).to_csv(RESULTS_DIR / "posthoc_recalibration.csv", index=False)

    # 최악 오류 셀
    worst = (pred.assign(abs_err=pred.pct_err.abs()).sort_values("abs_err", ascending=False)
             .groupby("set").head(5).sort_values(["set", "abs_err"], ascending=[True, False]))
    worst.round(2).to_csv(RESULTS_DIR / "worst_cells.csv", index=False)

    # 참고: 패밀리별 CV 최고 설정의 테스트 성능(선택에는 쓰지 않음)
    ref = []
    for fam, g in cv[cv.selectable].groupby("family"):
        r = g.iloc[0]
        c = next(x for x in cands if x["id"] == r.id)
        row = {"family": fam, "model": r.model, "feature_set": r.feature_set, "params": r.params, "cv_mean": r.cv_mean}
        for name, d in [("Valid", ho), ("Batch 2", b2), ("Batch 3", b3)]:
            p = fit_predict(c, dev, d)
            row[f"{name} MAPE"] = np.mean(np.abs(p - d.cycle_life) / d.cycle_life) * 100
        ref.append(row)
    pd.DataFrame(ref).round(2).to_csv(RESULTS_DIR / "reference_by_family.csv", index=False)

    # 피처셋 비교(패밀리 × 피처셋별 CV 최고)
    abl = (cv[cv.family != "M0"].sort_values("cv_mean").groupby(["family", "model", "feature_set"]).head(1)
           .sort_values(["family", "model", "feature_set"]))
    abl["label"] = abl.family + " " + abl.model
    abl.round(3).to_csv(RESULTS_DIR / "feature_set_ablation.csv", index=False)

    # ---- 5) 라벨 민감도: 선택한 설정을 고정하고 라벨 버전만 바꿔 다시 학습·평가 ----
    sens = []
    for variant in ["paper_rules", "provided_labels", "exclude_10"]:
        d = label_variant(df, variant)
        vb1 = d[(d.batch == "Batch 1") & d.in_model].reset_index(drop=True)
        vdev = vb1[~vb1.policy.isin(ho_pols)].reset_index(drop=True)
        vho = vb1[vb1.policy.isin(ho_pols)].reset_index(drop=True)
        vb3 = d[(d.batch == "Batch 3") & d.in_model].reset_index(drop=True)
        s = cross_validate(cand, vdev, repeated_group_folds(vdev))
        row = {"variant": variant, "n_B1": len(vb1), "n_dev": len(vdev), "n_B3": len(vb3), "cv_mean": s.mean()}
        for name, t in [("Valid", vho), ("Batch 2", b2), ("Batch 3", vb3)]:
            p = fit_predict(cand, vdev, t)
            row[f"{name} MAPE"] = np.mean(np.abs(p - t.cycle_life) / t.cycle_life) * 100
        sens.append(row)
    pd.DataFrame(sens).round(2).to_csv(RESULTS_DIR / "sensitivity_labels.csv", index=False)

    # 보조: Batch 1 전체(41셀)로 재학습했을 때의 테스트 성능
    full = []
    for name, d in [("Batch 2", b2), ("Batch 3", b3)]:
        p = fit_predict(cand, b1, d)
        full.append({"trained_on": "Batch 1 all (41)", "set": name} | ev.metrics(d.cycle_life, p))
    pd.DataFrame(full).round(3).to_csv(RESULTS_DIR / "refit_b1_all.csv", index=False)

    # 최종 모델 계수(선형 계열이면)
    final = cand["make"]().fit(dev[FEATURE_SETS[cand["feature_set"]]], np.log10(dev.cycle_life))
    est = final.steps[-1][1]
    coefs = {}
    if hasattr(est, "coef_"):
        coefs = dict(zip(FEATURE_SETS[cand["feature_set"]], np.round(est.coef_, 5).tolist()))
        coefs["intercept"] = round(float(est.intercept_), 5)

    summary = {"selected": {k: (v.item() if hasattr(v, "item") else v) for k, v in chosen.items()},
               "best_cv_overall": {"id": best_overall.id, "cv_mean": float(best_overall.cv_mean), "cv_se": float(best_overall.cv_se)},
               "coefficients_standardized": coefs, "split": split_info, "target_mape": TARGET_MAPE,
               "metrics": m, "runtime_sec": round(time.time() - t0, 1)}
    (RESULTS_DIR / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    # ---- 그림 ----
    ev.plot_pred_vs_actual(pred)
    ev.plot_cv_comparison(abl)
    ev.plot_error_by_group(pred)
    ev.plot_feature_shift(pd.concat([dev, b2, b3], ignore_index=True), range(len(dev)))
    ev.plot_recalibration(recal)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
