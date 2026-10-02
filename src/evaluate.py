"""지표 계산, 성능표, 그림."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FIG_DIR, TARGET_MAPE


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {"n": len(y_true),
            "MAPE": float(np.mean(np.abs(err) / y_true) * 100),
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "bias_pct": float(np.mean(err / y_true) * 100),       # (+)면 평균적으로 과대예측
            "over_share": float(np.mean(err > 0) * 100)}          # 과대예측 셀 비율(%)


def bootstrap_mape(y_true, y_pred, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ape = np.abs(np.asarray(y_pred) - np.asarray(y_true)) / np.asarray(y_true) * 100
    boots = [ape[rng.integers(0, len(ape), len(ape))].mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def performance_table(cv_mean, cv_std, valid, test2, test3) -> pd.DataFrame:
    """과제 Reporting format(회귀 + Batch 3 추가). Gap은 (+)가 성능 저하가 되도록 계산."""
    rows = [
        ("Train (Batch 1 CV)", cv_mean, f"정책 단위 GroupKFold 5-fold × 5회 검증 fold 평균 (표준편차 {cv_std:.2f})"),
        ("Valid (Batch 1 Hold-out)", valid, "학습에 쓰지 않은 B1 충전 정책 5개"),
        ("Test (Batch 2)", test2, "최종 필수 평가"),
        ("Gap (Train-Valid)", valid - cv_mean, "Valid − Train, (+) : 과적합 의심"),
        ("Gap (Valid-Test)", test2 - valid, "Test − Valid, (+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", test2 - TARGET_MAPE, f"Test − {TARGET_MAPE}, Target : 원논문 {TARGET_MAPE}%"),
        ("Test (Batch 3)", test3, "추가 평가 (선택)"),
        ("Gap (Batch2-Batch3)", test3 - test2, "Batch 3 − Batch 2, Test 성능 간 비교"),
        ("Gap (Target-Test), Batch 3", test3 - TARGET_MAPE, f"Batch 3 − {TARGET_MAPE}, 원논문 성능 비교"),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"]).round({"MAPE (%)": 2})


# ---------- figures (labels in English so they render on any machine) ----------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "font.size": 10, "savefig.dpi": 200})
    return plt


def _log_ticks(ax, which="both", ticks=(300, 400, 600, 1000, 2000)):
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    if which in ("x", "both"):
        ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(t) for t in ticks])
    if which in ("y", "both"):
        ax.set_yscale("log"); ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels([str(t) for t in ticks])


COLORS = {"Valid": "#1F4E79", "Batch 2": "#D9822B", "Batch 3": "#2E8B74"}


def plot_pred_vs_actual(pred: pd.DataFrame, path=FIG_DIR / "pred_vs_actual.png"):
    plt = _plt()
    sets = ["Valid", "Batch 2", "Batch 3"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    lo, hi = 300, 2600
    for ax, s in zip(axes, sets):
        d = pred[pred.set == s]
        ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1], color="#888", alpha=0.12, lw=0, label="±10%")
        ax.plot([lo, hi], [lo, hi], color="#555", lw=1)
        if s == "Batch 2":
            for st, mk in [("standard", "o"), ("new", "s")]:
                e = d[d.structure == st]
                ax.scatter(e.y_true, e.y_pred, s=22, marker=mk, color=COLORS[s],
                           facecolor=COLORS[s] if st == "standard" else "white", label=f"{st} (n={len(e)})")
        else:
            ax.scatter(d.y_true, d.y_pred, s=22, color=COLORS[s], label=f"n={len(d)}")
        mape = np.mean(np.abs(d.y_pred - d.y_true) / d.y_true) * 100
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual cycle life",
               ylabel="Predicted cycle life", title=f"{s}: MAPE {mape:.1f}%")
        _log_ticks(ax, "both")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_cv_comparison(ablation: pd.DataFrame, path=FIG_DIR / "cv_comparison.png"):
    plt = _plt()
    piv = ablation.pivot(index="label", columns="feature_set", values="cv_mean")
    err = ablation.pivot(index="label", columns="feature_set", values="cv_se")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cols = [c for c in ["F0_variance", "F1_capacity", "F2_full", "F3_full_policy"] if c in piv]
    w = 0.8 / len(cols)
    x = np.arange(len(piv))
    for i, c in enumerate(cols):
        ax.bar(x + i * w - 0.4 + w / 2, piv[c], w, yerr=err[c], capsize=2, label=c,
               color=["#9AA5B1", "#5B8DB8", "#1F4E79", "#D9822B"][i])
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=0)
    ax.set_ylabel("Batch 1 CV MAPE (%)"); ax.set_title("Best configuration per model × feature set (policy-grouped CV, mean ± SE)")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_error_by_group(pred: pd.DataFrame, path=FIG_DIR / "error_by_group.png"):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    for s in ["Valid", "Batch 2", "Batch 3"]:
        d = pred[pred.set == s]
        if s == "Batch 2":
            for st, fc in [("standard", COLORS[s]), ("new", "white")]:
                e = d[d.structure == st]
                ax.scatter(e.y_true, e.pct_err, s=20, color=COLORS[s], facecolor=fc, label=f"Batch 2 {st}")
        else:
            ax.scatter(d.y_true, d.pct_err, s=20, color=COLORS[s], label=s)
    ax.axhline(0, color="#555", lw=1)
    ax.axvline(534, color="#C0392B", lw=1, ls=":"); ax.text(540, ax.get_ylim()[1] * 0.9, "B1 min life 534", color="#C0392B", fontsize=8)
    _log_ticks(ax, "x")
    ax.set(xlabel="Actual cycle life", ylabel="Error (%)  (+) = over-prediction",
           title="Signed error vs actual life (B2 filled = standard, open = new)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    groups = [("Valid", pred[pred.set == "Valid"]), ("B2 standard", pred[(pred.set == "Batch 2") & (pred.structure == "standard")]),
              ("B2 new", pred[(pred.set == "Batch 2") & (pred.structure == "new")]), ("Batch 3", pred[pred.set == "Batch 3"])]
    vals = [np.abs(g.pct_err) for _, g in groups]
    ax.boxplot(vals, tick_labels=[f"{n}\n(n={len(g)})" for n, g in groups], showfliers=True)
    for i, v in enumerate(vals):
        ax.text(i + 1.25, np.mean(v), f"{np.mean(v):.1f}%", fontsize=8, va="center")
    ax.set_ylabel("Absolute error (%)"); ax.set_title("Absolute % error by group (text = MAPE)")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_feature_shift(df: pd.DataFrame, dev_idx, path=FIG_DIR / "feature_shift.png"):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    rng = np.random.default_rng(0)
    dev = df.loc[dev_idx]
    rows = [("B1 dev", dev, "#1F4E79"), ("B2", df[df.batch == "Batch 2"], "#D9822B"), ("B3", df[df.batch == "Batch 3"], "#2E8B74")]
    for ax, col, lab in [(axes[0], "cycle_life", "Cycle life"), (axes[1], "dq_logvar", "log10 Var(ΔQ100-10)")]:
        ax.axvspan(dev[col].min(), dev[col].max(), color="#1F4E79", alpha=0.08, lw=0)
        for i, (n, d, c) in enumerate(rows):
            ax.scatter(d[col], np.full(len(d), 2 - i) + rng.uniform(-0.15, 0.15, len(d)), s=14, color=c)
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels([n for n, _, _ in rows]); ax.set_xlabel(lab)
        if col == "cycle_life":
            _log_ticks(ax, "x")
    axes[0].set_title("Shaded = range seen in training (B1 development)")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_recalibration(recal: pd.DataFrame, path=FIG_DIR / "posthoc_recalibration.png"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for s, g in recal.groupby("set"):
        c = COLORS[s]
        ax.plot(g.k, g.MAPE_mean, marker="o", color=c, label=s)
        ax.fill_between(g.k, g.MAPE_p10, g.MAPE_p90, color=c, alpha=0.15, lw=0)
    ax.axhline(TARGET_MAPE, color="#555", ls="--", lw=1); ax.text(4.1, TARGET_MAPE + 0.6, "paper 9.1%", fontsize=8)
    ax.set(xlabel="Labelled cells from the new batch used for offset (k)", ylabel="MAPE on remaining cells (%)",
           title="Post-hoc: batch offset recalibration (band = 10-90%)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
