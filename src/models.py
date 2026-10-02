"""후보 모델 정의. 모든 모델은 log10(cycle_life)를 학습하고, 평가는 원래 사이클 단위로 한다.

M0 중앙값 기준선 / M1 단일 피처 선형회귀 / M2 ElasticNet / M3 Huber / M4 Gaussian Process / M5 RF·LightGBM
결측 대치와 표준화는 Pipeline 안에서 학습 fold에만 fit 한다.
"""
from __future__ import annotations

import itertools
import warnings

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import SEED
from features import FEATURE_SETS

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")

# 선택 대상 피처셋(DAY 1 계획). F3(정책 포함)은 보조 실험으로만 돌리고 선택하지 않는다.
SELECTABLE_SETS = ["F0_variance", "F1_capacity", "F2_full"]
COMPLEXITY = {"M0": 0, "M1": 1, "M2": 2, "M3": 2, "M4": 3, "M5": 4}


def _pipe(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def _gp():
    kernel = (ConstantKernel(1.0) * DotProduct(sigma_0=1.0)
              + ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(0.01))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=SEED)


def _lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=4, max_depth=2,
                         min_child_samples=3, subsample=0.8, subsample_freq=1, random_state=SEED, n_jobs=1, verbose=-1)


def candidates() -> list[dict]:
    """(id, family, feature_set, params, make) 목록."""
    out = [{"family": "M0", "model": "Median", "feature_set": "F0_variance", "params": "",
            "make": lambda: DummyRegressor(strategy="median")},
           {"family": "M1", "model": "Linear", "feature_set": "F0_variance", "params": "",
            "make": lambda: _pipe(LinearRegression())}]
    for fs in FEATURE_SETS:
        for a, l1 in itertools.product([0.0003, 0.001, 0.003, 0.01, 0.03, 0.1], [0.1, 0.5, 0.9]):
            out.append({"family": "M2", "model": "ElasticNet", "feature_set": fs, "params": f"alpha={a}, l1_ratio={l1}",
                        "make": (lambda a=a, l1=l1: _pipe(ElasticNet(alpha=a, l1_ratio=l1, max_iter=50000)))})
        for eps, a in itertools.product([1.35, 1.75], [0.0001, 0.01, 0.1]):
            out.append({"family": "M3", "model": "Huber", "feature_set": fs, "params": f"epsilon={eps}, alpha={a}",
                        "make": (lambda eps=eps, a=a: _pipe(HuberRegressor(epsilon=eps, alpha=a, max_iter=5000)))})
        out.append({"family": "M4", "model": "GaussianProcess", "feature_set": fs, "params": "linear+RBF+noise",
                    "make": lambda: _pipe(_gp())})
        for d, leaf in itertools.product([2, 3, None], [2, 4]):
            out.append({"family": "M5", "model": "RandomForest", "feature_set": fs, "params": f"max_depth={d}, min_leaf={leaf}",
                        "make": (lambda d=d, leaf=leaf: _pipe(RandomForestRegressor(
                            n_estimators=500, max_depth=d, min_samples_leaf=leaf, random_state=SEED, n_jobs=1)))})
        out.append({"family": "M5", "model": "LightGBM", "feature_set": fs, "params": "depth=2, leaves=4, lr=0.03, n=300",
                    "make": lambda: _pipe(_lgbm())})
    for i, c in enumerate(out):
        c["id"] = f"{c['family']}_{c['model']}_{c['feature_set']}_{i:03d}"
        c["n_features"] = len(FEATURE_SETS[c["feature_set"]]) if c["family"] != "M0" else 0
        c["selectable"] = c["feature_set"] in SELECTABLE_SETS
    return out
