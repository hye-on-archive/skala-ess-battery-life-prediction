"""Batch 1 분할: 충전 정책 단위 Hold-out + 정책 단위 반복 GroupKFold.

같은 정책의 셀이 학습과 검증에 동시에 들어가면 성능이 부풀려지므로(과제 문서의 누수 지적),
모든 분할은 정책(policy) 단위로 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import N_FOLDS, N_HOLDOUT_POLICIES, N_REPEATS, SEED


def holdout_policies(b1: pd.DataFrame, n: int = N_HOLDOUT_POLICIES, seed: int = SEED) -> list[str]:
    """정책 평균 log 수명 순으로 정렬해 n개 구간으로 나누고, 구간마다 정책 1개를 뽑는다."""
    rng = np.random.default_rng(seed)
    order = (b1.assign(y=np.log10(b1.cycle_life)).groupby("policy").y.mean()
             .sort_values().index.to_list())
    picked = [str(rng.choice(chunk)) for chunk in np.array_split(np.array(order, dtype=object), n)]
    return sorted(picked)


def repeated_group_folds(dev: pd.DataFrame, n_folds: int = N_FOLDS, n_repeats: int = N_REPEATS,
                         seed: int = SEED) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """(repeat, fold, train_idx, valid_idx) 목록. 정책을 섞은 뒤 순서대로 fold에 배정한다."""
    rng = np.random.default_rng(seed)
    policies = np.array(sorted(dev.policy.unique()), dtype=object)
    splits = []
    for r in range(n_repeats):
        perm = rng.permutation(policies)
        fold_of = {p: i % n_folds for i, p in enumerate(perm)}
        fold = dev.policy.map(fold_of).to_numpy()
        for k in range(n_folds):
            splits.append((r, k, np.where(fold != k)[0], np.where(fold == k)[0]))
    return splits
