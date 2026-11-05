"""Outlier handling.

``HANDLING`` is the outlier handling applied by every analysis script, read from
the ``OUTLIER_HANDLING`` environment variable:

    none         (main analysis) no participant is removed and no value is clipped
    mahalanobis  (replication) participants whose participant means on the screened
                 columns exceed the chi-square cut-off (alpha = .001) are removed; no
                 values are clipped
    winsor95     no participant is removed; each trial-level outcome is clipped at
                 the 5th and 95th percentiles of its trial values and each
                 participant-level predictor at those of its participant values

Participants missing any screened column cannot be screened and are kept. The
multiverse analysis constructs its alternative conditions explicitly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


@dataclass(frozen=True)
class OutlierPolicy:
    method: str = 'cooks_distance'          # 'cooks_distance', 'mahalanobis' or 'none'
    alpha: float = 0.001                    # Mahalanobis chi-square cut-off
    cooks_threshold: str | float = '4/n'    # '4/n', '4/(n-k-1)' or a number
    min_complete_cases: int = 8


HANDLING = os.environ.get('OUTLIER_HANDLING', 'none')
if HANDLING not in ('none', 'mahalanobis', 'winsor95'):
    raise ValueError(f'Unsupported OUTLIER_HANDLING: {HANDLING}')
WINSOR_PCT = 5
MAHALANOBIS = OutlierPolicy(method='mahalanobis', alpha=0.001)


@dataclass(frozen=True)
class OutlierResult:
    method: str
    screen_columns: tuple[str, ...]
    subject_ids: tuple[int, ...]
    n_complete_cases: int
    threshold_label: str


def filter_subjects(df: pd.DataFrame, subject_ids: Sequence[int], subject_col: str = 'sub_id') -> pd.DataFrame:
    if not subject_ids:
        return df.copy()
    return df[~df[subject_col].isin(subject_ids)].copy()


def winsorize(df: pd.DataFrame, outcomes=(), preds=(), pct: float = WINSOR_PCT) -> pd.DataFrame:
    """Clip trial-level ``outcomes`` at the pct / (100 - pct) percentiles of their trial
    values and participant-level ``preds`` at those of their participant values."""
    df = df.copy()
    for col in outcomes:
        vals = df[col].dropna()
        if len(vals):
            df[col] = df[col].clip(np.percentile(vals, pct), np.percentile(vals, 100 - pct))
    sub = df.drop_duplicates('sub_id')
    for col in preds:
        vals = sub[col].dropna()
        if len(vals) >= 4:
            df[col] = df[col].clip(np.percentile(vals, pct), np.percentile(vals, 100 - pct))
    return df


def mahalanobis_ids(df: pd.DataFrame, screen_cols) -> set[int]:
    """Participants flagged by Mahalanobis distance on their means of ``screen_cols``."""
    means = df.groupby('sub_id')[list(screen_cols)].mean().reset_index()
    return set(detect_subject_outliers(means, screen_columns=screen_cols, policy=MAHALANOBIS).subject_ids)


def excluded_ids(df: pd.DataFrame, screen_cols) -> set[int]:
    """Participants removed under ``HANDLING``: the Mahalanobis set, otherwise none."""
    return mahalanobis_ids(df, screen_cols) if HANDLING == 'mahalanobis' else set()


def winsorize_active(df: pd.DataFrame, outcomes=(), preds=()) -> pd.DataFrame:
    """``winsorize`` under ``HANDLING = 'winsor95'``; the data unchanged otherwise."""
    return winsorize(df, outcomes, preds) if HANDLING == 'winsor95' else df.copy()


def handle_outliers(df: pd.DataFrame, outcomes, preds=(), screen=None) -> tuple[pd.DataFrame, set[int]]:
    """Apply ``HANDLING`` to one model's data.

    Mahalanobis screens the participant means of ``screen`` (default: the outcomes
    and predictors). Returns the data and the removed participants.
    """
    outcomes, preds = list(outcomes), list(preds)
    ids = excluded_ids(df, screen or [*outcomes, *preds])
    return winsorize_active(filter_subjects(df, ids), outcomes, preds), ids


def describe_handling() -> str:
    if HANDLING == 'none':
        return 'Outlier handling: none; every participant retained, no values clipped.'
    if HANDLING == 'winsor95':
        return ('Outlier handling: winsorised at the 5th and 95th percentiles '
                '(trial-level outcomes on trial values, predictors on participant values); no one removed.')
    return 'Outlier handling: participants removed by Mahalanobis distance (chi-square, alpha = .001).'


def detect_subject_outliers(
    df: pd.DataFrame,
    *,
    screen_columns: Sequence[str],
    subject_col: str = 'sub_id',
    outcome_col: str | None = None,
    predictor_columns: Sequence[str] | None = None,
    policy: OutlierPolicy | None = None,
) -> OutlierResult:
    """Screen one row per subject on ``screen_columns`` and return the flagged IDs."""
    policy = policy or MAHALANOBIS
    screen_columns = tuple(dict.fromkeys(screen_columns))
    work = df.loc[:, [subject_col, *screen_columns]].drop_duplicates(subset=[subject_col])
    complete = work.dropna(subset=list(screen_columns)).copy()

    if policy.method == 'none':
        return OutlierResult('none', screen_columns, tuple(), len(complete), 'disabled')

    min_cases = max(policy.min_complete_cases, len(screen_columns) + 1)
    if len(complete) < min_cases:
        return OutlierResult(policy.method, screen_columns, tuple(), len(complete), 'skipped')

    if policy.method == 'mahalanobis':
        return _mahalanobis(complete, subject_col, screen_columns, policy)

    if policy.method == 'cooks_distance':
        if not outcome_col:
            raise ValueError("Cook's distance requires outcome_col.")
        if predictor_columns is None:
            predictor_columns = [c for c in screen_columns if c != outcome_col]
        predictor_columns = tuple(dict.fromkeys(predictor_columns))
        if not predictor_columns:
            raise ValueError("Cook's distance requires at least one predictor column.")
        return _cooks_distance(complete, subject_col, outcome_col, predictor_columns, policy)

    raise ValueError(f'Unsupported outlier method: {policy.method}')


def _mahalanobis(complete, subject_col, screen_columns, policy) -> OutlierResult:
    values = complete.loc[:, screen_columns].to_numpy(dtype=float)
    cov = np.atleast_2d(np.cov(values, rowvar=False)).astype(float)
    cov += np.eye(cov.shape[0]) * 1e-8
    centred = values - values.mean(axis=0)
    d2 = np.einsum('ij,jk,ik->i', centred, np.linalg.pinv(cov), centred)
    threshold = stats.chi2.ppf(1 - policy.alpha, df=len(screen_columns))
    ids = tuple(sorted(complete.loc[d2 > threshold, subject_col].astype(int)))
    return OutlierResult('mahalanobis', tuple(screen_columns), ids, len(complete),
                         f'p<{policy.alpha:g}, df={len(screen_columns)}')


def _cooks_distance(complete, subject_col, outcome_col, predictor_columns, policy) -> OutlierResult:
    design = sm.add_constant(complete.loc[:, predictor_columns].astype(float), has_constant='add')
    fit = sm.OLS(complete.loc[:, outcome_col].astype(float), design).fit()
    cooks_d = fit.get_influence().cooks_distance[0]
    threshold = _cooks_threshold(policy.cooks_threshold, len(complete), len(predictor_columns))
    ids = tuple(sorted(complete.loc[cooks_d > threshold, subject_col].astype(int)))
    return OutlierResult('cooks_distance', (outcome_col, *predictor_columns), ids, len(complete),
                         f"Cook's D > {threshold:.3g}")


def _cooks_threshold(threshold: str | float, n_obs: int, n_predictors: int) -> float:
    if isinstance(threshold, (int, float)):
        return float(threshold)
    if threshold == '4/n':
        return 4.0 / max(n_obs, 1)
    if threshold == '4/(n-k-1)':
        return 4.0 / max(n_obs - n_predictors - 1, 1)
    raise ValueError(f"Unsupported Cook's distance threshold: {threshold}")
