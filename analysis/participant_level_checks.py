"""Participant-level checks behind the equivalence and kinematics claims.

No mixed models. Every statistic is computed on raw participant means.
Under the Mahalanobis replication (``outlier_policy.HANDLING``) the participants
flagged on errors per minute and the four digital kinematic outcomes are removed:

1. descriptives and coefficients of variation per modality;
2. split-half reliability within each modality (odd versus even trials,
   Spearman correlation stepped up by Spearman-Brown);
3. cross-modality rank-order agreement (Spearman r_s and its P value).
   P values are Benjamini-Hochberg corrected across the nine correlations;
4. power against the pre-registered analysis, which planned N = 53 to detect a
   paper-digital correlation of r_s = .38 (Baykara et al., 2022; two-tailed
   alpha = .05, power = .80, Fisher z). For each cross-modality correlation:
   power at the available n to detect r_s = .38. Power at the observed
   correlation is not computed. It is a function of the P value and adds
   nothing to it.
   For each pairwise modality contrast: the smallest paired effect (d_z)
   detectable at 80% power with the complete pairs, against the observed d_z;
5. the speed-accuracy relation inside each version (completion time versus
   boundary errors), Benjamini-Hochberg corrected across the three versions;
6. redundancy among the eight digital outcomes (Spearman matrix per device);
7. error length against errors per minute, raw, log and rank;
8. maze completion time against maze navigation errors.

Writes participant_level_checks_output.md, participant_level_checks_summary.csv,
power_check.csv and cross_modality_rank_order.png into RESULTS.
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from common import RESULTS, code_block, fdr, heading, report
from load_tables import FTT_SCREEN, MODALITIES, build_predictors, load_flower
from outlier_policy import excluded_ids, filter_subjects

OUTCOMES = {'mistakes_per_min': 'Errors per minute', 'completion_time': 'Completion time',
            'error_count': 'Boundary errors'}
DIGITAL = {'completion_time': 'Completion time', 'error_count': 'Boundary errors',
           'ms_mistakes': 'Error length', 'deviation': 'Path deviation',
           'speed': 'Movement speed', 'acceleration': 'Acceleration',
           'stops': 'Number of stops', 'ms_stopped': 'Time stopped'}
PAIRS = [('paper', 'tablet'), ('paper', 'mouse'), ('tablet', 'mouse')]
PREREG_R, PREREG_N, ALPHA, POWER = 0.38, 53, 0.05, 0.80
Z_A, Z_B = stats.norm.ppf(1 - ALPHA / 2), stats.norm.ppf(POWER)


def split_half(flower: pd.DataFrame, col: str, mod: str) -> float:
    sub = flower[(flower['modality'] == mod) & flower[col].notna()]
    half = np.where(sub['trial'] % 2 == 1, 'odd', 'even')
    wide = sub.assign(half=half).groupby(['sub_id', 'half'])[col].mean().unstack().dropna()
    r = stats.spearmanr(wide['odd'], wide['even'])[0]
    return 2 * r / (1 + r)


def descriptives(means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in OUTCOMES.items():
        for mod in MODALITIES:
            v = means[(col, mod)].dropna()
            rows.append(dict(outcome=label, modality=mod, n=len(v), mean=v.mean(),
                             median=v.median(), sd=v.std(), cv=v.std() / v.mean()))
    return pd.DataFrame(rows)


def cross_modal(flower: pd.DataFrame, means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in OUTCOMES.items():
        rel = {mod: split_half(flower, col, mod) for mod in MODALITIES}
        for a, b in PAIRS:
            pair = means[[(col, a), (col, b)]].dropna()
            rho, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append(dict(outcome=label, pair=f'{a}-{b}', n=len(pair), rho=rho, p_rho=p,
                             rel_a=rel[a], rel_b=rel[b]))
    out = pd.DataFrame(rows)
    out['p_fdr'] = fdr(out['p_rho'])
    return out


def n_for_r(r: float) -> int:
    """N needed to detect correlation r (two-tailed, Fisher z)."""
    return int(np.ceil(((Z_A + Z_B) / np.arctanh(abs(r))) ** 2 + 3))


def power_for_r(r: float, n: int) -> float:
    return float(stats.norm.cdf(np.arctanh(abs(r)) * np.sqrt(n - 3) - Z_A))


def power_correlations(cm: pd.DataFrame) -> pd.DataFrame:
    out = cm[['outcome', 'pair', 'n', 'rho', 'p_fdr']].copy()
    out['power_prereg_r'] = [power_for_r(PREREG_R, n) for n in out['n']]
    return out


def min_dz(n: int) -> float:
    """Smallest paired d_z detectable at 80% power, two-tailed (noncentral t)."""
    df, crit = n - 1, stats.t.ppf(1 - ALPHA / 2, n - 1)
    lo, hi = 0.01, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        pw = stats.nct.sf(crit, df, mid * np.sqrt(n))
        lo, hi = (lo, mid) if pw >= POWER else (mid, hi)
    return hi


def power_contrasts(means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in OUTCOMES.items():
        for a, b in PAIRS:
            pair = means[[(col, a), (col, b)]].dropna()
            diff = pair.iloc[:, 1] - pair.iloc[:, 0]
            rows.append(dict(outcome=label, pair=f'{a}-{b}', n=len(pair),
                             dz_observed=diff.mean() / diff.std(), min_dz_80=min_dz(len(pair))))
    return pd.DataFrame(rows)


def speed_accuracy(means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mod in MODALITIES:
        pair = means[[('completion_time', mod), ('error_count', mod)]].dropna()
        rho, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append(dict(modality=mod, n=len(pair), rho=rho, p=p))
    out = pd.DataFrame(rows)
    out['p_fdr'] = fdr(out['p'])
    return out


def digital_redundancy(flower: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for mod in ['tablet', 'mouse']:
        m = flower[flower['modality'] == mod].groupby('sub_id')[list(DIGITAL)].mean()
        out[mod] = m.rename(columns=DIGITAL).corr(method='spearman')
    return out


def error_length_vs_epm(flower: pd.DataFrame) -> pd.DataFrame:
    dig = flower[flower['modality'].isin(['tablet', 'mouse'])]
    cols = ['mistakes_per_min', 'ms_mistakes']
    means = dig.groupby(['sub_id', 'modality'], observed=True)[cols].mean().reset_index()
    rows = []
    for level, d in [('trials', dig[cols].dropna()), ('participant-by-modality means', means[cols].dropna())]:
        x, y = d['mistakes_per_min'], d['ms_mistakes']
        rows.append(dict(level=level, n=len(d), pearson_raw=stats.pearsonr(x, y)[0],
                         pearson_log=stats.pearsonr(np.log1p(x), np.log1p(y))[0],
                         spearman=stats.spearmanr(x, y)[0]))
    return pd.DataFrame(rows)


def maze_tradeoff(flower: pd.DataFrame) -> pd.DataFrame:
    preds = build_predictors(verbose=False)
    maze = preds[preds['sub_id'].isin(flower['sub_id'])][['maze_mean_time', 'maze_mean_mistakes']].dropna()
    r, p = stats.pearsonr(maze['maze_mean_time'], maze['maze_mean_mistakes'])
    rho, p_rho = stats.spearmanr(maze['maze_mean_time'], maze['maze_mean_mistakes'])
    return pd.DataFrame([dict(n=len(maze), pearson=r, p=p, spearman=rho, p_spearman=p_rho)])


def plot_rank_order(means: pd.DataFrame, cm: pd.DataFrame) -> None:
    sns.set_theme(style='ticks', context='paper')
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)
    for ax, letter, (a, b) in zip(axes, 'abc', PAIRS):
        pair = means[[('mistakes_per_min', a), ('mistakes_per_min', b)]].dropna()
        ranks = pair.rank().to_numpy()
        sns.scatterplot(x=ranks[:, 0], y=ranks[:, 1], ax=ax, color='0.25', s=18, linewidth=0)
        row = cm[(cm['outcome'] == 'Errors per minute') & (cm['pair'] == f'{a}-{b}')].iloc[0]
        p_txt = '< 0.001' if row['p_fdr'] < 0.001 else f"= {row['p_fdr']:.3f}"
        ax.set_title(f"$r_s$ = {row['rho']:.2f}, $P$ {p_txt}", fontsize=8)
        ax.text(-0.18, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Rank on {a} (errors per minute)')
        ax.set_ylabel(f'Rank on {b} (errors per minute)')
        ax.plot([1, len(pair)], [1, len(pair)], color='0.6', lw=0.8, ls='--')
    sns.despine(fig)
    fig.savefig(RESULTS / 'cross_modality_rank_order.png', dpi=300)
    plt.close(fig)


def agreement_and_power(flower: pd.DataFrame, means: pd.DataFrame, fmt: dict) -> pd.DataFrame:
    """Cross-version correlations, power against the pre-registration and contrast sensitivity."""
    heading('Cross-modality rank order (Spearman, BH-FDR across the nine), with split-half reliability', 2)
    print('rho: Spearman correlation r_s. p_rho: raw P. p_fdr: Benjamini-Hochberg corrected P.')
    cm = cross_modal(flower, means)
    code_block(cm.to_string(**fmt))
    heading(f'Power against the pre-registered analysis (r = {PREREG_R}, N = {PREREG_N}, '
            f'alpha = {ALPHA}, power = {POWER})', 2)
    print(f'Check: N needed for r = {PREREG_R} is {n_for_r(PREREG_R)}.')
    print('power_prereg_r: power at this n to detect the pre-registered r_s.')
    pc = power_correlations(cm)
    code_block(pc.to_string(**fmt))
    heading('Sensitivity of the pairwise modality contrasts (participant means, complete pairs)', 2)
    print('dz_observed: mean difference (second minus first) / SD of differences. '
          'min_dz_80: smallest |d_z| detectable at 80% power with this n.')
    pt = power_contrasts(means)
    code_block(pt.to_string(**fmt))
    pd.concat([pc.assign(kind='correlation'), pt.assign(kind='contrast')]).to_csv(
        RESULTS / 'power_check.csv', index=False)
    return cm


def main() -> None:
    with report('participant_level_checks_output.md'):
        heading('Participant-level checks: rank order, redundancy, speed-accuracy')
        flower = load_flower()
        flower = filter_subjects(flower, excluded_ids(flower, FTT_SCREEN))
        print(f"Analysis sample: {flower['sub_id'].nunique()} participants, {len(flower)} trials.")
        means = flower.groupby(['sub_id', 'modality'], observed=True)[list(OUTCOMES)].mean().unstack()
        fmt = dict(index=False, float_format=lambda x: f'{x:.3g}' if abs(x) < 0.001 and x else f'{x:.3f}')
        heading('Descriptives and coefficient of variation (participant means)', 2)
        code_block(descriptives(means).to_string(**fmt))
        cm = agreement_and_power(flower, means, fmt)
        heading('Speed-accuracy within each version: completion time vs boundary errors (BH-FDR)', 2)
        code_block(speed_accuracy(means).to_string(**fmt))
        for mod, mat in digital_redundancy(flower).items():
            heading(f'Spearman correlations among the eight digital outcomes: {mod}', 2)
            code_block(mat.round(2).to_string())
        heading('Error length against errors per minute', 2)
        code_block(error_length_vs_epm(flower).to_string(**fmt))
        heading('Maze completion time against maze navigation errors', 2)
        code_block(maze_tradeoff(flower).to_string(**fmt))
        cm.to_csv(RESULTS / 'participant_level_checks_summary.csv', index=False)
        plot_rank_order(means, cm)


if __name__ == '__main__':
    main()
