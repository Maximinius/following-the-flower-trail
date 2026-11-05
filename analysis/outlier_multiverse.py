"""Outlier multiverse: the confirmatory families re-fitted under five outlier-handling conditions.

Conditions
    No removal   all participants retained, no values clipped (the main analysis)
    Mahalanobis  participants beyond the chi-square cut-off (alpha = .001) removed
    Cook's D     participants with Cook's distance above 4/n removed
    Winsor 99%   outcomes clipped at the 1st and 99th percentiles of their trial values and
                 predictors at those of their participant values
    Winsor 95%   as Winsor 99%, at the 5th and 95th percentiles

Families, each fitted with the same model as its main script (``modality_lmm``
and ``cognitive_lmm.confirmatory_models``) and FDR-corrected within family and
condition:
    the 11 confirmatory and the 4 exploratory modality contrasts (by-participant
    modality slope), with participants screened on their means of errors per minute
    and the four digital kinematic outcomes;
    the single-domain main effects of the maze, Flanker, gradCPT and WCST measures
    on errors per minute (H3 to H6, one model per measure), with participants
    screened on their means of errors per minute and the domain's measures.

Outputs: outlier_multiverse_output.md, multiverse_significance.csv,
multiverse_<family>.png (forest plot of the estimates across conditions),
multiverse_heatmap_<family>.png (FDR p by condition) and multiverse_matrix.png
(every effect by condition, tick for FDR p < .05, cross otherwise; Fig. S3).
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cognitive_lmm import OUTCOME, confirmatory_models
from common import RESULTS, boxcox_fit, heading, report, standardise_predictors, zscore
from load_tables import FTT_SCREEN, load_cognitive_battery, load_flower, load_maze_agg
from measure_names import CONTRAST_LABEL, label_for, qualified_label
from modality_lmm import (
    CONFIRMATORY, CONFIRMATORY_DIGITAL, EXPLORATORY, confirmatory_family, exploratory_family,
    split_samples, transform,
)
from outlier_policy import OutlierPolicy, detect_subject_outliers, filter_subjects, winsorize

# (label, method, winsor percentile)
CONDITIONS = [
    ('No removal', 'none', None),
    ('Mahalanobis', 'mahalanobis', None),
    ("Cook's D", 'cooks_distance', None),
    ('Winsor 99%', 'winsor', 1),
    ('Winsor 95%', 'winsor', 5),
]
COND_LABELS = [c[0] for c in CONDITIONS]
COND_COLORS = {'No removal': '#888888', 'Mahalanobis': '#1f77b4', "Cook's D": '#ff7f0e',
               'Winsor 99%': '#2ca02c', 'Winsor 95%': '#d62728'}
COND_MARKERS = {'No removal': 'o', 'Mahalanobis': 'o', "Cook's D": '^', 'Winsor 99%': 's', 'Winsor 95%': 's'}
MATRIX_COLUMNS = {'No removal': 'No removal (main)', 'Mahalanobis': 'Mahalanobis', "Cook's D": "Cook's distance",
                  'Winsor 99%': 'Winsorised 99%', 'Winsor 95%': 'Winsorised 95%'}

# (family key, display name, predictors, predictor loader)
DOMAINS = [
    ('Maze', 'Maze (H3)', ['maze_mean_time', 'maze_mean_mistakes'], load_maze_agg),
    ('Flanker', 'Flanker (H4)', ['flanker_congruency_effect', 'flanker_error_rate_incongruent'],
     load_cognitive_battery),
    ('gradCPT', 'gradCPT (H5)', ['gradcpt_rt_city_correct', 'gradcpt_rt_mountain_commission',
                                 'gradcpt_total_misses', 'gradcpt_mountain_withheld'], load_cognitive_battery),
    ('WCST', 'WCST (H6)', ['wcst_total_errors', 'wcst_perseverance_errors'], load_cognitive_battery),
]
FAMILY_NAMES = {'Confirmatory': 'Modality contrasts, confirmatory (H1, H2)',
                'Exploratory': 'Modality contrasts, exploratory',
                **{key: name for key, name, _, _ in DOMAINS}}


# ── Outlier conditions ────────────────────────────────────────────────────────

def removed_ids(df: pd.DataFrame, screen: list[str], method: str, outcome: str | None = None) -> set[int]:
    """Participants flagged on their means of ``screen`` (none for 'none' and 'winsor')."""
    if method not in ('mahalanobis', 'cooks_distance'):
        return set()
    policy = OutlierPolicy(method=method, alpha=0.001, cooks_threshold='4/n')
    means = df.groupby('sub_id')[screen].mean().reset_index()
    result = detect_subject_outliers(means, screen_columns=screen, policy=policy,
                                     outcome_col=outcome if method == 'cooks_distance' else None)
    return set(result.subject_ids)


def rows_from(table: pd.DataFrame, family: str, condition: str, effect) -> list[dict]:
    return [{'family': family, 'effect': effect(r), 'condition': condition, 'n': int(r['n_sub']),
             'beta': r['beta'], 'se': r['se'], 'p_uncorr': r['p'], 'p_fdr': r['p_fdr']}
            for _, r in table.iterrows()]


# ── Families ──────────────────────────────────────────────────────────────────

def modality_condition(flower: pd.DataFrame, condition) -> list[dict]:
    """Both modality families under one condition."""
    label, method, pct = condition
    removed = removed_ids(flower, FTT_SCREEN, method, OUTCOME)
    conf, exp = split_samples(filter_subjects(flower, removed))
    if method == 'winsor':
        conf = winsorize(conf, [c for c, _ in CONFIRMATORY], [], pct)
        exp = winsorize(exp, [c for c, _ in CONFIRMATORY_DIGITAL + EXPLORATORY], [], pct)
    conf, exp = transform(conf, exp, verbose=False)
    print(f'  {label:12} removed: {sorted(removed)}')
    effect = lambda r: f"{label_for(r['outcome'])} ({CONTRAST_LABEL[r['contrast']].lower()})"
    return (rows_from(confirmatory_family(conf, exp, verbose=False), 'Confirmatory', label, effect)
            + rows_from(exploratory_family(exp, verbose=False), 'Exploratory', label, effect))


def domain_condition(flower: pd.DataFrame, family: str, preds: list[str], loader, condition) -> list[dict]:
    """Single-domain main effects under one condition, as in ``cognitive_lmm``."""
    label, method, pct = condition
    df = flower.merge(loader()[['sub_id', *preds]], on='sub_id', how='inner')
    removed = removed_ids(df, [OUTCOME, *preds], method, OUTCOME)
    df = filter_subjects(df, removed)
    if method == 'winsor':
        df = winsorize(df, [OUTCOME], preds, pct)
    df[OUTCOME] = zscore(boxcox_fit(df[OUTCOME])[0])
    df = standardise_predictors(df, preds)
    print(f'  {label:12} removed: {sorted(removed)}')
    table = confirmatory_models(df, preds, verbose=False)
    return rows_from(table, family, label, lambda r: qualified_label(r['predictor']))


# ── Figures ───────────────────────────────────────────────────────────────────

def forest_figure(rows: pd.DataFrame, out_path) -> None:
    """Estimates with 95% CIs across conditions, one row per effect."""
    effects = list(dict.fromkeys(rows['effect']))
    offsets = np.linspace(-0.28, 0.28, len(CONDITIONS))
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.55 * len(effects) + 1.2)), constrained_layout=True)
    ax.axvline(0, color='#555', linewidth=0.9, linestyle='--', zorder=1)
    for ci, cond in enumerate(COND_LABELS):
        for yi, eff in enumerate(effects):
            r = rows[(rows['effect'] == eff) & (rows['condition'] == cond)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = yi + offsets[ci]
            ax.plot([r['beta'] - 1.96 * r['se'], r['beta'] + 1.96 * r['se']], [y, y],
                    color=COND_COLORS[cond], linewidth=1.5, zorder=2, solid_capstyle='round')
            ax.plot(r['beta'], y, marker=COND_MARKERS[cond], color=COND_COLORS[cond], markersize=6, zorder=3,
                    markerfacecolor=COND_COLORS[cond] if r['p_fdr'] < 0.05 else 'white')
    ax.set_yticks(range(len(effects)))
    ax.set_yticklabels(effects, fontsize=8)
    ax.set_ylim(len(effects) - 0.4, -0.6)
    ax.set_xlabel('Standardised β (95% CI); filled, FDR p < .05', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    handles = [mpatches.Patch(color=COND_COLORS[c], label=c) for c in COND_LABELS]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(CONDITIONS), fontsize=8, frameon=False)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def format_p(p: float) -> str:
    return '< .001' if p < 0.001 else f'{p:.3f}'.lstrip('0')


def heatmap_figure(rows: pd.DataFrame, out_path) -> None:
    """Significance grid: effects by condition, cells carry the FDR-corrected p."""
    effects = list(dict.fromkeys(rows['effect']))
    lookup = {(r['effect'], r['condition']): r['p_fdr'] for _, r in rows.iterrows()}
    fig, ax = plt.subplots(figsize=(1.35 * len(COND_LABELS) + 3.6, max(2.0, 0.42 * len(effects) + 0.9)))
    for xi, cond in enumerate(COND_LABELS):
        for yi, eff in enumerate(effects):
            p = lookup.get((eff, cond))
            fill = '#e8e8e8' if p is None else ('#1b6b3f' if p < 0.05 else '#f2c4c0')
            colour = 'white' if p is not None and p < 0.05 else '#3a3a3a'
            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=fill, edgecolor='white', linewidth=1.5))
            ax.text(xi + 0.5, yi + 0.5, '—' if p is None else format_p(p), ha='center', va='center',
                    fontsize=8, color=colour)
    ax.set_xlim(0, len(COND_LABELS))
    ax.set_ylim(len(effects), 0)
    ax.set_xticks([i + 0.5 for i in range(len(COND_LABELS))], COND_LABELS, fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(len(effects))], effects, fontsize=9)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def matrix_lines(rows: pd.DataFrame) -> list[tuple]:
    """(kind, family, effect) for every group header and effect row of the matrix."""
    lines = []
    for family in dict.fromkeys(rows['family']):
        lines.append(('group', family, None))
        lines += [('effect', family, e) for e in dict.fromkeys(rows.loc[rows['family'] == family, 'effect'])]
    return lines


def matrix_figure(rows: pd.DataFrame, out_path) -> None:
    """Every effect by condition in one grid: tick for FDR p < .05, cross otherwise. Black and white."""
    lookup = {(r['family'], r['effect'], r['condition']): r['p_fdr'] for _, r in rows.iterrows()}
    lines = matrix_lines(rows)
    row_h, col_w, label_w = 0.24, 1.1, 3.4
    fig_w, fig_h = label_w + col_w * len(COND_LABELS), row_h * (len(lines) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=label_w / fig_w, right=1, top=1 - 1.6 * row_h / fig_h, bottom=0)
    ax.set_xlim(0, len(COND_LABELS))
    ax.set_ylim(len(lines), 0)
    ax.axis('off')
    left = -label_w / col_w
    for xi, cond in enumerate(COND_LABELS):
        ax.text(xi + 0.5, -0.35, MATRIX_COLUMNS[cond], ha='center', va='bottom', fontsize=8)
    for yi, (kind, family, effect) in enumerate(lines):
        if kind == 'group':
            ax.text(left + 0.05, yi + 0.5, FAMILY_NAMES[family], ha='left', va='center', fontsize=8,
                    fontweight='bold')
            ax.plot([left, len(COND_LABELS)], [yi, yi], color='black', linewidth=0.8 if not yi else 0.4,
                    clip_on=False)
            continue
        ax.text(left + 0.2, yi + 0.5, effect, ha='left', va='center', fontsize=8)
        for xi, cond in enumerate(COND_LABELS):
            p = lookup.get((family, effect, cond))
            mark = '–' if p is None or pd.isna(p) else ('✓' if p < 0.05 else '✗')
            ax.text(xi + 0.5, yi + 0.5, mark, ha='center', va='center', fontsize=9, color='black')
    ax.plot([left, len(COND_LABELS)], [len(lines)] * 2, color='black', linewidth=0.8, clip_on=False)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────────

def summarise(table: pd.DataFrame) -> None:
    heading('Significance by family and condition (FDR p < .05)', 2)
    sig = table.assign(sig=table['p_fdr'] < 0.05)
    counts = sig.pivot_table(index='family', columns='condition', values='sig', aggfunc='sum', sort=False)
    print(counts[COND_LABELS].to_markdown())
    wide = sig.pivot_table(index=['family', 'effect'], columns='condition', values='sig', sort=False)
    flips = wide[wide.nunique(axis=1) > 1]
    print(f'\n{len(flips)} of {len(wide)} effects change significance across conditions.')
    if not flips.empty:
        print(flips[COND_LABELS].astype(bool).to_markdown())


def main() -> None:
    with report('outlier_multiverse_output.md'):
        heading('Outlier multiverse')
        print('Conditions: ' + ', '.join(COND_LABELS))
        flower = load_flower()
        rows = []
        heading('Modality contrasts', 2)
        for cond in CONDITIONS:
            rows += modality_condition(flower, cond)
        for family, name, preds, loader in DOMAINS:
            heading(name, 2)
            for cond in CONDITIONS:
                rows += domain_condition(flower, family, preds, loader, cond)
        table = pd.DataFrame(rows)
        table.to_csv(RESULTS / 'multiverse_significance.csv', index=False)
        for family in dict.fromkeys(table['family']):
            sub = table[table['family'] == family]
            forest_figure(sub, RESULTS / f'multiverse_{family}.png')
            heatmap_figure(sub, RESULTS / f'multiverse_heatmap_{family}.png')
        matrix_figure(table, RESULTS / 'multiverse_matrix.png')
        summarise(table)


if __name__ == '__main__':
    main()
