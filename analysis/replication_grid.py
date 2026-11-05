"""Replication grid: the main results (no outlier handling) against Mahalanobis exclusion.

Reads the summary tables of the main run (RESULTS) and of the Mahalanobis
replication (RESULTS/mahalanobis, written by ``run_pipeline.py`` with
OUTLIER_HANDLING=mahalanobis) and marks, for each effect, whether it is
significant under each handling: FDR p < .05, or BF10 > 3 for the Bayesian
modality comparisons.

Every effect is listed for the modality contrasts, the Bayesian comparisons, the
single-domain confirmatory predictors, the joint model and H7 (path deviation
as in H2). For the
single-domain interaction contrasts, the cognition x modality interactions on the
digital outcomes (modality-slope models) and the within-modality models, only
effects significant under at least one handling are listed.

Outputs: replication_grid_output.md, replication_grid.csv, replication_grid.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, code_block, heading, report
from measure_names import CONTRAST_LABEL, MODALITY_LABEL, label_for, qualified_label

REPLICATION = RESULTS / 'mahalanobis'
HANDLINGS = {'main': 'No exclusion (main)', 'mahalanobis': 'Mahalanobis'}
ALPHA = 0.05
BF_THRESHOLD = 3.0
DOMAINS = {'maze': 'Maze (H3)', 'flanker': 'Flanker (H4)', 'gradcpt': 'gradCPT (H5)', 'wcst': 'WCST (H6)'}


def _read(name: str) -> dict[str, pd.DataFrame]:
    return {'main': pd.read_csv(RESULTS / name), 'mahalanobis': pd.read_csv(REPLICATION / name)}


def _contrast(key: str) -> str:
    return CONTRAST_LABEL.get(key, CONTRAST_LABEL.get(key.title(), key.replace('_', ' '))).lower()


def _pair(tables: dict, keys: list[str], value: str, label, group: str, all_rows: bool,
          stat: str = 'p_fdr') -> list[dict]:
    """One row per effect with the value under each handling, keyed on ``keys``."""
    merged = tables['main'][keys + [value]].merge(tables['mahalanobis'][keys + [value]], on=keys,
                                                  how='outer', suffixes=('_main', '_mahalanobis'))
    rows = []
    for _, r in merged.iterrows():
        v = {h: r[f'{value}_{h}'] for h in HANDLINGS}
        sig = {h: (v[h] > BF_THRESHOLD if stat == 'BF10' else v[h] < ALPHA) if pd.notna(v[h]) else None
               for h in HANDLINGS}
        if all_rows or any(sig.values()):
            rows.append({'group': group, 'effect': label(r), 'stat': stat,
                         **{f'value_{h}': v[h] for h in HANDLINGS}, **{f'sig_{h}': sig[h] for h in HANDLINGS}})
    return rows


def collect() -> pd.DataFrame:
    rows = []
    mod = _read('modality_lmm_summary.csv')
    mod = {h: t[t['family'].isin(['confirmatory', 'exploratory'])] for h, t in mod.items()}
    rows += _pair(mod, ['outcome', 'contrast'], 'p_fdr',
                  lambda r: f"{label_for(r['outcome'])} ({_contrast(r['contrast'])})",
                  'Modality contrasts (H1, H2)', True)
    rows += _pair(_read('bayesian_modality_summary.csv'), ['A', 'B'], 'BF10',
                  lambda r: f"Errors per minute ({r['B']} vs. {r['A']})",
                  'Bayesian modality comparisons', True, stat='BF10')

    for name, title in DOMAINS.items():
        t = _read(f'{name}_lmm_summary.csv')
        conf = {h: d[d['block'] == 'confirmatory'] for h, d in t.items()}
        rows += _pair(conf, ['predictor'], 'p_fdr', lambda r: qualified_label(r['predictor']), title, True)
        inter = {h: d[d['block'] == 'interaction'] for h, d in t.items()}
        rows += _pair(inter, ['predictor', 'contrast'], 'p_fdr',
                      lambda r: f"{qualified_label(r['predictor'])} x modality ({_contrast(r['contrast'])})",
                      title, False)

    rows += _pair(_read('joint_lmm_summary.csv'), ['predictor'], 'p_fdr',
                  lambda r: qualified_label(r['predictor']), 'Joint model', True)
    h7 = {h: t[t['measure'] == 'line_rule'] for h, t in _read('spatial_difficulty_summary.csv').items()}
    rows += _pair(h7, ['term'], 'p', lambda r: r['term'],
                  'Path deviation by segment complexity (H7)', True, stat='p')

    ex = _read('joint_lmm_exploratory_interactions.csv')
    ex = {h: t[t['spec'] == 'slope'] for h, t in ex.items()}
    rows += _pair(ex, ['family', 'outcome', 'predictor'], 'p_fdr',
                  lambda r: f"{label_for(r['outcome'])}: {qualified_label(r['predictor'])} x modality",
                  'Cognition x modality, digital outcomes', False)
    rows += _pair(_read('standalone_modality_summary.csv'), ['block', 'modality', 'outcome', 'predictor'], 'p_bh',
                  lambda r: f"{MODALITY_LABEL[r['modality']]}: {label_for(r['outcome'])} ~ "
                            f"{qualified_label(r['predictor'])}",
                  'Within-modality models', False)
    return pd.DataFrame(rows)


def grid_figure(table: pd.DataFrame, out_path) -> None:
    """Every effect by handling: tick for significant, cross otherwise. Black and white."""
    lines = []
    for group in dict.fromkeys(table['group']):
        lines.append(('group', group, None))
        lines += [('effect', group, r) for _, r in table[table['group'] == group].iterrows()]

    row_h, col_w, label_w = 0.24, 1.5, 4.6
    cols = list(HANDLINGS)
    fig_w, fig_h = label_w + col_w * len(cols), row_h * (len(lines) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=label_w / fig_w, right=1, top=1 - 1.6 * row_h / fig_h, bottom=1.6 * row_h / fig_h)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(lines), 0)
    ax.axis('off')
    left = -label_w / col_w

    for xi, h in enumerate(cols):
        ax.text(xi + 0.5, -0.35, HANDLINGS[h], ha='center', va='bottom', fontsize=8)
    ax.plot([left, len(cols)], [0, 0], color='black', linewidth=0.8, clip_on=False)
    for yi, (kind, group, r) in enumerate(lines):
        y = yi + 0.5
        if kind == 'group':
            ax.text(left + 0.03, y, group, ha='left', va='center', fontsize=8, fontweight='bold')
            if yi:
                ax.plot([left, len(cols)], [yi, yi], color='black', linewidth=0.4, clip_on=False)
            continue
        ax.text(left + 0.12, y, r['effect'], ha='left', va='center', fontsize=7.5)
        for xi, h in enumerate(cols):
            sig = r[f'sig_{h}']
            mark = '–' if sig is None or (isinstance(sig, float) and np.isnan(sig)) else ('✓' if sig else '✗')
            ax.text(xi + 0.5, y, mark, ha='center', va='center', fontsize=9, color='black')
    ax.plot([left, len(cols)], [len(lines)] * 2, color='black', linewidth=0.8, clip_on=False)
    ax.text(left + 0.03, len(lines) + 0.9,
            f'✓ FDR p < .05 (Bayesian comparisons: BF10 > {BF_THRESHOLD:g});  ✗ not significant;  – not tested',
            ha='left', va='center', fontsize=7)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main() -> None:
    with report('replication_grid_output.md'):
        heading('Replication of the main results under Mahalanobis exclusion')
        table = collect()
        table.to_csv(RESULTS / 'replication_grid.csv', index=False)
        grid_figure(table, RESULTS / 'replication_grid.png')

        sig_main = table['sig_main'].eq(True)
        replicated = sig_main & table['sig_mahalanobis'].eq(True)
        counts = table.assign(sig_main=sig_main, replicated=replicated).groupby('group', sort=False).agg(
            effects=('effect', 'size'), significant_main=('sig_main', 'sum'), replicated=('replicated', 'sum'))
        print('Effects per group, significant in the main analysis, and of those significant under Mahalanobis')
        code_block(counts.reset_index())
        print(f'\n{int(replicated.sum())} of {int(sig_main.sum())} effects significant in the main analysis '
              'are also significant under Mahalanobis exclusion.')
        changed = table[sig_main != table['sig_mahalanobis'].eq(True)]
        if not changed.empty:
            print('\nEffects whose significance differs between the two handlings')
            code_block(changed[['group', 'effect', 'stat', 'value_main', 'value_mahalanobis']]
                       .to_string(index=False, float_format=lambda x: f'{x:.4g}'))


if __name__ == '__main__':
    main()
