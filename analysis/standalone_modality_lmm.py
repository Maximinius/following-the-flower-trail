"""Cognitive prediction of the Flower Trail outcomes within each modality.

Each model is fitted on one modality alone, with the data subset as though the
other conditions had not been collected:

    outcome ~ predictor + trial + (1 | participant)

The seven Flower Trail outcomes are fitted for the tablet and the mouse, and
errors per minute for paper as well. The two components of the primary
outcome, completion time and the boundary-error count, are fitted for all three
modalities and reported separately. FDR correction is applied within cognitive
domain and modality across the outcomes of that block. The change in marginal
R^2 (Nakagawa) from adding the predictor is reported alongside each slope.

Outliers follow ``outlier_policy.HANDLING``, separately for each cognitive
domain: the main analysis uses every participant and every value; in the
replication participants are screened by Mahalanobis distance
on their mean errors per minute and the domain's predictors. Outcomes are then
Box-Cox transformed and z-scored on the pooled data of the modalities in which
they exist, so that slopes are on one scale across modalities; predictors are
z-scored on their subject-level values.

Outputs: standalone_modality_output.md, standalone_modality_summary.csv,
         standalone_modality_forest.png (Figure 4),
         standalone_components_forest.png (Figure 3)
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import (RESULTS, boxcox_fit, code_block, fdr, heading, lmer_fit, report, stars,
                    standardise_predictors)
from load_tables import DIGITAL, MODALITIES, build_predictors, load_flower
from measure_names import MODALITY_LABEL, label_for, qualified_label
from outlier_policy import describe_handling, handle_outliers

warnings.filterwarnings('ignore')

ALPHA = 0.05

# (outcome, recorded on paper)
OUTCOMES = [
    ('mistakes_per_min', True),
    ('ms_mistakes', False),
    ('deviation', False),
    ('speed', False),
    ('acceleration', False),
    ('stops', False),
    ('ms_stopped', False),
]
COMPONENTS = [
    ('completion_time', True),
    ('error_count', True),
]

DOMAINS = {
    'Maze': ['maze_mean_time', 'maze_mean_mistakes'],
    'Flanker': ['flanker_congruency_effect', 'flanker_error_rate_incongruent'],
    'gradCPT': ['gradcpt_rt_city_correct', 'gradcpt_rt_mountain_commission',
                'gradcpt_total_misses', 'gradcpt_mountain_withheld'],
    'WCST': ['wcst_total_errors', 'wcst_perseverance_errors'],
}
ALL_PREDS = [c for cols in DOMAINS.values() for c in cols]
PRETTY = {c: qualified_label(c) for c in ALL_PREDS}

NS_COLOR = '#9aa5b1'
MOD_COLOR = {'paper': '#4c72b0', 'tablet': '#dd8452', 'mouse': '#55a868'}


def fit_standalone(data: pd.DataFrame, outcome: str, pred: str) -> dict | None:
    d = data[['sub_id', outcome, pred, 'trial']].dropna().copy()
    if d['sub_id'].nunique() < 10:
        return None
    full = lmer_fit(f'{outcome} ~ {pred} + trial', d)
    reduced = lmer_fit(f'{outcome} ~ trial', d)
    ci = full.conf_int().loc[pred]
    return {
        'outcome': outcome, 'predictor': pred,
        'n_sub': int(d['sub_id'].nunique()), 'n_rows': int(len(d)),
        'beta': float(full.fe_params[pred]), 'se': float(full.bse[pred]),
        'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]), 'p_raw': float(full.pvalues[pred]),
        't': float(full.tvalues[pred]), 'df': float(full.df[pred]),
        'dR2m': full.r2_marginal - reduced.r2_marginal,
    }


def prepare(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Box-Cox and z-score each outcome within the modalities in which it exists."""
    print(f'{domain} sample: outcome transforms (Box-Cox on the pooled data, then z-scored)')
    lines = []
    for col, on_paper in OUTCOMES + COMPONENTS:
        scope = MODALITIES if on_paper else DIGITAL
        mask = df['modality'].isin(scope)
        transformed, lmbda, shift = boxcox_fit(df.loc[mask, col])
        df.loc[mask, col] = (transformed - transformed.mean()) / transformed.std()
        df.loc[~mask, col] = np.nan
        lines.append(f"{label_for(col):<22} lambda = {lmbda:7.3f}   shift = {shift:8.3f}   scope = {'+'.join(scope)}")
    code_block('\n'.join(lines))
    return standardise_predictors(df, ALL_PREDS)


def run_grid(frames: dict[str, pd.DataFrame], outcome_spec, title: str) -> pd.DataFrame:
    """Every (modality, domain, outcome, predictor) model; FDR within modality x domain."""
    heading(title, 2)
    blocks = []
    for mod in MODALITIES:
        cols = [c for c, on_paper in outcome_spec if on_paper or mod != 'paper']
        if not cols:
            continue
        print(f'\n### {MODALITY_LABEL[mod]}\n')
        for domain, plist in DOMAINS.items():
            dmod = frames[domain][frames[domain]['modality'] == mod]
            rows = [r for col in cols for pred in plist if (r := fit_standalone(dmod, col, pred))]
            if not rows:
                continue
            res = pd.DataFrame(rows)
            res['p_bh'] = fdr(res['p_raw'])
            res['sig'] = res['p_bh'].map(stars)
            res['modality'] = mod
            res['domain'] = domain
            blocks.append(res)
            view = res.assign(outcome=res['outcome'].map(label_for), predictor=res['predictor'].map(PRETTY))
            print(f"{domain} ({dmod['sub_id'].nunique()} participants): FDR across {len(res)} tests "
                  f"({len(plist)} predictors x {len(cols)} outcomes)")
            code_block(view[['outcome', 'predictor', 'n_sub', 'beta', 'se', 'ci_lo', 'ci_hi',
                             't', 'df', 'p_raw', 'p_bh', 'sig', 'dR2m']]
                       .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return pd.concat(blocks, ignore_index=True)


def summarise(rows: pd.DataFrame, title: str) -> None:
    heading(title, 2)
    r = rows.assign(cell=rows.apply(lambda x: f"{x['beta']:.2f}{'*' if x['p_bh'] < ALPHA else ''}", axis=1),
                    outcome=rows['outcome'].map(label_for), predictor=rows['predictor'].map(PRETTY))
    for mod in MODALITIES:
        sub = r[r['modality'] == mod]
        if sub.empty:
            continue
        print(f'\n{MODALITY_LABEL[mod]}: standardised slope (asterisk = FDR-significant)')
        code_block(sub.pivot(index='predictor', columns='outcome', values='cell').to_string())
    print(f"\n{int((rows['p_bh'] < ALPHA).sum())} of {len(rows)} tests survived FDR correction.")
    for mod in MODALITIES:
        sub = rows[rows['modality'] == mod]
        k = int((sub['p_bh'] < ALPHA).sum())
        med = sub.loc[sub['p_bh'] < ALPHA, 'dR2m'].median() if k else np.nan
        print(f'  {MODALITY_LABEL[mod]}: {k}/{len(sub)} significant'
              + (f', median delta R2m of those = {med:.3f}' if k else ''))


def forest_grid(rows: pd.DataFrame, outcome_cols, mods, fname, figsize) -> None:
    """One forest panel per outcome (rows) and modality (columns)."""
    ys = np.arange(len(ALL_PREDS))[::-1]
    lo, hi = rows['ci_lo'].min(), rows['ci_hi'].max()
    pad = 0.08 * (hi - lo)
    xlim = (lo - pad, hi + pad)

    n_rows, n_cols = len(outcome_cols), len(mods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    k = 0
    for i, col in enumerate(outcome_cols):
        for j, mod in enumerate(mods):
            ax = axes[i][j]
            ax.axvline(0, color='#444444', lw=0.8, ls='--', zorder=1)
            sub = rows[(rows['outcome'] == col) & (rows['modality'] == mod)].set_index('predictor')
            for y, pred in zip(ys, ALL_PREDS):
                if pred not in sub.index:
                    continue
                r = sub.loc[pred]
                is_sig = r['p_bh'] < ALPHA
                c = MOD_COLOR[mod] if is_sig else NS_COLOR
                ax.plot([r['ci_lo'], r['ci_hi']], [y, y], color=c, lw=1.4, solid_capstyle='round', zorder=2)
                ax.plot([r['beta']], [y], 'o', ms=3.4, zorder=3, color=c if is_sig else 'white',
                        markeredgecolor=c, markeredgewidth=1.0)
            ax.set_xlim(xlim)
            ax.set_ylim(-0.7, len(ALL_PREDS) - 0.3)
            ax.text(0.012, 0.97, chr(97 + k), transform=ax.transAxes, fontsize=8,
                    fontweight='bold', va='top', ha='left')
            k += 1
            if i == 0:
                ax.set_title(MODALITY_LABEL[mod], fontsize=9, fontweight='bold', pad=6)
            if j == n_cols - 1:
                ax.text(1.02, 0.5, label_for(col), transform=ax.transAxes, fontsize=8,
                        rotation=270, va='center', ha='left')
            ax.tick_params(labelsize=6.5)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', color='#ececec', lw=0.5, zorder=0)
            ax.set_axisbelow(True)

    for j in range(n_cols):
        axes[-1][j].set_xlabel('Standardised β (95% CI)', fontsize=8)
    for i in range(n_rows):
        axes[i][0].set_yticks(ys)
        axes[i][0].set_yticklabels([PRETTY[p] for p in ALL_PREDS], fontsize=6.5)

    handles = [Line2D([], [], color=NS_COLOR, marker='o', lw=1.4, ms=4, markerfacecolor='white',
                      markeredgecolor=NS_COLOR, label='Not significant')]
    handles += [Line2D([], [], color=MOD_COLOR[m], marker='o', lw=1.4, ms=4,
                       label=f'{MODALITY_LABEL[m]}, FDR-significant') for m in mods]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.028, 0.975, 1])
    fig.savefig(RESULTS / fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main() -> None:
    with report('standalone_modality_output.md'):
        heading('Cognitive prediction of the Flower Trail outcomes within each modality')

        flower = load_flower()
        preds = build_predictors()
        df = flower.merge(preds[['sub_id', *ALL_PREDS]], on='sub_id', how='inner')
        print(describe_handling())
        print(f"Pooled sample: {df['sub_id'].nunique()} participants, {len(df)} trials.\n")
        outcome_cols = [c for c, _ in OUTCOMES + COMPONENTS]
        frames = {}
        for domain, plist in DOMAINS.items():
            frame, removed = handle_outliers(df, outcome_cols, plist, screen=['mistakes_per_min', *plist])
            if removed:
                print(f'{domain}: removed ({len(removed)}): {sorted(removed)}')
            frames[domain] = prepare(frame, domain)

        main_rows = run_grid(frames, OUTCOMES, 'Seven Flower Trail outcomes')
        comp_rows = run_grid(frames, COMPONENTS, 'Components of the primary outcome')
        summarise(main_rows, 'Summary: seven Flower Trail outcomes')
        summarise(comp_rows, 'Summary: completion time and boundary errors')

        pd.concat([main_rows.assign(block='primary'), comp_rows.assign(block='component')],
                  ignore_index=True).to_csv(RESULTS / 'standalone_modality_summary.csv', index=False)

        forest_grid(main_rows[main_rows['modality'].isin(DIGITAL)], [c for c, _ in OUTCOMES], DIGITAL,
                    'standalone_modality_forest.png', (7.0, 11.0))
        forest_grid(comp_rows, [c for c, _ in COMPONENTS], MODALITIES,
                    'standalone_components_forest.png', (8.4, 4.6))


if __name__ == '__main__':
    main()
