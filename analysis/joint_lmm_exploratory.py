"""Cognition-by-modality interactions on the digital Flower Trail outcomes.

For each outcome recorded in the two digital modalities, one model enters a
family of predictors together with their interaction with modality (mouse
versus tablet). There are two families: the seven Flanker, WCST and gradCPT
predictors, and the two maze predictors (completion time, navigation errors):

    outcome ~ predictors + modality + predictors x modality + trial + (1 + modality | participant)

The predictors vary between participants and modality within them, so the
by-participant modality slope supplies the error term for the interactions, as
in a repeated-measures ANOVA. The interaction coefficients are FDR-corrected
within outcome. A participant-level difference-score regression (each
participant's mean mouse-minus-tablet difference regressed by OLS on the same
predictors), which needs no random-effects structure, is the sensitivity analysis. Every model is fitted on
the participants with all predictors of the family observed. Outliers follow
``outlier_policy.HANDLING`` (no handling in the main analysis; a Mahalanobis
screen on the participant means of the outcome and the predictors in the
replication). Outcomes are then Box-Cox transformed where skewed and z-scored;
predictors are z-scored on their subject-level values. FDR runs within outcome
and family.

Outputs: joint_lmm_exploratory_output.md,
         joint_lmm_exploratory_interactions.csv (column ``spec``: slope, difference_score),
         kinematics_interaction_panels.png (slope models)
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (
    MOD_TABLET, RESULTS, boxcox_apply, boxcox_fit, code_block, fdr, heading, lmer_fit, print_fit,
    prune_vif, report, standardise_predictors, stars, zscore,
)
from load_tables import load_cognitive_battery, load_flower, load_maze_agg
from measure_names import TOKEN, clean_name, label_for, qualified_label
from outlier_policy import describe_handling, handle_outliers

PREDICTORS = [
    'flanker_congruency_effect', 'flanker_error_rate_incongruent',
    'wcst_total_errors', 'wcst_perseverance_errors',
    'gradcpt_rt_city_correct', 'gradcpt_total_misses', 'gradcpt_mountain_withheld',
]
MAZE_PREDICTORS = ['maze_mean_time', 'maze_mean_mistakes']
FIGURE_PREDICTORS = PREDICTORS + MAZE_PREDICTORS

# (outcome column, Box-Cox before modelling), in reporting order.
OUTCOMES = [
    ('completion_time', True),
    ('error_count', True),
    ('ms_mistakes', True),
    ('deviation', True),
    ('speed', False),
    ('acceleration', False),
    ('stops', True),
    ('ms_stopped', True),
]
FIGURE_OUTCOMES = ['completion_time', 'error_count', 'ms_mistakes', 'deviation', 'speed', 'acceleration']

SIG_COLOR = '#1f4e79'
NS_COLOR = '#9aa5b1'


def prepare(df: pd.DataFrame, outcome: str, preds: list[str], lmbda, shift) -> pd.DataFrame:
    """Transform and z-score the outcome and keep the model columns (not yet complete cases)."""
    data = df.copy()
    if lmbda is not None:
        data[outcome] = boxcox_apply(data[outcome], lmbda, shift)
    data[outcome] = zscore(data[outcome])
    return data[['sub_id', outcome, *preds, 'modality', 'trial']].dropna(subset=[outcome])


def fit_interaction_model(data: pd.DataFrame, outcome: str, preds: list[str]):
    """Fit the interaction model; return (fit, interaction table, data)."""
    data = standardise_predictors(data, preds).dropna().copy()
    if data['sub_id'].nunique() < len(preds) + 5:
        return None, None, data
    interactions = ' + '.join(f'{p}:{MOD_TABLET}' for p in preds)
    formula = f"{outcome} ~ {' + '.join(preds)} + {MOD_TABLET} + {interactions} + trial"
    fit = lmer_fit(formula, data, slope=MOD_TABLET)

    rows = []
    for term in fit.fe_params.index:
        if ':' not in term:
            continue
        ci = fit.conf_int().loc[term]
        rows.append({'outcome': outcome, 'predictor': term.split(':')[0],
                     'beta': fit.fe_params[term], 'se': fit.bse[term],
                     't': fit.tvalues[term], 'df': fit.df[term], 'p': fit.pvalues[term],
                     'ci_lo': ci[0], 'ci_hi': ci[1]})
    table = pd.DataFrame(rows)
    table['p_fdr'] = fdr(table['p'])
    table['n_sub'] = data['sub_id'].nunique()
    table['n_obs'] = len(data)
    table['spec'] = 'slope'
    return fit, table, data


def difference_score_model(data: pd.DataFrame, outcome: str, preds: list[str]) -> pd.DataFrame:
    """OLS of each participant's mean mouse-minus-tablet difference on the predictors."""
    data = standardise_predictors(data, preds).dropna()
    means = data.groupby(['sub_id', 'modality'], observed=True)[outcome].mean().unstack()
    diff = (means['mouse'] - means['tablet']).dropna()
    X = sm.add_constant(data.groupby('sub_id')[preds].first().loc[diff.index])
    fit = sm.OLS(diff, X).fit()
    ci = fit.conf_int().loc[preds]
    table = pd.DataFrame({'outcome': outcome, 'predictor': preds,
                          'beta': fit.params[preds].values, 'se': fit.bse[preds].values,
                          't': fit.tvalues[preds].values, 'df': fit.df_resid,
                          'p': fit.pvalues[preds].values, 'ci_lo': ci[0].values, 'ci_hi': ci[1].values})
    table['p_fdr'] = fdr(table['p'])
    table['n_sub'] = len(diff)
    table['n_obs'] = len(diff)
    table['spec'] = 'difference_score'
    return table


def _show_terms(table: pd.DataFrame, title: str) -> None:
    view = table.assign(predictor=table['predictor'].map(clean_name), sig=table['p_fdr'].map(stars))
    print(f'\n{title}')
    code_block(view[['predictor', 'beta', 'se', 't', 'df', 'p', 'p_fdr', 'sig']]
               .to_string(index=False, float_format=lambda x: f'{x:.4f}'))


def collinearity(flower, table, preds) -> list[str]:
    """VIF screen on the participant x modality complete cases; return the retained predictors."""
    pool = flower[['sub_id', 'modality']].drop_duplicates().merge(table, on='sub_id', how='inner')
    pool, _ = handle_outliers(pool, [], preds)
    model_preds, removed, final_vif = prune_vif(preds, pool.dropna(subset=preds))
    if removed:
        print('Removed for VIF > 5: ' + ', '.join(f'{p} ({v})' for p, v in removed))
    else:
        print('All VIF <= 5; every predictor retained.')
    code_block(final_vif[['predictor', 'VIF']].round(2))
    return model_preds


def fit_outcome(df, outcome, do_boxcox, preds) -> list[pd.DataFrame]:
    """Slope model and difference-score sensitivity for one outcome."""
    df, removed = handle_outliers(df, [outcome], preds)
    if removed:
        print(f'Removed ({len(removed)}): {sorted(removed)}\n')
    lmbda = shift = None
    if do_boxcox:
        _, lmbda, shift = boxcox_fit(df[outcome])
        print(f'Box-Cox lambda = {lmbda:.4f}\n')
    base = prepare(df, outcome, preds, lmbda, shift)
    fit, table, data = fit_interaction_model(base, outcome, preds)
    if fit is None:
        print(f"Skipped: only {data['sub_id'].nunique()} complete cases.")
        return []
    print(f"n = {data['sub_id'].nunique()} participants, {len(data)} trials.")
    print_fit(f'{label_for(outcome)} ~ predictors x modality (by-participant modality slope)', fit)
    _show_terms(table, 'Interaction terms (mouse vs. tablet), modality slope, FDR within outcome')
    diff = difference_score_model(base, outcome, preds)
    _show_terms(diff, f"Sensitivity: participant difference scores (OLS, n = {diff['n_sub'].iloc[0]}), "
                      'FDR within outcome')
    return [table, diff]


def run_family(family, flower, table, preds) -> list[pd.DataFrame]:
    """Fit every digital outcome for one predictor family."""
    heading(f'Family: {family}', 2)
    heading('Predictor collinearity', 3)
    model_preds = collinearity(flower, table, preds)
    rows = []
    for outcome, do_boxcox in OUTCOMES:
        heading(f'{label_for(outcome)} ({family})', 3)
        df = flower.merge(table, on='sub_id', how='left')
        rows += [t.assign(family=family) for t in fit_outcome(df, outcome, do_boxcox, model_preds)]
    return rows


def summarise(results: pd.DataFrame) -> None:
    heading('Interactions surviving FDR correction', 2)
    counts = results.groupby(['family', 'spec']).agg(
        tests=('p', 'size'), p_uncorrected=('p', lambda p: int((p < .05).sum())),
        fdr_survivors=('p_fdr', lambda p: int((p < .05).sum())))
    code_block(counts.reset_index())
    for spec in ('slope', 'difference_score'):
        sig = results[(results['spec'] == spec) & (results['p_fdr'] < 0.05)].copy()
        sig['outcome'] = sig['outcome'].map(label_for)
        sig['predictor'] = sig['predictor'].map(qualified_label)
        print(f'\n{spec}')
        code_block(sig[['family', 'outcome', 'predictor', 'beta', 'se', 't', 'df', 'p', 'p_fdr']]
                   .to_string(index=False, float_format=lambda x: f'{x:.3f}'))


def main() -> None:
    with report('joint_lmm_exploratory_output.md'):
        heading('Cognition x modality interactions on the digital Flower Trail outcomes')
        flower = load_flower(digital_only=True)
        print(describe_handling() + '\n')
        cog = load_cognitive_battery()[['sub_id', *PREDICTORS]]
        maze = load_maze_agg()[['sub_id', *MAZE_PREDICTORS]]
        rows = run_family('executive', flower, cog, PREDICTORS)
        rows += run_family('maze', flower, maze, MAZE_PREDICTORS)
        results = pd.concat(rows, ignore_index=True)
        results.to_csv(RESULTS / 'joint_lmm_exploratory_interactions.csv', index=False)
        summarise(results)
        plot_interaction_panels(results[results['spec'] == 'slope'])


# ── Interaction panels ──────────────────────────────────────────────────────────────────

def plot_interaction_panels(results: pd.DataFrame) -> None:
    """Forest plot of the predictor x modality interactions, one panel per outcome."""
    panels = [o for o in FIGURE_OUTCOMES if o in set(results['outcome'])]
    ys = np.arange(len(FIGURE_PREDICTORS))[::-1]
    lo = (results['beta'] - 1.96 * results['se']).min()
    hi = (results['beta'] + 1.96 * results['se']).max()
    pad = 0.08 * (hi - lo)
    xlim = (min(lo, 0) - pad, max(hi, 0) + pad)

    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.9 * nrows), sharex=True, sharey=True)
    flat = axes.ravel()
    for k, outcome in enumerate(panels):
        ax = flat[k]
        rows = results[results['outcome'] == outcome].set_index('predictor')
        ax.axvline(0, color='#444444', lw=1, ls='--', zorder=1)
        for y, pred in zip(ys, FIGURE_PREDICTORS):
            if pred not in rows.index:
                continue
            r = rows.loc[pred]
            sig = r['p_fdr'] < 0.05
            col = SIG_COLOR if sig else NS_COLOR
            ax.plot([r['beta'] - 1.96 * r['se'], r['beta'] + 1.96 * r['se']], [y, y],
                    color=col, lw=2, solid_capstyle='round', zorder=2)
            ax.plot([r['beta']], [y], 'o', ms=7, zorder=3, color=col if sig else 'white',
                    markeredgecolor=col, markeredgewidth=1.6)
        ax.set_xlim(xlim)
        ax.set_ylim(-0.7, len(FIGURE_PREDICTORS) - 0.3)
        ax.set_title(f'$\\bf{{{chr(97 + k)}}}$   {label_for(outcome)}', fontsize=10, loc='left', pad=8)
        if k >= len(panels) - ncols:
            ax.set_xlabel('β (mouse vs. tablet)', fontsize=9)
        ax.tick_params(labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', color='#e6e6e6', lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    for row in range(nrows):
        flat[row * ncols].set_yticks(ys)
        flat[row * ncols].set_yticklabels([qualified_label(p) for p in FIGURE_PREDICTORS], fontsize=9)

    handles = [
        plt.Line2D([], [], color=SIG_COLOR, marker='o', lw=2, ms=7, label='FDR-significant'),
        plt.Line2D([], [], color=NS_COLOR, marker='o', lw=2, ms=7, markerfacecolor='white',
                   markeredgecolor=NS_COLOR, label='Not significant'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.55, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(RESULTS / 'kinematics_interaction_panels.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
