"""Cross-domain joint model of Flower Trail errors per minute.

All cognitive predictors are entered together so that each coefficient is
adjusted for the others:

    errors_per_min ~ Flanker (2) + WCST (2) + gradCPT (3) + maze navigation errors per minute
                     + modality + trial + (1 | participant)

Commission-error RT is not entered because it is missing for participants
without commission errors. The model is fitted on the participants with every
predictor observed. Outliers follow ``outlier_policy.HANDLING`` (no handling in
the main analysis; a Mahalanobis screen on the participant means of errors per
minute and the predictors in the replication). The predictor set is
screened for collinearity (VIF > 5 drops the worst predictor) before fitting,
and the predictor p values are FDR-corrected as one family.

Outputs: joint_lmm_output.md, joint_lmm_summary.csv,
         joint_lmm_predictor_correlation_heatmap.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from common import (
    MOD_PAPER, RESULTS, boxcox_fit, code_block, fdr, heading, lmer_fit, print_fit, prune_vif, report,
    standardise_predictors, stars, vif_table, zscore,
)
from load_tables import JOINT_PREDS_FULL, build_predictors, load_flower
from measure_names import clean_name
from outlier_policy import describe_handling, handle_outliers

OUTCOME = 'mistakes_per_min'


def correlation_heatmap(sub_wide: pd.DataFrame, preds: list[str]) -> None:
    """Lower-triangle Pearson correlations between the participant-level predictors."""
    names = [clean_name(p) for p in preds]
    r = pd.DataFrame(np.nan, index=names, columns=names)
    annot = pd.DataFrame('', index=names, columns=names)
    for i, a in enumerate(preds):
        r.iloc[i, i] = 1.0
        annot.iloc[i, i] = '—'
        for j in range(i + 1, len(preds)):
            pair = sub_wide[[a, preds[j]]].dropna()
            if len(pair) >= 10:
                rv, pv = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                r.iloc[i, j] = r.iloc[j, i] = rv
                mark = stars(pv).replace('ns', '')
                annot.iloc[i, j] = annot.iloc[j, i] = f'{rv:.2f}{mark}'
    mask = np.triu(np.ones(r.shape, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(r, mask=mask, annot=annot, fmt='', cmap='RdBu_r', vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Pearson r'}, annot_kws={'size': 8})
    fig.tight_layout()
    fig.savefig(RESULTS / 'joint_lmm_predictor_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    with report('joint_lmm_output.md'):
        heading('Joint cognitive model of Flower Trail errors per minute')

        flower = load_flower()
        preds = build_predictors()
        df = flower.merge(preds[['sub_id', *JOINT_PREDS_FULL]], on='sub_id', how='inner')
        df, outlier_ids = handle_outliers(df, [OUTCOME], JOINT_PREDS_FULL)
        print(describe_handling())
        if outlier_ids:
            print(f'Removed ({len(outlier_ids)}): {sorted(outlier_ids)}')
        print(f"Analysis sample: {df['sub_id'].nunique()} participants, {len(df)} trials.\n")

        print('Participant-level availability of each predictor')
        code_block('\n'.join(f'  {clean_name(c):<32} {preds[c].notna().sum():>3} / {len(preds)}'
                             for c in JOINT_PREDS_FULL + ['gradcpt_rt_mountain_commission']))

        df[OUTCOME], lmbda, _ = boxcox_fit(df[OUTCOME])
        df[OUTCOME] = zscore(df[OUTCOME])
        print(f'\nBox-Cox lambda for errors per minute: {lmbda:.4f}')

        heading('Predictor collinearity', 2)
        sub_wide = standardise_predictors(df, JOINT_PREDS_FULL).groupby('sub_id')[JOINT_PREDS_FULL].first()
        initial = vif_table(sub_wide, JOINT_PREDS_FULL)
        print(f'Variance inflation factors, all predictors (complete cases, n = {len(sub_wide.dropna())})')
        code_block(initial[['predictor', 'VIF']].round(2))
        model_preds, removed, final = prune_vif(JOINT_PREDS_FULL, sub_wide)
        if removed:
            print('\nRemoved for VIF > 5: ' + ', '.join(f'{p} ({v})' for p, v in removed))
        else:
            print('\nAll VIF <= 5; every predictor retained.')
        correlation_heatmap(sub_wide, JOINT_PREDS_FULL)

        heading('Main-effects model', 2)
        formula = f"{OUTCOME} ~ {' + '.join(model_preds)} + {MOD_PAPER} + trial"
        data = standardise_predictors(df, model_preds)
        data = data[['sub_id', OUTCOME, *model_preds, 'modality', 'trial']].dropna()
        fit = lmer_fit(formula, data)
        print(f"n = {data['sub_id'].nunique()} participants, {len(data)} trials.")
        print_fit('Errors per minute ~ all cognitive predictors + modality + trial', fit)
        table = summary_table(fit, model_preds, data['sub_id'].nunique())
        show_summary(table, 'Summary (FDR across the predictor family)')
        table.to_csv(RESULTS / 'joint_lmm_summary.csv', index=False)


def summary_table(fit, preds: list[str], n_sub: int) -> pd.DataFrame:
    ci = fit.conf_int()
    table = pd.DataFrame({'predictor': preds, 'beta': fit.fe_params[preds].values,
                          'se': fit.bse[preds].values, 't': fit.tvalues[preds].values,
                          'df': fit.df[preds].values, 'p': fit.pvalues[preds].values,
                          'ci_lo': ci.loc[preds, 0].values, 'ci_hi': ci.loc[preds, 1].values})
    table['p_fdr'] = fdr(table['p'])
    table['n_sub'] = n_sub
    return table


def show_summary(table: pd.DataFrame, title: str) -> None:
    view = table.assign(predictor=table['predictor'].map(clean_name), sig=table['p_fdr'].map(stars))
    print(f'\n{title}')
    code_block(view[['predictor', 'beta', 'se', 't', 'df', 'p', 'p_fdr', 'sig']]
               .to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
