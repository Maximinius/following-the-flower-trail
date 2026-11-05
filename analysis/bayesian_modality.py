"""Descriptives and Bayesian mixed-model comparisons of Flower Trail performance by modality.

Participant-level means per modality are reported for errors per minute and its
two components, completion time and the boundary-error count (Table S2). Errors
per minute is compared across modalities with a Bayesian mixed model on the
participant x modality means (BayesFactor::lmBF, JZS priors: fixed-effect scale
r = 1/2, which for a two-level factor equals the r = sqrt(2)/2 of a JZS t-test;
random participant intercept). Every participant with at least one modality
enters, so no one is dropped for a missing modality. The omnibus model and each
pairwise model (fitted on the two modalities concerned) are tested against a
participant-only model, and the posterior gives the difference in errors per
minute and the standardised effect delta (Table S3). The complete-triplet
repeated-measures ANOVA and paired t-tests are kept as a sensitivity analysis.
Descriptives are computed on the raw values; the models use the data after
``outlier_policy.HANDLING`` (none in the main analysis; a Mahalanobis screen on
errors per minute and the four digital kinematic outcomes in the replication).

Outputs: bayesian_modality_output.md, bayesian_modality_summary.csv,
         bayesian_modality_complete_case.csv
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd
import pingouin as pg

from common import RESULTS, _py_frame, _r_frame, code_block, heading, report
from load_tables import FTT_SCREEN, MODALITIES, load_flower
from measure_names import label_for
from outlier_policy import describe_handling, excluded_ids, filter_subjects, winsorize_active

OUTCOMES = ['mistakes_per_min', 'completion_time', 'error_count']
PRIMARY = 'mistakes_per_min'
BF_SEED = 20260929
POSTERIOR_DRAWS = 20000

_R_LMBF = """
function(d, seed, draws) {
  suppressPackageStartupMessages(library(BayesFactor))
  set.seed(seed)
  d$sub_id <- factor(d$sub_id)
  d$modality <- factor(d$modality)
  full <- lmBF(y ~ modality + sub_id, data = d, whichRandom = 'sub_id',
               rscaleFixed = 'medium', rscaleRandom = 'nuisance', progress = FALSE)
  null <- lmBF(y ~ sub_id, data = d, whichRandom = 'sub_id',
               rscaleRandom = 'nuisance', progress = FALSE)
  bf <- extractBF(full / null)
  post <- as.matrix(posterior(full, iterations = draws, progress = FALSE))
  effects <- post[, paste0('modality-', levels(d$modality)), drop = FALSE]
  data.frame(bf10 = bf$bf, error = bf$error,
             level = levels(d$modality),
             effect_mean = colMeans(effects),
             sigma = mean(sqrt(post[, 'sig2'])),
             stringsAsFactors = FALSE)
}
"""

_R_PAIR = """
function(d, seed, draws, a, b) {
  suppressPackageStartupMessages(library(BayesFactor))
  set.seed(seed)
  d$sub_id <- factor(d$sub_id)
  d$modality <- factor(d$modality, levels = c(a, b))
  full <- lmBF(y ~ modality + sub_id, data = d, whichRandom = 'sub_id',
               rscaleFixed = 'medium', rscaleRandom = 'nuisance', progress = FALSE)
  null <- lmBF(y ~ sub_id, data = d, whichRandom = 'sub_id',
               rscaleRandom = 'nuisance', progress = FALSE)
  bf <- extractBF(full / null)
  post <- as.matrix(posterior(full, iterations = draws, progress = FALSE))
  diff <- post[, paste0('modality-', a)] - post[, paste0('modality-', b)]
  delta <- diff / sqrt(post[, 'sig2'])
  q <- function(x) unname(quantile(x, c(.025, .975)))
  data.frame(bf10 = bf$bf, error = bf$error,
             diff = mean(diff), diff_lo = q(diff)[1], diff_hi = q(diff)[2],
             delta = mean(delta), delta_lo = q(delta)[1], delta_hi = q(delta)[2])
}
"""


def subject_means(flower: pd.DataFrame, col: str) -> pd.DataFrame:
    """Participant x modality table of means, NaN where a modality is missing."""
    means = (flower.dropna(subset=[col])
             .groupby(['sub_id', 'modality'], observed=False)[col].mean()
             .unstack('modality'))
    return means.reindex(columns=MODALITIES)


def _long(means: pd.DataFrame) -> pd.DataFrame:
    return (means.reset_index().melt(id_vars='sub_id', var_name='modality', value_name='y')
            .dropna().assign(sub_id=lambda d: d['sub_id'].astype(str),
                             modality=lambda d: d['modality'].astype(str)))


def bayes_mixed(means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Omnibus and pairwise lmBF mixed models on every available participant x modality mean."""
    from rpy2 import robjects as ro

    long = _long(means)
    omni = _py_frame(ro.r(_R_LMBF)(_r_frame(long), BF_SEED, POSTERIOR_DRAWS))
    rows = []
    for a, b in combinations(MODALITIES, 2):
        pair = long[long['modality'].isin([a, b])]
        per_sub = pair.groupby('sub_id')['modality'].nunique()
        res = _py_frame(ro.r(_R_PAIR)(_r_frame(pair), BF_SEED, POSTERIOR_DRAWS, a, b)).iloc[0]
        rows.append({'A': a, 'B': b, 'n_sub': len(per_sub), 'n_both': int((per_sub == 2).sum()),
                     **res.to_dict()})
    table = pd.DataFrame(rows)
    table['BF01'] = 1.0 / table['bf10']
    table = table.rename(columns={'bf10': 'BF10', 'error': 'BF_error'})
    return omni, table


def main() -> None:
    with report('bayesian_modality_output.md'):
        heading('Flower Trail performance by modality')
        flower = load_flower()
        excluded = excluded_ids(flower, FTT_SCREEN)
        flower = filter_subjects(flower, excluded)
        print(describe_handling())
        if excluded:
            print(f'Removed ({len(excluded)}): {sorted(excluded)}')
        print(f"Analysis sample: {flower['sub_id'].nunique()} participants, {len(flower)} trials.")

        heading('Participant-level descriptives', 2)
        for col in OUTCOMES:
            means = subject_means(flower, col)
            desc = pd.DataFrame({
                'n': means.notna().sum(), 'mean': means.mean(), 'median': means.median(),
                'SD': means.std(), 'min': means.min(), 'max': means.max(),
            })
            print(f'\n{label_for(col)}')
            code_block(desc.to_string(float_format=lambda x: f'{x:.2f}'))

        heading('Bayesian mixed-model comparisons of errors per minute', 2)
        means = subject_means(winsorize_active(flower, [PRIMARY]), PRIMARY)
        pattern = means.notna().apply(lambda r: '+'.join(m for m in MODALITIES if r[m]), axis=1)
        print(f'Participants: n = {len(means)}; modalities available:')
        code_block(pattern.value_counts().rename_axis('modalities').reset_index(name='n'))
        omni, table = bayes_mixed(means)
        bf = omni['bf10'].iloc[0]
        print(f"\nOmnibus modality effect: BF10 = {bf:.4g} (BF01 = {1 / bf:.4g}, "
              f"error {100 * omni['error'].iloc[0]:.2g}%).")
        print('Posterior mean deviation of each modality from the grand mean (errors per minute)')
        code_block(omni[['level', 'effect_mean']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        print('\nPairwise mixed models: posterior difference A - B (errors per minute, 95% CrI), '
              'standardised delta and JZS Bayes factors')
        code_block(table.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        table.to_csv(RESULTS / 'bayesian_modality_summary.csv', index=False)

        heading('Sensitivity: complete triplets', 2)
        triplets = means.dropna()
        print(f'Participants with data in all three modalities: n = {len(triplets)}.\n')
        long = (triplets.reset_index()
                .melt(id_vars='sub_id', var_name='modality', value_name=PRIMARY))

        anova = pg.rm_anova(data=long, dv=PRIMARY, within='modality', subject='sub_id', detailed=True,
                          correction=True)
        effect = anova.loc[anova['Source'] == 'modality'].iloc[0]
        error = anova.loc[anova['Source'] == 'Error'].iloc[0]
        print(f"Repeated-measures ANOVA: F({int(effect['DF'])}, {int(error['DF'])}) = {effect['F']:.2f}, "
              f"p = {effect['p_unc']:.4f}, Greenhouse-Geisser p = {effect['p_GG_corr']:.4f}, "
              f"epsilon = {effect['eps']:.3f}.\n")

        tests = pg.pairwise_tests(data=long, dv=PRIMARY, within='modality', subject='sub_id',
                                  parametric=True, padjust='fdr_bh', effsize='hedges')
        tests['BF10'] = [float(pg.bayesfactor_ttest(t, len(triplets), paired=True)) for t in tests['T']]
        tests['BF01'] = 1.0 / tests['BF10']
        tests['n'] = len(triplets)
        complete = tests[['A', 'B', 'n', 'T', 'dof', 'p_unc', 'p_corr', 'hedges', 'BF10', 'BF01']]
        print('Paired t-tests (Benjamini-Hochberg corrected) with JZS Bayes factors')
        code_block(complete.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        complete.to_csv(RESULTS / 'bayesian_modality_complete_case.csv', index=False)


if __name__ == '__main__':
    main()
