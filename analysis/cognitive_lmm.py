"""Single-domain cognitive models of Flower Trail errors per minute.

One cognitive task at a time (maze, Flanker, gradCPT, WCST), this module fits

1. confirmatory models, one per predictor:
       errors_per_min ~ predictor + modality + trial + (1 | participant)
2. a within-domain joint model with every predictor of the task entered together;
3. exploratory interaction models, one per predictor:
       errors_per_min ~ predictor * modality + trial + (1 + modality | participant)
   The predictor varies between participants and modality within them, so the
   by-participant modality slope supplies the error term for the interaction, as
   in a repeated-measures ANOVA. The omnibus predictor-by-modality interaction is
   tested by a Satterthwaite F-test. Only where that test is significant are the
   tablet- and mouse-versus-paper contrasts tested, from the same fit.

p values are FDR-corrected within each family; for the contrasts the family is
the contrasts tested in that domain. Outliers follow ``outlier_policy.HANDLING``
(no handling in the main analysis; a Mahalanobis screen on the participant
means of errors per minute and the predictors in the replication).
Errors per minute is then Box-Cox transformed and z-scored; predictors are
z-scored on their subject-level values.
The four domain scripts (``maze_lmm.py`` etc.) declare the task and call
:func:`run_domain`. ``outlier_multiverse.py`` reuses :func:`confirmatory_models`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import (
    MOD_PAPER, MOUSE_VS_PAPER, RESULTS, TABLET_VS_PAPER, boxcox_fit, code_block, fdr,
    format_f, heading, lmer_fit, print_fit, report, standardise_predictors,
    stars, vif_table, zscore,
)
from load_tables import load_cognitive_battery, load_flower
from measure_names import MODALITY_ORDER, clean_name, label_for
from outlier_policy import describe_handling, filter_subjects, handle_outliers

OUTCOME = 'mistakes_per_min'
ALPHA = 0.05
CONTRASTS = {'tablet_vs_paper': TABLET_VS_PAPER, 'mouse_vs_paper': MOUSE_VS_PAPER}


@dataclass
class DomainSpec:
    name: str
    hypothesis: str
    title: str
    predictors: list[str]
    load_predictors: Callable[[], pd.DataFrame] = field(default=None)

    def predictor_table(self) -> pd.DataFrame:
        if self.load_predictors is not None:
            table = self.load_predictors()
        else:
            table = load_cognitive_battery()
        return table[['sub_id', *self.predictors]].copy()


def _model_frame(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    return df[['sub_id', OUTCOME, *preds, 'modality', 'trial']].dropna().copy()


def _term_row(fit, term: str, data: pd.DataFrame, **extra) -> dict:
    ci = fit.conf_int().loc[term]
    return {
        **extra,
        'beta': float(fit.fe_params[term]),
        'se': float(fit.bse[term]),
        't': float(fit.tvalues[term]),
        'df': float(fit.df[term]),
        'p': float(fit.pvalues[term]),
        'ci_lo': float(ci[0]),
        'ci_hi': float(ci[1]),
        'n_sub': int(data['sub_id'].nunique()),
        'n_obs': int(len(data)),
    }


def _show(table: pd.DataFrame, cols: list[str]) -> None:
    view = table[cols].copy()
    view['predictor'] = view['predictor'].map(clean_name)
    if 'contrast' in view.columns:
        view['contrast'] = view['contrast'].map(clean_name)
    view['sig'] = view['p_fdr'].map(lambda p: stars(p) if p == p else 'not tested')
    code_block(view.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


def confirmatory_models(df: pd.DataFrame, preds: list[str], verbose: bool = True) -> pd.DataFrame:
    """One model per predictor, outcome ~ predictor + modality + trial + (1 | participant).

    ``df`` holds the transformed outcome and standardised predictors. p values are
    FDR-corrected across the predictors.
    """
    rows = []
    for pred in preds:
        data = _model_frame(df, [pred])
        fit = lmer_fit(f"{OUTCOME} ~ {pred} + {MOD_PAPER} + trial", data)
        if verbose:
            print_fit(f'Errors per minute ~ {label_for(pred)}', fit)
        rows.append(_term_row(fit, pred, data, block='confirmatory', predictor=pred))
    conf = pd.DataFrame(rows)
    conf['p_fdr'] = fdr(conf['p'])
    return conf


def run_domain(spec: DomainSpec) -> None:
    with report(f'{spec.name}_lmm_output.md'):
        heading(f'{spec.hypothesis}: {spec.title} and Flower Trail errors per minute')

        flower = load_flower()
        preds = spec.predictor_table()
        df = flower.merge(preds, on='sub_id', how='inner')
        df, outlier_ids = handle_outliers(df, [OUTCOME], spec.predictors)

        print(f"Flower Trail: {flower['sub_id'].nunique()} participants; "
              f"{spec.title}: {preds.dropna()['sub_id'].nunique()} with complete scores.")
        print(describe_handling())
        if outlier_ids:
            print(f'Removed ({len(outlier_ids)}): {sorted(outlier_ids)}')
        print(f"Analysis sample: {df['sub_id'].nunique()} participants, {len(df)} trials.\n")
        print('Predictor descriptives (participant level)')
        code_block(preds[spec.predictors].describe().round(3).to_string())

        df[OUTCOME], lmbda, _ = boxcox_fit(df[OUTCOME])
        df[OUTCOME] = zscore(df[OUTCOME])
        df = standardise_predictors(df, spec.predictors)
        print(f'\nBox-Cox lambda for errors per minute: {lmbda:.4f}')

        # ── Confirmatory: one predictor per model ──────────────────────────
        heading('Confirmatory models', 2)
        conf = confirmatory_models(df, spec.predictors)
        print('\nConfirmatory summary (FDR within domain)')
        _show(conf, ['predictor', 'beta', 'se', 't', 'df', 'p', 'p_fdr', 'n_sub'])

        # ── Predictor collinearity ─────────────────────────────────────────
        heading('Predictor correlations and variance inflation', 2)
        sub = df.drop_duplicates('sub_id')[spec.predictors]
        print('Participant-level correlations (pairwise complete)')
        corr = sub.corr(min_periods=10).round(3)
        corr.index = corr.columns = [clean_name(c) for c in corr.columns]
        code_block(corr.to_string())
        vif = vif_table(sub, spec.predictors)
        if vif is not None:
            print(f'\nVariance inflation factors (n = {len(sub.dropna())} complete cases)')
            code_block(vif[['predictor', 'VIF']].round(3))
        else:
            print('\nToo few complete cases for variance inflation factors.')

        # ── Within-domain joint model ──────────────────────────────────────
        heading('Joint model with all predictors of the task', 2)
        data = _model_frame(df, spec.predictors)
        joint = pd.DataFrame()
        if data['sub_id'].nunique() > len(spec.predictors) + 2:
            fit = lmer_fit(f"{OUTCOME} ~ {' + '.join(spec.predictors)} + {MOD_PAPER} + trial", data)
            print_fit(f"Errors per minute ~ {' + '.join(label_for(p) for p in spec.predictors)}", fit)
            joint = pd.DataFrame([_term_row(fit, p, data, block='joint', predictor=p)
                                  for p in spec.predictors])
            joint['p_fdr'] = fdr(joint['p'])
            print('\nJoint-model summary (FDR within domain)')
            _show(joint, ['predictor', 'beta', 'se', 't', 'df', 'p', 'p_fdr', 'n_sub'])
        else:
            print('Too few complete cases to fit the joint model.')

        # ── Exploratory: predictor-by-modality interactions ───────────────
        heading('Exploratory interaction models (by-participant modality slope)', 2)
        rows, omnibus = [], {}
        for pred in spec.predictors:
            data = _model_frame(df, [pred])
            fit = lmer_fit(f"{OUTCOME} ~ {pred} * {MOD_PAPER} + trial", data, slope=MOD_PAPER)
            print_fit(f'Errors per minute ~ {label_for(pred)} x modality', fit)
            omnibus[pred] = fit.interaction_f()
            print(f"Satterthwaite F-test of the {label_for(pred)} x modality interaction: "
                  f"{format_f(omnibus[pred])}")
            tested = omnibus[pred]['p'] < ALPHA
            for contrast, key in CONTRASTS.items():
                row = _term_row(fit, f'{pred}:{key}', data, block='interaction',
                                predictor=pred, contrast=contrast, tested=tested,
                                F=omnibus[pred]['F'], num_df=omnibus[pred]['num_df'],
                                den_df=omnibus[pred]['den_df'], p_F=omnibus[pred]['p'])
                if not tested:
                    row.update(t=float('nan'), df=float('nan'), p=float('nan'))
                rows.append(row)
        inter = pd.DataFrame(rows)
        inter['p_fdr'] = float('nan')
        if inter['tested'].any():
            inter.loc[inter['tested'], 'p_fdr'] = fdr(inter.loc[inter['tested'], 'p'])
        print(f'\nInteraction contrasts versus paper, tested where the omnibus F has p < {ALPHA} '
              f'(FDR across the {int(inter["tested"].sum())} tested contrasts)')
        _show(inter, ['predictor', 'contrast', 'beta', 'se', 't', 'df', 'p', 'p_fdr', 'F', 'p_F'])

        summary = pd.concat([conf, joint, inter], ignore_index=True)
        summary.to_csv(RESULTS / f'{spec.name}_lmm_summary.csv', index=False)

        _plot_confirmatory(spec, flower, preds, outlier_ids, conf)
        _plot_interactions(spec, df, omnibus)


# ── Figures ───────────────────────────────────────────────────────────────────

def _grid(n: int):
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True, squeeze=False)
    return fig, axes.ravel()


def _annotate(ax, text: str, letter: str) -> None:
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.7))
    ax.text(-0.15, 1.02, letter, transform=ax.transAxes, fontsize=14, fontweight='bold')


def _plot_confirmatory(spec, flower, preds, outlier_ids, conf) -> None:
    """Participant-by-modality mean errors per minute against each raw predictor."""
    means = flower.groupby(['sub_id', 'modality'], observed=True)[OUTCOME].mean().reset_index()
    plot_df = filter_subjects(means.merge(preds, on='sub_id', how='inner'), outlier_ids)

    fig, axes = _grid(len(spec.predictors))
    for i, (ax, pred) in enumerate(zip(axes, spec.predictors)):
        row = conf.loc[conf['predictor'] == pred].iloc[0]
        colour = 'steelblue' if row['p_fdr'] < 0.05 else 'grey'
        sns.regplot(x=pred, y=OUTCOME, data=plot_df[[pred, OUTCOME]].dropna(), ax=ax, ci=95,
                    scatter_kws={'color': 'lightgrey', 's': 20, 'alpha': 0.6},
                    line_kws={'color': colour, 'linewidth': 3})
        _annotate(ax, rf"$\beta$ = {row['beta']:.2f} | $p_{{FDR}}$ = {row['p_fdr']:.3f} "
                      f"{stars(row['p_fdr'])}", chr(97 + i))
        ax.set_xlabel(label_for(pred), fontsize=10)
        ax.set_ylabel(label_for(OUTCOME), fontsize=10)
        ax.set_ylim(bottom=0)
    for ax in axes[len(spec.predictors):]:
        ax.set_visible(False)
    fig.savefig(RESULTS / f'{spec.name}_confirmatory.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_interactions(spec, df, omnibus) -> None:
    """Modality-specific regression lines on the standardised model data."""
    colours = dict(zip(MODALITY_ORDER, sns.color_palette('tab10', n_colors=3)))
    fig, axes = _grid(len(spec.predictors))
    for i, (ax, pred) in enumerate(zip(axes, spec.predictors)):
        data = df[[pred, OUTCOME, 'modality']].dropna()
        ax.scatter(data[pred], data[OUTCOME], color='lightgrey', s=15, alpha=0.6, zorder=1)
        for mod, colour in colours.items():
            sub = data[data['modality'] == mod]
            if not sub.empty:
                sns.regplot(x=pred, y=OUTCOME, data=sub, ax=ax, scatter=False, ci=95,
                            color=colour, label=mod.capitalize(),
                            line_kws={'linewidth': 2.5, 'zorder': 2})
        p = omnibus[pred]['p']
        _annotate(ax, rf'Interaction F-test: $p$ = {p:.3f} {stars(p)}', chr(97 + i))
        ax.set_xlabel(f'{label_for(pred)} (z)', fontsize=10)
        ax.set_ylabel(f'{label_for(OUTCOME)} (z)', fontsize=10)
    for ax in axes[len(spec.predictors):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0),
               title='Modality', fontsize=10, frameon=True, ncol=3)
    fig.savefig(RESULTS / f'{spec.name}_exploratory.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
