"""H1 and H2: input modality and Flower Trail performance.

Confirmatory models, as pre-registered, with a by-participant modality slope:
    outcome ~ modality + trial + (1 + modality | participant)
for errors per minute, completion time and the boundary-error count (paper,
tablet, mouse; all three pairwise contrasts), and for path deviation and
movement speed (tablet and mouse only, since paper records neither). The mouse-
versus-tablet contrast of the three-modality outcomes comes from the same model
refitted with tablet as reference. The 11 confirmatory contrasts form one FDR
family.

Exploratory models (mouse versus tablet) fit the remaining digital outcomes,
error length, acceleration, number of stops and time stopped, with the same
random part. These 4 contrasts form a second FDR family.

Outliers follow ``outlier_policy.HANDLING``: the main analysis uses every
participant and every value; in the Mahalanobis replication participants are
screened on their means of errors per minute and the four digital kinematic
outcomes. Outcomes are then Box-Cox transformed where skewed and z-scored before
modelling. Figures are drawn from the untransformed participant means.
``outlier_multiverse.py`` reuses ``split_samples``, ``transform`` and the two
family functions.

Outputs: modality_lmm_output.md, modality_lmm_summary.csv,
         modality_confirmatory.png (Figure 1 with the full y-axis), modality_exploratory.png (Figure 2)
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (
    MOD_PAPER, MOD_TABLET, MOUSE_VS_PAPER, MOUSE_VS_TABLET, RESULTS, TABLET_VS_PAPER,
    boxcox_fit, code_block, fdr, heading, lmer_fit, print_fit, report, zscore,
)
from load_tables import FTT_SCREEN, load_flower
from measure_names import clean_name, label_for
from outlier_policy import describe_handling, excluded_ids, filter_subjects, winsorize_active

# (model column, Box-Cox). Pre-registered outcomes recorded in all three modalities.
CONFIRMATORY = [('mistakes_per_min', True), ('completion_time', True), ('error_count', True)]
# Pre-registered outcomes recorded in the digital modalities only.
CONFIRMATORY_DIGITAL = [('deviation', True), ('speed', False)]
# Exploratory digital-only outcomes.
EXPLORATORY = [('sec_mistakes', True), ('acceleration', False), ('stops', True), ('sec_stopped', True)]
# Panels of Figure 2, all mouse-versus-tablet contrasts.
FIGURE2 = ['sec_mistakes', 'deviation', 'speed', 'acceleration', 'stops', 'sec_stopped']
THREE_WAY = [(MOD_PAPER, TABLET_VS_PAPER), (MOD_PAPER, MOUSE_VS_PAPER), (MOD_TABLET, MOUSE_VS_TABLET)]
# Column shown in the figure for each model column.
PLOT_COLUMN = {'sec_mistakes': 'ms_mistakes', 'sec_stopped': 'ms_stopped'}

# Percentile cap applied to the plotted participant means of the two
# heavy-tailed outcomes in Figure 2. Models use the untouched data.
WINSOR_DISPLAY = {'deviation': 90, 'ms_mistakes': 90}


def fit_modality(df: pd.DataFrame, outcome: str, mod_term: str):
    data = df[['sub_id', outcome, 'modality', 'trial']].dropna()
    fit = lmer_fit(f"{outcome} ~ {mod_term} + trial", data, slope=mod_term)
    return fit, data


def contrast_row(fit, data, outcome, key) -> dict:
    ci = fit.conf_int().loc[key]
    return {
        'outcome': outcome, 'contrast': clean_name(key),
        'beta': float(fit.fe_params[key]), 'se': float(fit.bse[key]),
        't': float(fit.tvalues[key]), 'df': float(fit.df[key]), 'p': float(fit.pvalues[key]),
        'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]),
        'n_sub': int(data['sub_id'].nunique()), 'n_obs': int(len(data)),
    }


def split_samples(flower: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all modalities, digital only), with the second-scale columns added."""
    flower = flower.copy()
    flower['sec_stopped'] = flower['ms_stopped'] / 1000.0
    flower['sec_mistakes'] = flower['ms_mistakes'] / 1000.0
    digital = flower[flower['modality'] != 'paper'].copy()
    digital['modality'] = digital['modality'].cat.remove_unused_categories()
    return flower, digital


def transform(conf: pd.DataFrame, exp: pd.DataFrame, verbose: bool = True):
    """Box-Cox where skewed, then z-score, every outcome of both samples."""
    conf, exp = conf.copy(), exp.copy()
    for frame, spec in ((conf, CONFIRMATORY), (exp, CONFIRMATORY_DIGITAL + EXPLORATORY)):
        for col, do_boxcox in spec:
            if do_boxcox:
                frame[col], lmbda, _ = boxcox_fit(frame[col])
                if verbose:
                    print(f'  {label_for(col)}: lambda = {lmbda:.4f}')
            frame[col] = zscore(frame[col])
    return conf, exp


def confirmatory_family(conf: pd.DataFrame, exp: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """The 11 pre-registered contrasts, FDR-corrected as one family."""
    rows = []
    for col, _ in CONFIRMATORY:
        for mod_term in (MOD_PAPER, MOD_TABLET):
            fit, data = fit_modality(conf, col, mod_term)
            if verbose:
                ref = 'paper' if mod_term == MOD_PAPER else 'tablet'
                print_fit(f'{label_for(col)} ~ modality (reference: {ref})', fit)
            rows += [contrast_row(fit, data, col, key) for term, key in THREE_WAY if term == mod_term]
    for col, _ in CONFIRMATORY_DIGITAL:
        fit, data = fit_modality(exp, col, MOD_TABLET)
        if verbose:
            print_fit(f'{label_for(col)} ~ modality (reference: tablet)', fit)
        rows.append(contrast_row(fit, data, col, MOUSE_VS_TABLET))
    table = pd.DataFrame(rows)
    table['p_fdr'] = fdr(table['p'])
    return table


def exploratory_family(exp: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """The 4 exploratory mouse-versus-tablet contrasts, FDR-corrected as one family."""
    rows = []
    for col, _ in EXPLORATORY:
        fit, data = fit_modality(exp, col, MOD_TABLET)
        if verbose:
            print_fit(f'{label_for(col)} ~ modality (reference: tablet)', fit)
        rows.append(contrast_row(fit, data, col, MOUSE_VS_TABLET))
    table = pd.DataFrame(rows)
    table['p_fdr'] = fdr(table['p'])
    return table


def error_length_correlation(exp_raw: pd.DataFrame) -> None:
    heading('Error length and errors per minute', 2)
    pair = exp_raw[['sub_id', 'modality', 'ms_mistakes', 'mistakes_per_min']].dropna()
    cell = pair.groupby(['sub_id', 'modality'], observed=True)[['ms_mistakes', 'mistakes_per_min']].mean()
    print(f"Pearson r between error length and errors per minute (digital trials): "
          f"{pair['ms_mistakes'].corr(pair['mistakes_per_min']):.3f} across {len(pair)} trials, "
          f"{cell['ms_mistakes'].corr(cell['mistakes_per_min']):.3f} across {len(cell)} "
          f"participant-by-modality means.")


def main() -> None:
    with report('modality_lmm_output.md'):
        heading('H1 and H2: modality effects on Flower Trail performance')
        flower = load_flower()
        excluded = excluded_ids(flower, FTT_SCREEN)
        conf_raw, exp_raw = split_samples(filter_subjects(flower, excluded))
        print(describe_handling())
        if excluded:
            print(f'Removed ({len(excluded)}): {sorted(excluded)}')
        print(f"Confirmatory sample: {conf_raw['sub_id'].nunique()} participants, {len(conf_raw)} trials.")
        print(f"Exploratory sample (digital): {exp_raw['sub_id'].nunique()} participants, {len(exp_raw)} trials.")

        heading('Box-Cox transforms', 2)
        conf, exp = transform(winsorize_active(conf_raw, [c for c, _ in CONFIRMATORY]),
                              winsorize_active(exp_raw, [c for c, _ in CONFIRMATORY_DIGITAL + EXPLORATORY]))
        heading('Confirmatory models (by-participant modality slope)', 2)
        conf_table = confirmatory_family(conf, exp)
        heading('Exploratory models: digital outcomes, mouse versus tablet (by-participant modality slope)', 2)
        exp_table = exploratory_family(exp)

        heading('FDR-corrected contrasts', 2)
        summary = pd.concat([conf_table.assign(family='confirmatory'),
                             exp_table.assign(family='exploratory')], ignore_index=True)
        view = summary.assign(outcome=summary['outcome'].map(label_for))
        code_block(view[['family', 'outcome', 'contrast', 'beta', 'se', 't', 'df', 'p', 'p_fdr', 'n_sub', 'n_obs']]
                   .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        summary.to_csv(RESULTS / 'modality_lmm_summary.csv', index=False)
        error_length_correlation(exp_raw)

        plot_confirmatory(conf_raw, conf_table[conf_table['outcome'] == 'mistakes_per_min'])
        digital = pd.concat([conf_table[conf_table['contrast'] == clean_name(MOUSE_VS_TABLET)], exp_table])
        plot_exploratory(exp_raw, digital)


# ── Figures ───────────────────────────────────────────────────────────────────

def add_sig_bracket(ax, x1, x2, y, h, p) -> None:
    if p > 0.05:
        return
    text = '***' if p < 0.001 else '**' if p < 0.01 else '*'
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color='black')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', color='black',
            fontsize=12, fontweight='bold')


def plot_confirmatory(raw: pd.DataFrame, table: pd.DataFrame) -> None:
    """Participant mean errors per minute by modality, full y-axis (figure1.py draws Figure 1)."""
    out = 'mistakes_per_min'
    order = ['paper', 'tablet', 'mouse']
    subj = raw.groupby(['sub_id', 'modality'], observed=False)[out].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    sns.boxplot(data=subj, x='modality', y=out, ax=ax, order=order, showfliers=False,
                palette='tab10', hue='modality', legend=False)
    sns.stripplot(data=subj, x='modality', y=out, ax=ax, order=order, alpha=0.4, size=3,
                  palette='tab10', hue='modality', legend=False)
    ax.set_ylabel(label_for(out))
    ax.set_xlabel('')

    p_fdr = table.set_index('contrast')['p_fdr']
    y_max = subj[out].max()
    y_range = y_max - subj[out].min()
    h = y_range * 0.02
    add_sig_bracket(ax, 0, 1, y_max + y_range * 0.05, h, p_fdr[clean_name(TABLET_VS_PAPER)])
    add_sig_bracket(ax, 1, 2, y_max + y_range * 0.13, h, p_fdr[clean_name(MOUSE_VS_TABLET)])
    add_sig_bracket(ax, 0, 2, y_max + y_range * 0.23, h, p_fdr[clean_name(MOUSE_VS_PAPER)])
    ax.set_ylim(0, y_max + y_range * 0.33)
    fig.savefig(RESULTS / 'modality_confirmatory.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_exploratory(raw: pd.DataFrame, table: pd.DataFrame) -> None:
    """Figure 2: participant means of the digital outcomes, tablet versus mouse."""
    order = ['tablet', 'mouse']
    p_fdr = table.set_index('outcome')['p_fdr']
    fig, axes = plt.subplots(3, 2, figsize=(8, 10.5), constrained_layout=True)
    axes = axes.ravel()

    for i, model_col in enumerate(FIGURE2):
        ax = axes[i]
        out = PLOT_COLUMN.get(model_col, model_col)
        subj = raw.groupby(['sub_id', 'modality'], observed=False)[out].mean().reset_index()

        pct = WINSOR_DISPLAY.get(out)
        if pct is not None:
            cap = np.percentile(subj[out].dropna(), pct)
            n_capped = int((subj[out] > cap).sum())
            subj[out] = subj[out].clip(upper=cap)
            print(f'[figure 2] {label_for(out)}: plotted means capped at the {pct}th percentile '
                  f'(cap = {cap:.4g}, {n_capped} of {subj[out].notna().sum()} means capped).')

        sns.boxplot(data=subj, x='modality', y=out, ax=ax, order=order, showfliers=False,
                    palette='tab10', hue='modality', legend=False)
        sns.stripplot(data=subj, x='modality', y=out, ax=ax, order=order, alpha=0.4, size=3,
                      palette='tab10', hue='modality', legend=False)
        ax.set_ylabel(label_for(out))
        ax.set_xlabel('')
        ax.text(-0.1, 1.05, chr(97 + i), transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='bottom', ha='right')

        y_ceiling = np.percentile(subj[out].dropna(), 98)
        y_min = subj[out].min()
        y_range = y_ceiling - y_min
        add_sig_bracket(ax, 0, 1, y_ceiling + y_range * 0.05, y_range * 0.02, p_fdr[model_col])
        ax.set_ylim(y_min - y_range * 0.05, y_ceiling + y_range * 0.18)

    for ax in axes[len(FIGURE2):]:
        ax.set_visible(False)
    fig.savefig(RESULTS / 'modality_exploratory.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
