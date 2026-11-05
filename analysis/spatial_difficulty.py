"""H7: path deviation by geometric complexity of the flower trail.

The inner boundary is divided into twenty arc-length segments, each classified
as complex or simple by a median split on its curvature. Per-participant mean
path deviation in complex and simple segments, in each digital modality, is
modelled as

    deviation ~ segment_class * modality + (1 + modality + segment_class | participant)

fitted by REML in lmerTest, with Satterthwaite type III F-tests. Both factors
are coded -0.5 / +0.5, so each main effect is averaged over the other factor
and the random slopes are single numeric columns. If that fit is singular or
fails to converge, the slopes are decorrelated, ``(1 + modality + segment_class
|| participant)``, and if that also fails only the modality slope is kept,
``(1 + modality | participant)``. The report names the structure fitted and why.

Path deviation is the H2 measure (0 while the drawn line is clear of both walls,
the overlap or the distance outside otherwise, 0 in the start/finish gate),
Box-Cox transformed and z-scored. Deviation is recorded in the tablet and mouse modalities only.

Outliers follow ``outlier_policy.HANDLING``: the main analysis uses every
participant and every value; in the replication participants are screened by
Mahalanobis distance on their means of errors per minute and the four digital
kinematic outcomes.

Reads:   h7_segment_means.csv
Outputs: spatial_difficulty_output.md, spatial_difficulty_summary.csv (F-tests),
         spatial_difficulty.png, spatial_difficulty_segments.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS, _r_frame, boxcox_fit, code_block, heading, report, zscore
from load_tables import FTT_SCREEN, load_flower, load_h7_means
from flower_geometry import (
    N_SEGMENTS, build_flower_geometry, compute_arc_lengths, segment_curvature,
)
from outlier_policy import describe_handling, excluded_ids, filter_subjects, winsorize_active

# (random part, description), tried in order until a fit is regular and converged.
STRUCTURES = [
    ('(1 + mod + seg | sub_id)', 'correlated by-participant slopes for modality and segment class'),
    ('(1 + mod + seg || sub_id)', 'uncorrelated by-participant slopes for modality and segment class'),
    ('(1 + mod | sub_id)', 'by-participant slope for modality only'),
]
MEASURES = {
    'line_rule': 'Path deviation as in H2 (1-px line rule, Box-Cox, z-scored)',
}
TERMS = {'seg': 'Difficulty (complex vs. simple)', 'mod': 'Modality (mouse vs. tablet)',
         'seg:mod': 'Difficulty x modality'}

R_FIT = '''
function(d, f) {
  msgs <- character(0)
  m <- withCallingHandlers(
    lmerTest::lmer(as.formula(f), data = d, REML = TRUE),
    warning = function(w) { msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(w) { msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleMessage") })
  msgs <- trimws(unique(c(msgs, m@optinfo$conv$lme4$messages)))
  a <- as.data.frame(anova(m, type = 3))
  co <- as.data.frame(summary(m)$coefficients)
  list(singular = lme4::isSingular(m), messages = paste(msgs, collapse = "; "),
       anova = cbind(term = rownames(a), a), coef = cbind(term = rownames(co), co),
       varcorr = paste(capture.output(print(lme4::VarCorr(m))), collapse = "\\n"))
}
'''


# ── Data ──────────────────────────────────────────────────────────────────────

def classify_segments(inner) -> set:
    """Median split of the arc-length segments of the inner boundary on curvature."""
    inner_arr = np.array(inner, dtype=float)
    cumlen = compute_arc_lengths(inner)
    curvature = segment_curvature(inner_arr, cumlen, N_SEGMENTS)
    threshold = np.median(curvature)
    complex_segments = set(np.where(curvature > threshold)[0])

    heading('Segment classification (median split on curvature)', 2)
    print('| Segment | Curvature (rad/px) | Class |')
    print('|---|---|---|')
    for s in range(N_SEGMENTS):
        print(f"| {s:02d} | {curvature[s]:.4f} | {'complex' if s in complex_segments else 'simple'} |")
    print(f'\nMedian curvature = {threshold:.4f} rad/px; {len(complex_segments)} complex, '
          f'{N_SEGMENTS - len(complex_segments)} simple segments.')
    return complex_segments


def cell_means() -> pd.DataFrame:
    """Participant x modality x segment-class means after the active outlier handling."""
    agg = load_h7_means()
    excluded = excluded_ids(load_flower(), FTT_SCREEN)
    if excluded:
        print(f'Removed ({len(excluded)}): {sorted(excluded)}')
    return winsorize_active(filter_subjects(agg, excluded), list(MEASURES))


# ── Model ─────────────────────────────────────────────────────────────────────

def model_frame(agg: pd.DataFrame, measure: str) -> pd.DataFrame:
    data = agg[['sub_id', 'modality', 'difficulty', measure]].dropna().rename(columns={measure: 'y'})
    if measure == 'line_rule':
        data['y'], lmbda, shift = boxcox_fit(data['y'])
        data['y'] = zscore(data['y'])
        print(f'Box-Cox lambda = {lmbda:.4f} (shift {shift:.2f}), then z-scored.')
    data['seg'] = np.where(data['difficulty'] == 'complex', 0.5, -0.5)
    data['mod'] = np.where(data['modality'] == 'mouse', 0.5, -0.5)
    data['sub_id'] = data['sub_id'].astype(str)
    return data


def fit_structure(data: pd.DataFrame, random: str) -> dict:
    from rpy2 import robjects as ro
    res = ro.r(R_FIT)(_r_frame(data[['sub_id', 'y', 'seg', 'mod']]), f'y ~ seg * mod + {random}')
    out = dict(zip(res.names, res))
    return {'singular': bool(out['singular'][0]), 'messages': str(out['messages'][0]),
            'anova': _to_pandas(out['anova']), 'coef': _to_pandas(out['coef']),
            'varcorr': str(out['varcorr'][0])}


def _to_pandas(r_frame) -> pd.DataFrame:
    from common import _py_frame
    return _py_frame(r_frame).reset_index(drop=True)


def fit_cascade(data: pd.DataFrame) -> tuple[dict, str, str]:
    """First structure whose fit is regular and converged; (fit, structure, reason)."""
    reasons = []
    for random, description in STRUCTURES:
        fit = fit_structure(data, random)
        problem = 'singular fit' if fit['singular'] else (fit['messages'] or '')
        if not problem:
            why = 'the maximal structure fitted without problems' if not reasons else '; '.join(reasons)
            return fit, f'{random}: {description}', why
        reasons.append(f'{random} rejected ({problem})')
        print(f'{random}: rejected ({problem}).')
    return fit, f'{random}: {description}', '; '.join(reasons) + '; last structure kept regardless'


def report_measure(agg: pd.DataFrame, measure: str) -> list[dict]:
    heading(MEASURES[measure], 2)
    data = model_frame(agg, measure)
    print(f"{data['sub_id'].nunique()} participants, {len(data)} participant x modality x segment-class means.")
    fit, structure, why = fit_cascade(data)
    print(f'\nRandom-effects structure fitted: {structure}. Reason: {why}.\n')
    print('Random effects')
    code_block(fit['varcorr'])
    print('\nSatterthwaite type III F-tests\n')
    print('| Term | F | NumDF | DenDF | p |')
    print('|---|---|---|---|---|')
    rows = []
    for _, r in fit['anova'].iterrows():
        print(f"| {TERMS[r['term']]} | {r['F value']:.3f} | {r['NumDF']:.0f} | {r['DenDF']:.1f} | {r['Pr(>F)']:.4f} |")
        rows.append({'measure': measure, 'term': TERMS[r['term']], 'F': r['F value'], 'num_df': r['NumDF'],
                     'den_df': r['DenDF'], 'p': r['Pr(>F)'], 'n_sub': data['sub_id'].nunique(),
                     'random_effects': structure, 'structure_reason': why})
    print('\nFixed effects (factors coded -0.5 / +0.5: simple / complex, tablet / mouse)')
    code_block(fit['coef'].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    return rows


def main() -> None:
    with report('spatial_difficulty_output.md'):
        heading('H7: path deviation by geometric complexity')
        inner, _ = build_flower_geometry()
        complex_segments = classify_segments(inner)

        heading('Data', 2)
        print(describe_handling())
        agg = cell_means()
        print(f"Digital modalities; {agg['sub_id'].nunique()} participants.\n")
        print('Mean path deviation (px) by modality and segment class\n')
        print(agg.groupby(['modality', 'difficulty'])[list(MEASURES)].agg(['mean', 'std', 'count'])
              .round(3).to_markdown())

        rows = report_measure(agg, 'line_rule')
        pd.DataFrame(rows).to_csv(RESULTS / 'spatial_difficulty_summary.csv', index=False)
        plot_bars(agg)
        plot_segments(inner, complex_segments)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_bars(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    colours = {'tablet': '#1f77b4', 'mouse': '#ff7f0e'}
    width = 0.35
    x = np.arange(2)
    for j, mod in enumerate(['tablet', 'mouse']):
        means, sems = [], []
        for diff in ['simple', 'complex']:
            v = agg[(agg['modality'] == mod) & (agg['difficulty'] == diff)]['line_rule'].dropna()
            means.append(v.mean())
            sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.bar(x + (j - 0.5) * width, means, width, yerr=1.96 * np.array(sems), capsize=4,
               label=mod.capitalize(), color=colours[mod])
    ax.set_xticks(x)
    ax.set_xticklabels(['Simple', 'Complex'])
    ax.set_ylabel('Path deviation (pixels)')
    ax.set_xlabel('Segment geometry')
    ax.legend(title='Modality', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / 'spatial_difficulty.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_segments(inner, complex_segments) -> None:
    pts = np.array(inner)
    cumlen = compute_arc_lengths(inner)
    boundaries = np.linspace(0, cumlen[-1], N_SEGMENTS + 1)
    seg_of_vertex = np.clip(np.searchsorted(boundaries[1:], cumlen), 0, N_SEGMENTS - 1)
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    for s in range(N_SEGMENTS):
        m = seg_of_vertex == s
        ax.plot(pts[m, 0], pts[m, 1], '.', color='#d62728' if s in complex_segments else '#7f7f7f', ms=3)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(RESULTS / 'spatial_difficulty_segments.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
