"""Collect SI Appendix Tables S1 to S10 from the outputs of a pipeline run.

Each table is stored with its title, header, rows, note and layout in
``<results>/si_tables/si_tables.json``; the tables named in ``EXPORT`` are also
written there as .csv.
"""
import json

import pandas as pd

from common import RESULTS as R, fdr
from load_tables import load_flower
from measure_names import LABEL, qualified_label
from participant_level_checks import OUTCOMES, descriptives, speed_accuracy

OUT = R / 'si_tables'
EXPORT = ['S1', 'S6', 'S7', 'S8', 'S9', 'S10']

PRED_ORDER = ['maze_mean_time', 'maze_mean_mistakes', 'maze_mean_mistakes_per_min',
              'flanker_congruency_effect', 'flanker_error_rate_incongruent',
              'gradcpt_rt_city_correct', 'gradcpt_rt_mountain_commission', 'gradcpt_total_misses',
              'gradcpt_mountain_withheld', 'wcst_total_errors', 'wcst_perseverance_errors']
OUT_ORDER = ['mistakes_per_min', 'completion_time', 'error_count', 'ms_mistakes', 'deviation', 'speed',
             'acceleration', 'stops', 'ms_stopped']
CONTRAST = {'Tablet_vs_Paper': 'Tablet vs. paper', 'Mouse_vs_Paper': 'Mouse vs. paper',
            'Mouse_vs_Tablet': 'Mouse vs. tablet'}
MOD = {'paper': 'Paper', 'tablet': 'Tablet', 'mouse': 'Mouse'}
NOTE_Z = ('Outcomes were Box-Cox transformed where skewed and z-scored, and predictors z-scored at the participant '
          'level, so β is in standard-deviation units. ')


def p(v):
    if v is None or pd.isna(v):
        return '–'
    return '< 0.001' if v < 0.001 else f'{v:.3f}'


def f2(v):
    return '–' if v is None or pd.isna(v) else f'{v:.2f}'.replace('-', '−')


def df_(v):
    return '–' if pd.isna(v) else (f'{v:.0f}' if v >= 100 else f'{v:.1f}')


def pred(c):
    """Measure label with its task, e.g. 'Flanker congruency effect'."""
    return qualified_label(c).replace(': ', ' ')


def lab(c):
    return LABEL.get(c, c)


def spec(title, header, rows, note, widths, group_col=None, size=8):
    return dict(title=title, header=header, rows=rows, note=note, widths=widths, group_col=group_col, size=size)


# ── Tables ────────────────────────────────────────────────────────────────────

def s1():
    m = pd.read_csv(R / 'modality_lmm_summary.csv')
    rows = [[r.family.capitalize(), lab(r.outcome), CONTRAST[r.contrast], f2(r.beta), f2(r.se), f2(r.t), df_(r.df),
             p(r.p_fdr), str(int(r.n_sub)), str(int(r.n_obs))] for _, r in m.iterrows()]
    return spec('Modality contrasts',
                ['Family', 'Outcome', 'Contrast', 'β', 'SE', '*t*', 'df', '*P* (FDR)', '*n* participants',
                 '*n* trials'], rows,
                NOTE_Z + 'Trial number was a covariate in every model. *t* tests use Satterthwaite degrees of '
                'freedom. *P* values are FDR-corrected within the confirmatory family (11 contrasts) and the '
                'exploratory family (4 contrasts). All contrasts were fitted with a by-participant random slope '
                'for modality.', [1.0, 1.4, 1.2, 0.6, 0.6, 0.7, 0.6, 0.8, 1.0, 0.8], group_col=1)


def s2(flower, means):
    rows = [[r.outcome, MOD[r.modality], str(int(r.n)), f'{r["mean"]:.2f}', f'{r["median"]:.2f}', f'{r.sd:.2f}']
            for _, r in descriptives(means).iterrows()]
    units = {'ms_mistakes': 'ms', 'deviation': 'pixels', 'speed': 'pixels per s', 'acceleration': 'pixels per s²',
             'stops': 'count', 'ms_stopped': 'ms'}
    dig = flower[flower.modality.isin(['tablet', 'mouse'])].groupby(['sub_id', 'modality'], observed=True)[
        list(units)].mean()
    for col, u in units.items():
        for mod in ['tablet', 'mouse']:
            v = dig.xs(mod, level='modality')[col].dropna()
            rows.append([f'{LABEL[col]} ({u})', MOD[mod], str(len(v)), f'{v.mean():.2f}', f'{v.median():.2f}',
                         f'{v.std():.2f}'])
    return spec('Participant-level descriptive statistics', ['Measure', 'Version', '*n*', 'Mean', 'Median', '*SD*'],
                rows, 'Statistics are computed over participant means. Errors per minute and boundary errors are '
                'per trial; completion time is in s.', [2.6, 1.0, 0.6, 0.9, 0.9, 0.9], group_col=0)


def s3():
    b = pd.read_csv(R / 'bayesian_modality_summary.csv')
    cc = pd.read_csv(R / 'bayesian_modality_complete_case.csv')
    rows = [['Mixed model', f'{MOD[r.A]} vs. {MOD[r.B]}', f'{int(r.n_sub)} ({int(r.n_both)} with both)',
             f'{r["diff"]:.1f} [{r.diff_lo:.1f}, {r.diff_hi:.1f}]'.replace('-', '−'),
             f'{r.delta:.2f} [{r.delta_lo:.2f}, {r.delta_hi:.2f}]'.replace('-', '−'), f'{r.BF10:.1f}']
            for _, r in b.iterrows()]
    rank = {'paper': 0, 'tablet': 1, 'mouse': 2}
    paired = []
    for _, r in cc.iterrows():
        if rank[r.A] > rank[r.B]:
            r = r.copy()
            r['A'], r['B'], r['T'], r['hedges'] = r.B, r.A, -r['T'], -r.hedges
        paired.append(['Paired t test, complete triplets', f'{MOD[r.A]} vs. {MOD[r.B]}', str(int(r.n)),
                       f't[{r.dof:.0f}] = {r["T"]:.2f}'.replace('-', '−'), f'g = {r.hedges:.2f}'.replace('-', '−'),
                       f'{r.BF10:.1f}'])
    rows += sorted(paired, key=lambda x: [rank[w.lower()] for w in x[1].split(' vs. ')])
    return spec('Bayesian comparisons of errors per minute between versions',
                ['Analysis', 'Comparison', '*n*', 'Difference [95% CrI] or test', 'δ [95% CrI] or effect size',
                 'BF_10_'], rows,
                'Mixed model: BayesFactor lmBF on participant means, random participant intercept, JZS prior '
                'r = ½; the difference is the posterior mean difference in errors per minute (first minus second '
                'version) and δ its standardised size. Paired t tests: Pingouin, JZS prior r = √2/2, on the '
                'participants with all three versions; *g*, Hedges\' *g*.', [2.2, 1.3, 1.3, 1.9, 1.9, 0.8],
                group_col=0)


def s4():
    pc = pd.read_csv(R / 'power_check.csv')
    pc = pc[pc.kind == 'correlation']
    rel = pd.read_csv(R / 'participant_level_checks_summary.csv')
    if 'p_fdr' not in rel:
        rel['p_fdr'] = fdr(rel['p_rho'])
    rel = rel.set_index(['outcome', 'pair'])
    rows = []
    for _, r in pc.iterrows():
        a, b = r.pair.split('-')
        rows.append([r.outcome, f'{MOD[a]}–{MOD[b]}', str(int(r.n)), f'{r.rho:.2f}',
                     p(rel.loc[(r.outcome, r.pair)].p_fdr), f'{r.power_prereg_r:.2f}'])
    reliab = []
    for o in OUTCOMES.values():
        sub = rel.xs(o, level='outcome')
        vals = [sub.loc['paper-tablet', 'rel_a'], sub.loc['paper-tablet', 'rel_b'], sub.loc['paper-mouse', 'rel_b']]
        reliab.append(f'{o.lower()} ' + ', '.join(f'{v:.2f}' for v in vals))
    return spec('Agreement between versions and power against the pre-registered analysis',
                ['Outcome', 'Versions', '*n*', 'Correlation (*r*_s_)', '*P*', 'Power to detect *r*_s_ = 0.38'], rows,
                '*r*_s_, Spearman correlation of participant means. *P*, two-tailed *P* value of *r*_s_, '
                'FDR-corrected (Benjamini–Hochberg) across the nine correlations. Power is for the available *n* '
                'and uses the Fisher *z* approximation of the pre-registration (two-tailed α = 0.05). The '
                'pre-registered analysis assumed *r*_s_ = 0.38 and required *N* = 53. Split-half reliabilities '
                '(odd against even trials, Spearman–Brown corrected; paper, tablet, mouse): '
                + '; '.join(reliab) + '.', [1.6, 1.2, 0.6, 1.4, 1.5, 1.6], group_col=0)


def s5(means):
    rows = [[MOD[r.modality], str(int(r.n)), f'{r.rho:.2f}'.replace('-', '−'), p(r.p_fdr)]
            for _, r in speed_accuracy(means).iterrows()]
    return spec('Speed–accuracy trade-off within each version', ['Version', '*n*', 'Correlation (*r*_s_)', '*P*'],
                rows, 'Spearman correlation between participant-mean completion time and participant-mean boundary '
                'errors. *P* values are FDR-corrected (Benjamini–Hochberg) across the three versions.',
                [1.2, 0.8, 1.4, 1.4])


def s6():
    rows = []
    for dom, fname in [('Maze', 'maze'), ('Flanker', 'flanker'), ('gradCPT', 'gradcpt'), ('WCST', 'wcst')]:
        d = pd.read_csv(R / f'{fname}_lmm_summary.csv')
        for c in [x for x in PRED_ORDER if x in set(d.predictor)]:
            m = d[(d.block == 'confirmatory') & (d.predictor == c)].iloc[0]
            it = d[(d.block == 'interaction') & (d.predictor == c)].iloc[0]
            rows.append([dom, pred(c), str(int(m.n_sub)), f2(m.beta), f2(m.se), f'{f2(m.t)} [{df_(m.df)}]',
                         p(m.p_fdr), f'*F*[{it.num_df:.0f}, {df_(it.den_df)}] = {it.F:.2f}', p(it.p_F)])
    return spec('Single-domain cognitive models of errors per minute (H3 to H6)',
                [['', '', '', ['Main effect', 4], ['Interaction with modality', 2]],
                 ['Task', 'Measure', '*n*', 'β', 'SE', '*t* [df]', '*P*', '*F* [df]', '*P*']], rows,
                NOTE_Z + 'Each measure was entered in its own model with modality and trial as covariates and a '
                'random participant intercept. Main-effect *P* values are FDR-corrected within task. Interactions '
                'are type III Satterthwaite *F* tests from models with a by-participant random slope for modality.',
                [0.8, 2.4, 0.5, 0.6, 0.6, 1.1, 0.7, 1.6, 0.7], group_col=0)


def s7():
    j = pd.read_csv(R / 'joint_lmm_summary.csv')
    rows = [[pred(r.predictor), f2(r.beta), f2(r.se), f'{f2(r.t)} [{df_(r.df)}]', p(r.p_fdr)]
            for _, r in j.iterrows()]
    n = int(j.n_sub.iloc[0])
    return spec('Joint model of errors per minute', ['Measure', 'β', 'SE', '*t* [df]', '*P*'], rows,
                NOTE_Z + f'All measures were entered together with modality and trial as covariates (*n* = {n}). '
                '*P* values are FDR-corrected across the measures. gradCPT commission-error RT was not entered '
                'because it is undefined without commission errors, and WCST perseveration errors were removed for '
                'collinearity.', [2.8, 0.7, 0.7, 1.2, 0.8]), n


def s8():
    e = pd.read_csv(R / 'joint_lmm_exploratory_interactions.csv')
    e = e[e.spec == 'slope']
    rows = []
    for o in [x for x in OUT_ORDER if x in set(e.outcome)]:
        for fam in ['executive', 'maze']:
            g = e[(e.outcome == o) & (e.family == fam)].set_index('predictor')
            for c in [x for x in PRED_ORDER if x in g.index]:
                s = g.loc[c]
                rows.append([lab(o), fam.capitalize(), pred(c), f2(s.beta), f2(s.se), f'{f2(s.t)} [{df_(s.df)}]',
                             p(s.p_fdr)])
    return spec('Cognition-by-device interactions on the digital outcomes (mouse relative to tablet)',
                ['Outcome', 'Family', 'Measure', 'β', 'SE', '*t* [df]', '*P*'], rows,
                NOTE_Z + 'Measures of each family entered each model jointly, with a by-participant random slope for '
                'modality. β is the interaction: the change in the measure\'s slope from tablet to mouse. *P* values '
                'are FDR-corrected within family and outcome.', [1.5, 1.0, 2.6, 0.7, 0.7, 1.2, 0.7], group_col=1)


def s9():
    w = pd.read_csv(R / 'standalone_modality_summary.csv')
    w = w.assign(mo=w.modality.map({'paper': 0, 'tablet': 1, 'mouse': 2}),
                 oo=w.outcome.map({o: i for i, o in enumerate(OUT_ORDER)}),
                 po=w.predictor.map({x: i for i, x in enumerate(PRED_ORDER)}))
    rows = [[MOD[r.modality], lab(r.outcome), pred(r.predictor), str(int(r.n_sub)),
             f'{f2(r.beta)} [{f2(r.ci_lo)}, {f2(r.ci_hi)}]', f'{f2(r.t)} [{df_(r.df)}]', p(r.p_bh), f'{r.dR2m:.3f}']
            for _, r in w.sort_values(['mo', 'oo', 'po']).iterrows()]
    return spec('Cognitive measures and FTT outcomes within each version',
                ['Version', 'Outcome', 'Measure', '*n*', 'β [95% CI]', '*t* [df]', '*P*', 'ΔR²_m_'], rows,
                NOTE_Z + 'Each model related one measure to one outcome within one version, with trial as a covariate '
                'and a random participant intercept. Outcomes were transformed on data pooled across versions so '
                'that slopes share one scale. *P* values are FDR-corrected within task and version. ΔR²_m_, gain in '
                'marginal R² from adding the measure.', [0.8, 1.4, 2.6, 0.5, 1.5, 1.1, 0.7, 0.7], group_col=1,
                size=7.5)


def s10():
    sd = pd.read_csv(R / 'spatial_difficulty_summary.csv')
    sd = sd[sd.measure == 'line_rule']
    rows = [[r.term, f'*F*[{r.num_df:.0f}, {df_(r.den_df)}] = {r.F:.2f}', p(r.p)] for _, r in sd.iterrows()]
    structure = sd.random_effects.iloc[0].split(':')[0].replace('mod', 'modality').replace('seg', 'segment class')
    return spec('Path deviation by segment complexity (H7)', ['Term', 'Test', '*P*'], rows,
                'Participant-by-segment-class mean path deviation on the tablet and the mouse, the measure of the H2 '
                'contrast (1-px line rule, Box-Cox transformed and z-scored), modelled with segment class, modality '
                f'and their interaction (each coded −0.5/+0.5). Random effects: {structure.replace("sub_id", "participant")}. '
                'Type III Satterthwaite *F* tests.', [2.6, 1.6, 0.8])


def plain(text):
    return text.replace('*', '').replace('_', '')


def write_csv(name, t):
    header = t['header'][-1] if isinstance(t['header'][0], list) else t['header']
    rows = [[plain(v) for v in row] for row in t['rows']]
    pd.DataFrame(rows, columns=[plain(h) for h in header]).to_csv(OUT / f'table_{name}.csv', index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flower = load_flower()
    means = flower.groupby(['sub_id', 'modality'], observed=True)[list(OUTCOMES)].mean().unstack()
    tables = {'S1': s1(), 'S2': s2(flower, means), 'S3': s3(), 'S4': s4(), 'S5': s5(means), 'S6': s6()}
    tables['S7'], n_joint = s7()
    tables.update({'S8': s8(), 'S9': s9(), 'S10': s10()})
    json.dump({'tables': tables, 'S7n': n_joint}, open(OUT / 'si_tables.json', 'w'), ensure_ascii=False, indent=1)
    for name in EXPORT:
        write_csv(name, tables[name])
    print('wrote', OUT, {k: len(v['rows']) for k, v in tables.items()})


if __name__ == '__main__':
    main()
