"""Helpers shared by the analysis scripts.

Output location, report logging, the Box-Cox and standardisation steps, lmerTest
fitting and summaries, FDR correction, the VIF screen and the pandas-R data
frame conversion all live here so that every script applies them identically.
"""
from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from measure_names import clean_name

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get('RESULTS_DIR', REPO_ROOT / 'results'))
RESULTS.mkdir(parents=True, exist_ok=True)

MOD_PAPER = "C(modality, Treatment('paper'))"
MOD_TABLET = "C(modality, Treatment('tablet'))"
TABLET_VS_PAPER = f"{MOD_PAPER}[T.tablet]"
MOUSE_VS_PAPER = f"{MOD_PAPER}[T.mouse]"
MOUSE_VS_TABLET = f"{MOD_TABLET}[T.mouse]"

# pymer4 drives lmerTest through rpy2, which needs R_HOME. R is installed inside
# the conda environment, so it is derived from the interpreter.
_r_home = Path(sys.executable).resolve().parent.parent / 'lib' / 'R'
if 'R_HOME' not in os.environ and _r_home.exists():
    os.environ['R_HOME'] = str(_r_home)


# ── Report logging ─────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, handle):
        self.handle = handle

    def write(self, text):
        sys.__stdout__.write(text)
        self.handle.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self.handle.flush()


@contextmanager
def report(filename: str):
    """Mirror everything printed inside the block into RESULTS/filename."""
    original = sys.stdout
    with open(RESULTS / filename, 'w', encoding='utf-8') as handle:
        sys.stdout = _Tee(handle)
        try:
            yield
        finally:
            sys.stdout = original


def heading(text: str, level: int = 1) -> None:
    print(f"\n{'#' * level} {text}\n")


def code_block(obj) -> None:
    print('```')
    print(obj if isinstance(obj, str) else obj.to_string(index=False))
    print('```')


# ── Transforms ─────────────────────────────────────────────────────────────────

def boxcox_fit(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Box-Cox with maximum-likelihood lambda after a positive shift.

    Returns the transformed series, lambda and the shift applied.
    """
    vals = series.dropna().to_numpy(dtype=float)
    shift = max(0.0, -vals.min() + 0.5)
    _, lmbda = stats.boxcox(vals + shift)
    return boxcox_apply(series, lmbda, shift), float(lmbda), float(shift)


def boxcox_apply(series: pd.Series, lmbda: float, shift: float) -> pd.Series:
    out = series.astype(float).copy()
    mask = out.notna()
    x = out[mask].to_numpy() + shift
    out[mask] = np.log(x) if abs(lmbda) < 1e-10 else (x ** lmbda - 1) / lmbda
    return out


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std()
    return (series - series.mean()) / sd if sd > 0 else series


def standardise_predictors(df: pd.DataFrame, preds, subject_stat: str = 'first') -> pd.DataFrame:
    """z-score between-subject predictors on their subject-level values.

    One value per subject enters the mean and SD, so trial repetition does not
    inflate the effective sample size.
    """
    df = df.copy()
    for pred in preds:
        if pred not in df.columns:
            continue
        if subject_stat == 'mean':
            sub_vals = df.groupby('sub_id')[pred].mean()
        else:
            sub_vals = df.drop_duplicates('sub_id')[pred]
        if sub_vals.std() > 0:
            df[pred] = (df[pred] - sub_vals.mean()) / sub_vals.std()
    return df



# ── lmerTest fitting ───────────────────────────────────────────────────────────
#
# Every mixed model is fitted by REML in lmerTest through pymer4. Each fixed
# effect is tested by t = estimate / SE on Satterthwaite degrees of freedom.
# Formulas are written in patsy syntax and terms keep their patsy names.

_CATEGORICAL = re.compile(r"C\((\w+),\s*Treatment\('([^']+)'\)\)")


def _to_lme4(formula: str) -> tuple[str, dict[str, str]]:
    """Rewrite a patsy formula into lme4 syntax.

    ``C(modality, Treatment('paper'))`` becomes a bare ``modality`` term, with
    the reference level recorded so the factor can be releveled to match.
    """
    refs: dict[str, str] = {}

    def take(match: re.Match) -> str:
        var, ref = match.group(1), match.group(2)
        refs[var] = ref
        return var

    return _CATEGORICAL.sub(take, formula), refs


def _term_map(refs: dict[str, str], data: pd.DataFrame) -> dict[str, str]:
    """lme4 coefficient name -> the patsy name for the same term.

    lme4 writes a contrast as ``modalitytablet``; patsy writes it as
    ``C(modality, Treatment('paper'))[T.tablet]``. Interaction terms are joined
    by ':' in both, so mapping the components is enough.
    """
    mapping = {}
    for var, ref in refs.items():
        for level in data[var].astype(str).unique():
            if level != ref:
                mapping[f'{var}{level}'] = f"C({var}, Treatment('{ref}'))[T.{level}]"
    return mapping


class LmerFit:
    """Coefficient table of an lmerTest fit, indexed by patsy term name.

    ``fe_params`` estimates, ``bse`` standard errors, ``tvalues``, ``df``
    (Satterthwaite), ``pvalues`` and ``conf_int()``.
    """

    def __init__(self, model, refs: dict[str, str], data: pd.DataFrame):
        table = model.result_fit.to_pandas()
        rename = _term_map(refs, data)
        index = [rename.get(t, t) if ':' not in t
                 else ':'.join(rename.get(part, part) for part in t.split(':'))
                 for t in table['term']]
        table.index = index
        self.model = model
        self.table = table
        self.fe_params = table['estimate']
        self.params = self.fe_params
        self.bse = table['std_error']
        self.tvalues = table['t_stat']
        self.pvalues = table['p_value']
        self.df = table['df']

    def conf_int(self) -> pd.DataFrame:
        return self.table[['conf_low', 'conf_high']].set_axis([0, 1], axis=1)

    @property
    def r2_marginal(self) -> float:
        """Nakagawa marginal R^2: variance explained by the fixed effects."""
        return float(self.model.result_fit_stats.to_pandas()['R2_marginal'].iloc[0])

    def interaction_f(self) -> dict:
        """Satterthwaite Type III F-test of the highest-order term of this fit.

        lme4 orders interactions after main effects, so this is the last row of
        the ANOVA table.
        """
        self.model.anova(auto_ss_3=False)
        row = self.model.result_anova.to_pandas().iloc[-1]
        return {'F': float(row['F value']), 'num_df': float(row['NumDF']),
                'den_df': float(row['DenDF']), 'p': float(row['Pr(>F)'])}


def lmer_fit(formula: str, data: pd.DataFrame, groups: str = 'sub_id',
             slope: str | None = None) -> LmerFit:
    """Fit ``formula`` (patsy syntax) with lmerTest via pymer4.

    The random part is ``(1 | groups)``, or ``(1 + slope | groups)`` when a
    within-participant term is given as ``slope`` (e.g. ``MOD_PAPER``), so that
    each participant has their own effect of that term.
    """
    import polars as pl
    from pymer4.models import lmer

    fixed, refs = _to_lme4(formula)
    random = '1'
    if slope is not None:
        slope_term, slope_refs = _to_lme4(slope)
        refs.update(slope_refs)
        random = f'1 + {slope_term}'

    frame = data.copy()
    for var in refs:
        frame[var] = frame[var].astype(str)
    model = lmer(f'{fixed} + ({random}|{groups})', data=pl.from_pandas(frame))
    if refs:
        model.set_factors({var: [ref] + sorted(set(frame[var]) - {ref})
                           for var, ref in refs.items()})
    model.fit(summarize=False)
    return LmerFit(model, refs, frame)


# ── Model reporting ────────────────────────────────────────────────────────────

def print_fit(title: str, fit: LmerFit) -> None:
    """Print a fit's coefficient table with model terms renamed to their tokens."""
    view = fit.table[['estimate', 'std_error', 'conf_low', 'conf_high',
                      't_stat', 'df', 'p_value']].copy()
    view.index = [clean_name(name) for name in view.index]
    view.index.name = 'term'
    print(f"\n--- {title} ---\n")
    print('```text')
    print(view.to_string(float_format=lambda v: f'{v:.3f}'))
    print('```')


def stars(p: float) -> str:
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg corrected p values."""
    return multipletests(np.asarray(pvals, dtype=float), method='fdr_bh')[1]


def format_f(res: dict) -> str:
    return f"F({res['num_df']:.0f}, {res['den_df']:.1f}) = {res['F']:.3f}, p = {res['p']:.4f}"


# ── Predictor screening ────────────────────────────────────────────────────────

def vif_table(sub_df: pd.DataFrame, preds) -> pd.DataFrame | None:
    """Variance inflation factors on subject-level complete cases."""
    complete = sub_df[list(preds)].dropna()
    if len(complete) < len(preds) + 2:
        return None
    design = add_constant(complete.to_numpy())
    return pd.DataFrame({
        'predictor': [clean_name(p) for p in preds],
        'column': list(preds),
        'VIF': [variance_inflation_factor(design, i + 1) for i in range(len(preds))],
    })


def prune_vif(preds, sub_df: pd.DataFrame, threshold: float = 5.0):
    """Drop the highest-VIF predictor until every VIF is at or below threshold."""
    preds = list(preds)
    removed = []
    while True:
        table = vif_table(sub_df, preds)
        if table is None:
            break
        worst = table.loc[table['VIF'].idxmax()]
        if worst['VIF'] <= threshold:
            break
        removed.append((worst['predictor'], round(float(worst['VIF']), 2)))
        preds.remove(worst['column'])
    return preds, removed, vif_table(sub_df, preds)


# ── pandas <-> R data frames ────────────────────────────────────────────────────

def _r_frame(frame: pd.DataFrame):
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(frame)


def _py_frame(r_obj) -> pd.DataFrame:
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_obj)
