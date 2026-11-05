"""Measure names used throughout the pipeline.

Each measure has three names:

``column``  the name in the data;
``token``   the identifier used for model terms in the report tables;
``label``   the reader-facing name used in figures and tables, matching the
            manuscript. ``short`` is a compact form for dense plots.

Labels carry no units; units belong in axis captions and table notes.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Measure:
    column: str
    token: str
    label: str
    short: str
    domain: str = ''


# ── Flower Trail Task ─────────────────────────────────────────────────────────
# Reporting order: errors per minute, error length, then the movement measures.
FTT = [
    Measure('mistakes_per_min', 'Errors_Per_Min',  'Errors per minute', 'Errors/min', 'FTT'),
    Measure('ms_mistakes',      'Error_Length',    'Error length',      'Error len.', 'FTT'),
    Measure('deviation',        'Path_Deviation',  'Path deviation',    'Deviation',  'FTT'),
    Measure('speed',            'Movement_Speed',  'Movement speed',    'Speed',      'FTT'),
    Measure('acceleration',     'Acceleration',    'Acceleration',      'Accel.',     'FTT'),
    Measure('stops',            'Number_of_Stops', 'Number of stops',   'Stops',      'FTT'),
    Measure('ms_stopped',       'Time_Stopped',    'Time stopped',      'Time stop.', 'FTT'),
    # Components of the primary outcome.
    Measure('completion_time',  'Completion_Time', 'Completion time',   'Time',       'FTT'),
    Measure('error_count',      'Boundary_Errors', 'Boundary errors',   'Errors',     'FTT'),
]

# Second-scale versions of the millisecond columns, used for model fitting.
FTT_SECONDS = [
    Measure('sec_stopped',  'Time_Stopped', 'Time stopped', 'Time stop.', 'FTT'),
    Measure('sec_mistakes', 'Error_Length', 'Error length', 'Error len.', 'FTT'),
]

# ── Snellgrove Maze Task ──────────────────────────────────────────────────────
# Navigation errors are wrong turns off the solution path (on_path 1 -> 0).
MAZE = [
    Measure('maze_mean_time',             'Maze_Completion_Time',           'Maze completion time',              'Maze time',            'Maze'),
    Measure('maze_mean_mistakes',         'Maze_Navigation_Errors',         'Maze navigation errors',            'Maze nav. errors',     'Maze'),
    Measure('maze_mean_mistakes_per_min', 'Maze_Navigation_Errors_Per_Min', 'Maze navigation errors per minute', 'Maze nav. errors/min', 'Maze'),
]

# ── Eriksen Flanker Task ──────────────────────────────────────────────────────
FLANKER = [
    Measure('flanker_congruency_effect',      'Flanker_Congruency_Effect',      'Congruency effect',      'Congruency',     'Flanker'),
    Measure('flanker_error_rate_incongruent', 'Flanker_Incongruent_Error_Rate', 'Incongruent error rate', 'Incong. errors', 'Flanker'),
]

# ── gradCPT ───────────────────────────────────────────────────────────────────
# City scenes were the go stimulus and mountain scenes the no-go stimulus.
GRADCPT = [
    Measure('gradcpt_rt_city_correct',        'GradCPT_Go_RT',         'Go-trial RT',             'Go RT',         'gradCPT'),
    Measure('gradcpt_rt_mountain_commission', 'GradCPT_Commission_RT', 'Commission-error RT',     'Comm. RT',      'gradCPT'),
    Measure('gradcpt_total_misses',           'GradCPT_Go_Errors',     'Go-trial errors',         'Go errors',     'gradCPT'),
    Measure('gradcpt_mountain_withheld',      'GradCPT_NoGo_Correct',  'Correct no-go responses', 'No-go correct', 'gradCPT'),
]

# ── Wisconsin Card Sorting Task ───────────────────────────────────────────────
# Perseveration errors are scored as consecutive incorrect responses.
WCST = [
    Measure('wcst_total_errors',        'WCST_Total_Errors',        'WCST total errors',    'Total errors',   'WCST'),
    Measure('wcst_perseverance_errors', 'WCST_Perseveration_Proxy', 'Perseveration errors', 'Persev. errors', 'WCST'),
]

ALL_MEASURES = FTT + MAZE + FLANKER + GRADCPT + WCST
_ALL_WITH_ALIASES = ALL_MEASURES + FTT_SECONDS

COLUMN = {m.column: m for m in _ALL_WITH_ALIASES}
TOKEN = {m.column: m.token for m in _ALL_WITH_ALIASES}
LABEL = {m.column: m.label for m in _ALL_WITH_ALIASES}
SHORT = {m.column: m.short for m in _ALL_WITH_ALIASES}
LABEL_BY_TOKEN = {m.token: m.label for m in _ALL_WITH_ALIASES}
SHORT_BY_TOKEN = {m.token: m.short for m in _ALL_WITH_ALIASES}

COGNITIVE = MAZE + FLANKER + GRADCPT + WCST
PREDICTOR_ORDER = [m.column for m in COGNITIVE]

# ── Modality ──────────────────────────────────────────────────────────────────
MODALITY_LABEL = {'paper': 'Paper', 'tablet': 'Tablet', 'mouse': 'Mouse'}
MODALITY_ORDER = ['paper', 'tablet', 'mouse']
DIGITAL_MODALITIES = ['tablet', 'mouse']

CONTRAST_LABEL = {
    'Tablet_vs_Paper': 'Tablet vs. paper',
    'Mouse_vs_Paper':  'Mouse vs. paper',
    'Mouse_vs_Tablet': 'Mouse vs. tablet',
}

# ── Model-term renaming ───────────────────────────────────────────────────────
_STRUCTURAL = {
    "C(modality, Treatment('paper'))[T.tablet]": 'Tablet_vs_Paper',
    "C(modality, Treatment('paper'))[T.mouse]":  'Mouse_vs_Paper',
    "C(modality, Treatment('tablet'))[T.mouse]": 'Mouse_vs_Tablet',
    'tablet_vs_paper': 'Tablet_vs_Paper',
    'mouse_vs_paper':  'Mouse_vs_Paper',
    'mouse_vs_tablet': 'Mouse_vs_Tablet',
    'trial':     'Trial',
    'Intercept': 'Intercept',
    'Group Var': 'Subject_Var',
}

RENAME_DICT = {**TOKEN, **_STRUCTURAL}


def clean_name(name: str) -> str:
    """Map a statsmodels term to its report token.

    Interaction terms (``a:b``), random-effect variances (``x Var``) and
    covariances (``a x b Cov``) are handled by recursing into their parts.
    """
    if name in RENAME_DICT:
        return RENAME_DICT[name]
    if ':' in name:
        return ' × '.join(clean_name(p.strip()) for p in name.split(':'))
    if ' x ' in name and 'Cov' in name:
        target = name.replace('Cov', '').strip()
        parts = [clean_name(p.strip().replace('Group', 'Subj')) for p in target.split(' x ')]
        return ' × '.join(parts) + ' Cov'
    if name.endswith(' Var'):
        return f"{clean_name(name.replace(' Var', '').strip())} Var"
    return name


def label_for(name: str, default=None) -> str:
    """Reader-facing label for a data column or a report token."""
    if name in LABEL:
        return LABEL[name]
    if name in LABEL_BY_TOKEN:
        return LABEL_BY_TOKEN[name]
    return default if default is not None else name.replace('_', ' ').capitalize()


def short_for(name: str, default=None) -> str:
    """Compact label for a data column or a report token."""
    if name in SHORT:
        return SHORT[name]
    if name in SHORT_BY_TOKEN:
        return SHORT_BY_TOKEN[name]
    return default if default is not None else label_for(name)


def qualified_label(column: str) -> str:
    """Label prefixed with its task, for tables that mix cognitive domains."""
    m = COLUMN.get(column)
    if m is None or not m.domain or m.domain == 'FTT':
        return label_for(column)
    if m.label.lower().startswith(m.domain.lower()):
        return m.label
    return f'{m.domain}: {m.label[0].lower()}{m.label[1:]}'
