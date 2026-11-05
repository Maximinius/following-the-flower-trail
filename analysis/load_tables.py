"""Read the analysis tables, and the variable lists shared by the models."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

BASE = Path(os.environ.get('FTT_DATA_DIR', Path(__file__).resolve().parent.parent / 'data'))

MODALITIES = ['paper', 'tablet', 'mouse']
DIGITAL = ['tablet', 'mouse']

# Cognitive predictors of the cross-domain joint model.
JOINT_PREDS_FULL = [
    'flanker_congruency_effect', 'flanker_error_rate_incongruent',
    'wcst_total_errors', 'wcst_perseverance_errors',
    'gradcpt_rt_city_correct', 'gradcpt_total_misses', 'gradcpt_mountain_withheld',
    'maze_mean_mistakes_per_min',
]

# Participant means screened for the Mahalanobis replication of models without
# cognitive predictors: errors per minute and the four digital kinematic outcomes.
FTT_SCREEN = ['mistakes_per_min', 'deviation', 'speed', 'stops', 'ms_stopped']


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE / name, float_precision='round_trip')


def load_flower(digital_only: bool = False) -> pd.DataFrame:
    """Trial-level Flower Trail data, for all three modalities or the digital two."""
    levels = DIGITAL if digital_only else MODALITIES
    flower = _read('flower_trials.csv')
    flower = flower[flower['modality'].isin(levels)].copy()
    flower['modality'] = pd.Categorical(flower['modality'], categories=levels, ordered=False)
    return flower


def load_cognitive_battery() -> pd.DataFrame:
    """One row per participant: Flanker, WCST and gradCPT scores."""
    return _read('cognitive_scores.csv')


def load_maze_agg() -> pd.DataFrame:
    """One row per participant: maze means across the ten maze trials."""
    return _read('maze_scores.csv')


def load_h7_means() -> pd.DataFrame:
    """Mean path deviation per participant, modality and segment class."""
    return _read('h7_segment_means.csv')


def build_predictors(verbose: bool = True) -> pd.DataFrame:
    """Cognitive battery outer-merged with the maze means, one row per participant."""
    cog = load_cognitive_battery()
    maze = load_maze_agg()
    merged = cog.merge(maze, on='sub_id', how='outer')
    if verbose:
        both = len(set(cog['sub_id']) & set(maze['sub_id']))
        print(f"Predictors: cognitive battery {cog['sub_id'].nunique()}, maze {maze['sub_id'].nunique()}, "
              f"both {both}, merged {merged['sub_id'].nunique()}")
    return merged
