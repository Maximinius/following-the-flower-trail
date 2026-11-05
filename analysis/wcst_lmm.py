"""H6: Wisconsin Card Sorting Task performance as a predictor of Flower Trail errors per minute.

Predictors are total errors and perseveration errors. See ``cognitive_lmm.py``.
"""
from cognitive_lmm import DomainSpec, run_domain

run_domain(DomainSpec(
    name='wcst',
    hypothesis='H6',
    title='Wisconsin Card Sorting Task',
    predictors=['wcst_total_errors', 'wcst_perseverance_errors'],
))
