"""H4: Eriksen Flanker Task performance as a predictor of Flower Trail errors per minute.

Predictors are the congruency effect (incongruent minus congruent mean RT) and
the incongruent-trial error rate. See ``cognitive_lmm.py``.
"""
from cognitive_lmm import DomainSpec, run_domain

run_domain(DomainSpec(
    name='flanker',
    hypothesis='H4',
    title='Eriksen Flanker Task',
    predictors=['flanker_congruency_effect', 'flanker_error_rate_incongruent'],
))
