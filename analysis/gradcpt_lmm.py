"""H5: gradCPT performance as a predictor of Flower Trail errors per minute.

Predictors are go-trial RT, commission-error RT, go-trial errors and correct
no-go responses. Commission-error RT is missing for participants who made no
commission errors, so its models use fewer participants. See ``cognitive_lmm.py``.
"""
from cognitive_lmm import DomainSpec, run_domain

run_domain(DomainSpec(
    name='gradcpt',
    hypothesis='H5',
    title='gradCPT',
    predictors=['gradcpt_rt_city_correct', 'gradcpt_rt_mountain_commission',
                'gradcpt_total_misses', 'gradcpt_mountain_withheld'],
))
