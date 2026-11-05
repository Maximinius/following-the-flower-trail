"""H3: Snellgrove Maze Task performance as a predictor of Flower Trail errors per minute.

Predictors are the participant's mean maze completion time and mean number of
maze navigation errors. Maze navigation errors per minute is not entered alongside them
because of its collinearity with the two components. See ``cognitive_lmm.py``.
"""
from cognitive_lmm import DomainSpec, run_domain
from load_tables import load_maze_agg

run_domain(DomainSpec(
    name='maze',
    hypothesis='H3',
    title='Snellgrove Maze Task',
    predictors=['maze_mean_time', 'maze_mean_mistakes'],
    load_predictors=load_maze_agg,
))
