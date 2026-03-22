import re

with open('task-3-astar_island/query_strategy.py', 'r') as f:
    content = f.read()

content = content.replace('from solution_spatial import _get_calibrated_prior, _initial_terrain_prior', 'from predictor import _get_calibrated_prior, _initial_terrain_prior')
content = content.replace('from solution_spatial import _get_adaptive_tau, _get_prior_mean', 'from predictor import _get_adaptive_tau, _get_prior_mean')

with open('task-3-astar_island/query_strategy.py', 'w') as f:
    f.write(content)
