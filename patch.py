with open('/Users/m/Programmering/2025/nm-i-ai/task-3-astar_island/query_strategy.py', 'r') as f:
    content = f.read()

content = content.replace('marginal_value = prior_entropy / ((planned_n + 1.0) ** 2)', 'marginal_value = prior_entropy / (planned_n + 1.0)')

with open('/Users/m/Programmering/2025/nm-i-ai/task-3-astar_island/query_strategy.py', 'w') as f:
    f.write(content)
