import re

with open('task-3-astar_island/utils.py', 'r') as f:
    content = f.read()

content = re.sub(r'FLOOR_EPS = 0\.001', 'FLOOR_EPS = 0.01', content)

with open('task-3-astar_island/utils.py', 'w') as f:
    f.write(content)
