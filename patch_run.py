with open('/Users/m/Programmering/2025/nm-i-ai/task-3-astar_island/run.py', 'r') as f:
    content = f.read()

content = content.replace(
    "prediction = predict_full_grid_vectorized(\n            seed_idx, observation_store, seed_analyses[seed_idx]\n        )",
    "prediction = predict_full_grid_vectorized(\n            seed_idx, observation_store, seed_analyses\n        )"
)

with open('/Users/m/Programmering/2025/nm-i-ai/task-3-astar_island/run.py', 'w') as f:
    f.write(content)
