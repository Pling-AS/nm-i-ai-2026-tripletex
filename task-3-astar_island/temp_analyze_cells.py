import json
import numpy as np
from client import AstarClient
from features import SeedAnalysis
from utils import NUM_CLASSES

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > eps
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))

def entropy(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    mask = p > eps
    if not mask.any():
        return 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))

def analyze_cells(round_id, seed_index):
    client = AstarClient()
    detail = client.get_round_detail(round_id)
    state = detail.initial_states[seed_index]
    seed_analysis = SeedAnalysis(state["grid"], state["settlements"])
    
    data = client.get_analysis(round_id, seed_index)
    gt = np.array(data["ground_truth"], dtype=np.float64)
    pred = np.array(data["prediction"], dtype=np.float64)
    
    h, w = gt.shape[:2]
    cells = []
    for y in range(h):
        for x in range(w):
            gt_cell = gt[y, x]
            pred_cell = pred[y, x]
            cell_kl = kl_divergence(gt_cell, pred_cell)
            cell_entropy = entropy(gt_cell)
            weighted_kl = cell_entropy * cell_kl
            archetype = seed_analysis.get_archetype(y, x)
            dist_s = float(seed_analysis.dist_to_settlement[y, x])
            dist_r = float(seed_analysis.dist_to_ruin[y, x])
            terrain = int(seed_analysis.grid[y, x])
            cells.append({
                'y': y, 'x': x, 'kl': cell_kl, 'entropy': cell_entropy, 'weighted_kl': weighted_kl,
                'archetype': str(archetype), 'terrain': terrain, 'dist_s': dist_s, 'dist_r': dist_r,
                'gt': gt_cell.tolist(), 'pred': pred_cell.tolist()
            })
    return cells

round_id = '3eb0c25d-28fa-48ca-b8e1-fc249e3918e9'
all_cells = []
for seed in range(5):
    cells = analyze_cells(round_id, seed)
    for cell in cells:
        cell['seed'] = seed
    all_cells.extend(cells)

# Sort by KL descending
all_cells.sort(key=lambda c: c['kl'], reverse=True)

# Print top 20 worst cells
print("Top 20 worst cells by KL:")
for i, cell in enumerate(all_cells[:20]):
    print(f"{i+1}: Seed {cell['seed']}, ({cell['y']},{cell['x']}), KL={cell['kl']:.4f}, Entropy={cell['entropy']:.4f}, Terrain={cell['terrain']}, DistS={cell['dist_s']:.1f}, DistR={cell['dist_r']:.1f}")
    print(f"    Archetype: {cell['archetype']}")
    print(f"    GT: {cell['gt']}")
    print(f"    Pred: {cell['pred']}")
    print()
