import re

with open('task-3-astar_island/predictor.py', 'r') as f:
    content = f.read()

# 1. Add TEMP_ALPHA
if 'TEMP_ALPHA' not in content:
    content = re.sub(
        r'FIELD_ALPHA = 1\.0',
        'FIELD_ALPHA = 1.0\nTEMP_ALPHA = 0.95',
        content
    )

# 2. Fix _archetype_backoff_chain
old_backoff = '''def _archetype_backoff_chain(archetype: CellArchetype) -> list[CellArchetype]:
    """Progressively coarser archetypes: full → drop coastal → drop dist → terrain only."""
    t, coast, dist, adj_sett = archetype
    return [
        archetype,
        CellArchetype(t, False, dist, adj_sett),
        CellArchetype(t, False, 3, adj_sett),
        CellArchetype(t, False, 3, False),
    ]'''

new_backoff = '''def _archetype_backoff_chain(archetype: CellArchetype) -> list[CellArchetype]:
    """Progressively coarser archetypes: full → drop coastal → drop dist → terrain only."""
    t, coast, dist, adj_sett = archetype
    chain = []
    for arch in [
        archetype,
        CellArchetype(t, False, dist, adj_sett),
        CellArchetype(t, False, dist, False),
        CellArchetype(t, False, 3, False),
    ]:
        if arch not in chain:
            chain.append(arch)
    return chain'''

content = content.replace(old_backoff, new_backoff)

# 3. Disable spatial smoothing and add temperature scaling
old_predict_end = '''    prediction = apply_floor_and_normalize_grid(prediction, class_masks)
    prediction = _spatial_smooth(prediction, class_masks, grid)
    prediction = apply_correction(prediction, seed_analysis, _correction_table)

    return prediction'''

new_predict_end = '''    prediction = apply_floor_and_normalize_grid(prediction, class_masks)
    prediction = _spatial_smooth(prediction, class_masks, grid, max_beta=0.0)
    prediction = apply_correction(prediction, seed_analysis, _correction_table)

    # Apply temperature scaling
    prediction = np.power(prediction, TEMP_ALPHA)
    prediction = apply_floor_and_normalize_grid(prediction, class_masks)

    return prediction'''

content = content.replace(old_predict_end, new_predict_end)

with open('task-3-astar_island/predictor.py', 'w') as f:
    f.write(content)
