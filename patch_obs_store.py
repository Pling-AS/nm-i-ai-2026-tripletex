import re

with open('task-3-astar_island/observation_store.py', 'r') as f:
    content = f.read()

# Add settlement_vitality to __init__
old_init = '''        self._archetype_obs_count: dict[CellArchetype, int] = defaultdict(int)'''
new_init = '''        self._archetype_obs_count: dict[CellArchetype, int] = defaultdict(int)
        self.settlement_vitality: dict | None = None'''
content = content.replace(old_init, new_init)

# Add to save
old_save = '''        meta = {
            "seeds_count": self.seeds_count,
            "height": self.height,
            "width": self.width,
            "archetype_pools": arch_data,
        }'''
new_save = '''        meta = {
            "seeds_count": self.seeds_count,
            "height": self.height,
            "width": self.width,
            "archetype_pools": arch_data,
            "settlement_vitality": getattr(self, "settlement_vitality", None),
        }'''
content = content.replace(old_save, new_save)

# Add to load
old_load = '''        for key_str, pool in meta["archetype_pools"].items():'''
new_load = '''        store.settlement_vitality = meta.get("settlement_vitality")
        for key_str, pool in meta["archetype_pools"].items():'''
content = content.replace(old_load, new_load)

with open('task-3-astar_island/observation_store.py', 'w') as f:
    f.write(content)
