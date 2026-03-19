"""Static map analysis — precompute BFS tables, neighbors, shelf adjacency.

Built once on round 0, reused every round. The grid is static (walls/shelves
never move), so all-pairs BFS gives O(1) distance + next-hop lookups.
"""

from __future__ import annotations

from collections import deque

from bot.model import Pos

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right


class MapCache:
    __slots__ = (
        "width",
        "height",
        "walls",
        "walkable",
        "neighbors",
        "dist",
        "next_hop",
        "shelf_adj",
        "shelf_cells",
        "drop_zones",
        "drop_zone_basin",
        "narrow_cells",
        "corridor_id",
        "corridor_segments",
    )

    def __init__(
        self,
        width: int,
        height: int,
        walls: set[Pos],
        item_positions: list[Pos],
        drop_off_zones: list[Pos],
    ) -> None:
        self.width = width
        self.height = height
        self.walls = walls
        self.drop_zones = drop_off_zones

        self.walkable: set[Pos] = set()
        self.neighbors: dict[Pos, list[Pos]] = {}
        self.dist: dict[Pos, dict[Pos, int]] = {}
        self.next_hop: dict[Pos, dict[Pos, Pos | None]] = {}
        self.shelf_adj: dict[Pos, list[Pos]] = {}
        self.drop_zone_basin: dict[Pos, int] = {}
        self.narrow_cells: set[Pos] = set()
        self.corridor_id: dict[Pos, int] = {}
        self.corridor_segments: dict[int, set[Pos]] = {}

        self.shelf_cells: set[Pos] = set(item_positions)

        self._build_walkable()
        self._build_neighbors()
        self._build_bfs_tables()
        self._build_shelf_adjacency(item_positions)
        self._build_basins()
        self._build_corridors()

    def _build_walkable(self) -> None:
        blocked = self.walls | self.shelf_cells
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in blocked:
                    self.walkable.add((x, y))

    def _build_neighbors(self) -> None:
        for pos in self.walkable:
            x, y = pos
            nbrs = []
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.walkable:
                    nbrs.append((nx, ny))
            self.neighbors[pos] = nbrs

    def _build_bfs_tables(self) -> None:
        """All-pairs BFS on the static walkable grid."""
        for src in self.walkable:
            dist_map: dict[Pos, int] = {src: 0}
            next_hop_map: dict[Pos, Pos | None] = {src: None}
            q: deque[Pos] = deque([src])
            while q:
                cur = q.popleft()
                d = dist_map[cur]
                for nb in self.neighbors[cur]:
                    if nb not in dist_map:
                        dist_map[nb] = d + 1
                        next_hop_map[nb] = next_hop_map[cur] if cur != src else nb
                        q.append(nb)
            self.dist[src] = dist_map
            self.next_hop[src] = next_hop_map

    def _build_shelf_adjacency(self, item_positions: list[Pos]) -> None:
        """For each shelf cell containing items, find walkable adjacent cells."""
        shelf_cells = set(item_positions)
        for shelf in shelf_cells:
            sx, sy = shelf
            adj = []
            for dx, dy in DIRS:
                nb = (sx + dx, sy + dy)
                if nb in self.walkable:
                    adj.append(nb)
            if adj:
                self.shelf_adj[shelf] = adj

    def _build_basins(self) -> None:
        """Assign each walkable cell to its nearest drop zone (Voronoi)."""
        if len(self.drop_zones) <= 1:
            for pos in self.walkable:
                self.drop_zone_basin[pos] = 0
            return

        for pos in self.walkable:
            best_zone = 0
            best_dist = float("inf")
            for i, dz in enumerate(self.drop_zones):
                d = self.dist.get(pos, {}).get(dz, 9999)
                if d < best_dist:
                    best_dist = d
                    best_zone = i
            self.drop_zone_basin[pos] = best_zone

    def _build_corridors(self) -> None:
        blocked = self.walls | self.shelf_cells
        for pos in self.walkable:
            nbrs = self.neighbors[pos]
            if len(nbrs) != 2:
                continue
            (x1, y1), (x2, y2) = nbrs
            if not (x1 == x2 or y1 == y2):
                continue
            x, y = pos
            if x1 == x2:
                left = (x - 1, y)
                right = (x + 1, y)
                if left in self.shelf_cells and right in self.shelf_cells:
                    self.narrow_cells.add(pos)
            else:
                up = (x, y - 1)
                down = (x, y + 1)
                if up in self.shelf_cells and down in self.shelf_cells:
                    self.narrow_cells.add(pos)

        visited: set[Pos] = set()
        seg_id = 0
        for cell in self.narrow_cells:
            if cell in visited:
                continue
            segment: set[Pos] = set()
            stack = [cell]
            while stack:
                c = stack.pop()
                if c in visited or c not in self.narrow_cells:
                    continue
                visited.add(c)
                segment.add(c)
                for nb in self.neighbors[c]:
                    if nb in self.narrow_cells and nb not in visited:
                        stack.append(nb)
            if segment:
                self.corridor_segments[seg_id] = segment
                for c in segment:
                    self.corridor_id[c] = seg_id
                seg_id += 1

    def distance(self, a: Pos, b: Pos) -> int:
        return self.dist.get(a, {}).get(b, 9999)

    def get_next_hop(self, src: Pos, dst: Pos) -> Pos | None:
        return self.next_hop.get(src, {}).get(dst)

    def best_pickup_cell(self, bot_pos: Pos, item_pos: Pos) -> Pos | None:
        adj_cells = self.shelf_adj.get(item_pos, [])
        if not adj_cells:
            return None
        wide = [c for c in adj_cells if c not in self.narrow_cells]
        candidates = wide if wide else adj_cells
        best = None
        best_d = 9999
        for cell in candidates:
            d = self.distance(bot_pos, cell)
            if d < best_d:
                best_d = d
                best = cell
        return best

    def nearest_drop_zone(self, pos: Pos) -> Pos:
        best = self.drop_zones[0]
        best_d = self.distance(pos, best)
        for dz in self.drop_zones[1:]:
            d = self.distance(pos, dz)
            if d < best_d:
                best_d = d
                best = dz
        return best
