from __future__ import annotations

from bot.batch_planner import BatchPlanner, TripPlan
from bot.coordinator import Coordinator
from bot.map_cache import MapCache
from bot.model import GameState, Pos
from bot.pipeline import PipelinePlanner


class Planner:
    __slots__ = ("mc", "_batch", "_coordinator", "_pipeline")

    def __init__(self) -> None:
        self.mc: MapCache | None = None
        self._batch: BatchPlanner = BatchPlanner()
        self._coordinator: Coordinator = Coordinator()
        self._pipeline: PipelinePlanner = PipelinePlanner()

    def plan(self, state: GameState) -> list[dict]:
        if self.mc is None:
            self._init_map(state)
        assert self.mc is not None

        if len(state.bots) == 1:
            return [self._batch.plan(state, self.mc)]

        return self._pipeline.plan(state, self.mc)

    def _init_map(self, state: GameState) -> None:
        item_positions = [item.position for item in state.items]
        self.mc = MapCache(
            width=state.width,
            height=state.height,
            walls=state.walls,
            item_positions=item_positions,
            drop_off_zones=state.drop_off_zones,
        )

    def get_bot_trip(self, bot_id: int) -> TripPlan | None:
        planner = self._coordinator._planners.get(bot_id)
        return planner._trip if planner else None
