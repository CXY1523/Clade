"""Regression coverage for fixed-seed simulation repeatability."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Session, create_engine

from ...core import database
from ...models.environment import MapState
from ...schemas.requests import TurnCommand
from ...services.geo.map_evolution import MapEvolutionService
from ..engine import SimulationEngine
from ..pipeline import Pipeline, PipelineConfig
from ..stages import FinalizeStage, MapEvolutionStage, ParsePressuresStage


async def _run_seeded_core_scenario(monkeypatch, seed: int) -> tuple[tuple, ...]:
    """Run the real core map pipeline against a fresh isolated database."""

    test_database_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr(database, "engine", test_database_engine)
    SQLModel.metadata.create_all(test_database_engine)

    random.seed(seed)
    np.random.seed(seed)
    map_evolution = MapEvolutionService(width=8, height=4)

    with Session(test_database_engine) as session:
        session.add(
            MapState(
                id=1,
                turn_index=0,
                sea_level=0.0,
                global_avg_temperature=15.0,
                stage_name=map_evolution.current_stage_name,
                stage_progress=0,
                stage_duration=map_evolution.stage_duration,
            )
        )
        session.commit()

    simulation_engine = SimulationEngine.__new__(SimulationEngine)
    simulation_engine.turn_counter = 0
    simulation_engine._event_callback = None
    simulation_engine.environment = SimpleNamespace(
        parse_pressures=lambda pressures: pressures,
        apply_pressures=lambda pressures: {"temperature": 1.0},
    )
    simulation_engine.escalation_service = SimpleNamespace(
        register=lambda pressures, turn_index: [],
    )
    simulation_engine.map_evolution = map_evolution
    simulation_engine.map_manager = SimpleNamespace(
        reclassify_terrain_by_sea_level=lambda sea_level: None,
    )
    simulation_engine._use_tectonic_system = False
    simulation_engine._pipeline = Pipeline(
        [ParsePressuresStage(), MapEvolutionStage(), FinalizeStage()],
        PipelineConfig(
            continue_on_error=False,
            emit_stage_events=False,
            validate_dependencies=False,
        ),
    )

    trace = []
    for _ in range(20):
        await simulation_engine.run_turn_with_pipeline(TurnCommand(rounds=1))
        with Session(test_database_engine) as session:
            state = session.get(MapState, 1)
            assert state is not None
            trace.append(
                (
                    simulation_engine.turn_counter,
                    state.turn_index,
                    state.stage_name,
                    state.stage_progress,
                    state.stage_duration,
                    state.sea_level,
                    state.global_avg_temperature,
                )
            )

    test_database_engine.dispose()
    return tuple(trace)


@pytest.mark.asyncio
async def test_fixed_seed_repeats_core_pipeline_results(monkeypatch):
    """The same seed and initial state must produce the same core turn trace."""

    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()
    try:
        first_run = await _run_seeded_core_scenario(monkeypatch, seed=42)
        repeated_run = await _run_seeded_core_scenario(monkeypatch, seed=42)
        different_seed_run = await _run_seeded_core_scenario(monkeypatch, seed=43)
    finally:
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)

    assert repeated_run == first_run
    assert different_seed_run != first_run
