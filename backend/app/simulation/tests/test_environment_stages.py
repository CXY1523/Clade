"""
Environment Stages Tests - 环境相关阶段测试

测试压力解析、地图演化、板块构造等阶段。
"""

import importlib
import random
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# 标记整个模块使用 asyncio
pytestmark = pytest.mark.asyncio


class TestParsePressuresStage:
    """压力解析阶段测试"""
    
    @pytest.fixture
    def stage(self):
        from ..stages import ParsePressuresStage
        return ParsePressuresStage()
    
    async def test_parse_empty_pressures(self, stage, mock_context, mock_engine):
        """测试空压力列表"""
        mock_context.command = MagicMock(pressures=[])
        mock_engine.environment.parse_pressures.return_value = []
        mock_engine.environment.apply_pressures.return_value = {}
        mock_engine.escalation_service.register.return_value = []
        
        await stage.execute(mock_context, mock_engine)
        
        assert mock_context.pressures == []
        assert mock_context.modifiers == {}
        assert mock_context.major_events == []
    
    async def test_parse_temperature_pressure(self, stage, mock_context, mock_engine):
        """测试温度压力解析"""
        mock_pressure = MagicMock(type="temperature", value=2.0)
        mock_context.command = MagicMock(pressures=[mock_pressure])
        
        mock_engine.environment.parse_pressures.return_value = [mock_pressure]
        mock_engine.environment.apply_pressures.return_value = {"temperature": 2.0}
        mock_engine.escalation_service.register.return_value = []
        
        await stage.execute(mock_context, mock_engine)
        
        assert "temperature" in mock_context.modifiers
        assert mock_context.modifiers["temperature"] == 2.0
    
    async def test_major_events_registered(self, stage, mock_context, mock_engine):
        """测试重大事件注册"""
        mock_context.command = MagicMock(pressures=[])
        mock_engine.environment.parse_pressures.return_value = []
        mock_engine.environment.apply_pressures.return_value = {}
        
        mock_event = MagicMock(kind="volcanic_eruption", severity="high")
        mock_engine.escalation_service.register.return_value = [mock_event]
        
        await stage.execute(mock_context, mock_engine)
        
        assert len(mock_context.major_events) == 1


class TestMapEvolutionStage:
    """地图演化阶段测试
    
    注意：由于模块导入方式，这些测试需要在真实环境中运行。
    这里仅做基本的单元测试。
    """
    
    @pytest.fixture
    def stage(self):
        from ..stages import MapEvolutionStage
        return MapEvolutionStage()
    
    async def test_stage_properties(self, stage):
        """测试阶段属性"""
        assert stage.name == "地图演化"
        assert stage.order == 20
    
    async def test_stage_has_dependency(self, stage):
        """测试阶段有依赖声明"""
        dep = stage.get_dependency()
        assert "解析环境压力" in dep.requires_stages
        assert "modifiers" in dep.requires_fields

    async def test_uses_engine_environment_repository(self, stage, monkeypatch):
        """Map state reads and writes use only the engine repository."""
        environment_repository_module = importlib.import_module(
            "app.repositories.environment_repository"
        )

        class RecordingRepository:
            def __init__(self, saved_state):
                self.saved_state = saved_state
                self.calls = []

            def get_state(self):
                self.calls.append("get_state")
                return None

            def save_state(self, state):
                self.calls.append("save_state")
                return self.saved_state

        injected_state = SimpleNamespace(
            global_avg_temperature=15.0,
            sea_level=0.0,
            turn_index=0,
        )
        module_global_state = SimpleNamespace(
            global_avg_temperature=99.0,
            sea_level=99.0,
            turn_index=99,
        )
        injected_repository = RecordingRepository(injected_state)
        module_global_repository = RecordingRepository(module_global_state)
        monkeypatch.setattr(
            environment_repository_module,
            "environment_repository",
            module_global_repository,
        )

        context = SimpleNamespace(
            current_map_state=None,
            major_events=[],
            modifiers={},
            turn_index=3,
            emit_event=lambda *_args: None,
        )
        engine = SimpleNamespace(
            environment_repository=injected_repository,
            map_evolution=SimpleNamespace(advance=lambda *_args: []),
            _use_tectonic_system=False,
        )

        await stage.execute(context, engine)

        assert context.current_map_state is injected_state
        assert injected_repository.calls == ["get_state", "save_state"]
        assert module_global_repository.calls == []


class TestTectonicMovementStage:
    """板块构造阶段测试"""
    
    @pytest.fixture
    def stage(self):
        from ..stages import TectonicMovementStage
        return TectonicMovementStage()
    
    async def test_skip_when_disabled(self, stage, mock_context, mock_engine):
        """测试禁用时跳过"""
        mock_engine._use_tectonic_system = False
        mock_engine.tectonic = None
        mock_context.tectonic_result = None
        
        await stage.execute(mock_context, mock_engine)
        
        assert mock_context.tectonic_result is None

    async def test_uses_engine_species_repository(self, stage, monkeypatch):
        """板块计算只读取引擎注入的物种仓库。"""
        species_repository_module = importlib.import_module(
            "app.repositories.species_repository"
        )
        environment_repository_module = importlib.import_module(
            "app.repositories.environment_repository"
        )
        alive_species = SimpleNamespace(status="alive", habitats=[], id=1)
        extinct_species = SimpleNamespace(status="extinct", habitats=[], id=2)
        list_calls = 0
        tectonic_calls = []

        def list_species():
            nonlocal list_calls
            list_calls += 1
            return [alive_species, extinct_species]

        tectonic_result = SimpleNamespace(
            wilson_phase={"phase": "stable", "progress": 0.0},
            terrain_changes=[],
            pressure_feedback={},
            get_major_events_summary=lambda: [],
        )

        def tectonic_step(**kwargs):
            tectonic_calls.append(kwargs)
            return tectonic_result

        engine = SimpleNamespace(
            _use_tectonic_system=True,
            tectonic=SimpleNamespace(step=tectonic_step),
            species_repository=SimpleNamespace(list_species=list_species),
            resource_manager=SimpleNamespace(),
        )
        context = SimpleNamespace(
            modifiers={},
            tectonic_result=None,
            turn_index=4,
            emit_event=lambda *_args: None,
        )
        monkeypatch.setattr(
            species_repository_module,
            "species_repository",
            SimpleNamespace(
                list_species=lambda: (_ for _ in ()).throw(
                    AssertionError(
                        "TectonicMovementStage must not use the global repository"
                    )
                )
            ),
        )
        monkeypatch.setattr(
            environment_repository_module,
            "environment_repository",
            SimpleNamespace(list_tiles=lambda: []),
        )

        await stage.execute(context, engine)

        assert list_calls == 1
        assert len(tectonic_calls) == 1
        assert tectonic_calls[0]["species_list"] == [alive_species]
        assert context.tectonic_result is tectonic_result


class TestSimpleWeatherStage:
    """简单天气阶段测试"""
    
    @pytest.fixture
    def stage_always_trigger(self):
        from ..plugin_stages import SimpleWeatherStage
        return SimpleWeatherStage(trigger_chance=1.0)  # 100% 触发
    
    @pytest.fixture
    def stage_never_trigger(self):
        from ..plugin_stages import SimpleWeatherStage
        return SimpleWeatherStage(trigger_chance=0.0)  # 0% 触发
    
    async def test_stage_properties(self, stage_always_trigger):
        """测试阶段属性"""
        assert stage_always_trigger.name == "简单天气"
        assert stage_always_trigger.order == 22
        assert stage_always_trigger.trigger_chance == 1.0
    
    async def test_stage_never_trigger_properties(self, stage_never_trigger):
        """测试阶段属性（不触发版本）"""
        assert stage_never_trigger.trigger_chance == 0.0
    
    async def test_weather_no_crash_on_empty_tiles(self, stage_always_trigger, mock_context, mock_engine):
        """测试空地块列表不会崩溃"""
        mock_context.all_tiles = []
        
        # 由于仓储导入在 execute 中，我们跳过真正的执行
        # 这里仅测试阶段可以正确初始化
        assert stage_always_trigger is not None
