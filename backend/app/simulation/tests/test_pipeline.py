"""
Pipeline Tests - 流水线测试

测试 Pipeline 执行器、StageLoader 和模式切换功能。
"""

import asyncio
import ast
import inspect
import textwrap
from unittest.mock import MagicMock, AsyncMock

import pytest

from ..pipeline import Pipeline, PipelineConfig, PipelineBuilder, PipelineResult
from ..stage_config import StageLoader, stage_registry, AVAILABLE_MODES
from ..ecological_realism_stage import EcologicalRealismStage
from ..stages import (
    AutoHybridizationStage,
    BaseStage,
    BuildReportStage,
    EmbeddingPluginsStage,
    EmbeddingStage,
    FetchSpeciesStage,
    GeneActivationStage,
    InitStage,
    SpeciationStage,
    StageDependency,
    get_default_stages,
)
from ..context import SimulationContext

# 标记整个模块使用 asyncio
pytestmark = pytest.mark.asyncio


# ============================================================================
# Test Stages
# ============================================================================

class SimpleTestStage(BaseStage):
    """简单测试阶段"""
    
    def __init__(self, order: int = 10, name: str = "测试阶段"):
        super().__init__(order=order, name=name)
        self.executed = False
        self.execution_count = 0
    
    async def execute(self, ctx, engine):
        self.executed = True
        self.execution_count += 1
        ctx._test_stage_executed = True


class FailingTestStage(BaseStage):
    """失败的测试阶段"""
    
    def __init__(self, order: int = 20, name: str = "失败阶段"):
        super().__init__(order=order, name=name)
    
    async def execute(self, ctx, engine):
        raise RuntimeError("故意失败")


class DependentTestStage(BaseStage):
    """有依赖的测试阶段"""
    
    def __init__(self, order: int = 30, name: str = "依赖阶段"):
        super().__init__(order=order, name=name)
    
    def get_dependency(self) -> StageDependency:
        return StageDependency(
            requires_stages={"测试阶段"},
            requires_fields={"_test_stage_executed"},
            writes_fields={"_dependent_result"},
        )
    
    async def execute(self, ctx, engine):
        if not getattr(ctx, "_test_stage_executed", False):
            raise RuntimeError("依赖未满足")
        ctx._dependent_result = True


class InternallyBudgetedTestStage(BaseStage):
    """A stage whose remote request owns its complete deadline."""

    uses_internal_request_budget = True

    def __init__(self, delay: float = 0.02):
        super().__init__(order=40, name="internally-budgeted")
        self.delay = delay
        self.completed = False

    async def execute(self, ctx, engine):
        await asyncio.sleep(self.delay)
        self.completed = True


class SlowBusinessTestStage(BaseStage):
    def __init__(self, delay: float = 0.02):
        super().__init__(order=50, name="slow-business")
        self.delay = delay
        self.cancelled = False

    async def execute(self, ctx, engine):
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestPipeline:
    """Pipeline 执行器测试"""
    
    @pytest.fixture
    def simple_stages(self):
        return [SimpleTestStage(10, "阶段1"), SimpleTestStage(20, "阶段2")]
    
    @pytest.fixture
    def mock_ctx(self):
        ctx = SimulationContext(turn_index=0)
        ctx.command = MagicMock(pressures=[], rounds=1)
        return ctx
    
    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine._use_embedding_integration = False
        return engine
    
    async def test_execute_all_stages(self, simple_stages, mock_ctx, mock_engine):
        """测试执行所有阶段"""
        config = PipelineConfig(validate_dependencies=False)
        pipeline = Pipeline(simple_stages, config)
        
        result = await pipeline.execute(mock_ctx, mock_engine)
        
        assert result.success
        assert len(result.stage_results) == 2
        for stage in simple_stages:
            assert stage.executed
    
    async def test_stage_order_preserved(self, mock_ctx, mock_engine):
        """测试阶段顺序保持"""
        stages = [
            SimpleTestStage(30, "阶段C"),
            SimpleTestStage(10, "阶段A"),
            SimpleTestStage(20, "阶段B"),
        ]
        config = PipelineConfig(validate_dependencies=False)
        pipeline = Pipeline(stages, config)
        
        result = await pipeline.execute(mock_ctx, mock_engine)
        
        # 验证按顺序执行
        assert result.stage_results[0].stage_name == "阶段A"
        assert result.stage_results[1].stage_name == "阶段B"
        assert result.stage_results[2].stage_name == "阶段C"
    
    async def test_continue_on_error(self, mock_ctx, mock_engine):
        """测试错误时继续"""
        stages = [
            SimpleTestStage(10, "正常阶段1"),
            FailingTestStage(20, "失败阶段"),
            SimpleTestStage(30, "正常阶段2"),
        ]
        config = PipelineConfig(
            continue_on_error=True,
            validate_dependencies=False,
        )
        pipeline = Pipeline(stages, config)
        
        result = await pipeline.execute(mock_ctx, mock_engine)
        
        assert not result.success
        assert len(result.failed_stages) == 1
        assert "失败阶段" in result.failed_stages
        # 第三个阶段仍然执行
        assert stages[2].executed
    
    async def test_stop_on_error(self, mock_ctx, mock_engine):
        """测试错误时停止"""
        stages = [
            SimpleTestStage(10, "正常阶段1"),
            FailingTestStage(20, "失败阶段"),
            SimpleTestStage(30, "正常阶段2"),
        ]
        config = PipelineConfig(
            continue_on_error=False,
            validate_dependencies=False,
        )
        pipeline = Pipeline(stages, config)
        
        result = await pipeline.execute(mock_ctx, mock_engine)
        
        assert not result.success
        # 第三个阶段不应执行
        assert not stages[2].executed
    
    async def test_timing_metrics(self, simple_stages, mock_ctx, mock_engine):
        """测试时间统计"""
        config = PipelineConfig(
            log_timing=True,
            validate_dependencies=False,
        )
        pipeline = Pipeline(simple_stages, config)
        
        result = await pipeline.execute(mock_ctx, mock_engine)
        
        assert result.metrics is not None
        assert result.metrics.total_duration_ms >= 0
        assert len(result.metrics.stage_metrics) == 2
    
    async def test_partial_execution_only_stage(self, mock_ctx, mock_engine):
        """测试只执行单个阶段"""
        stages = [
            SimpleTestStage(10, "阶段A"),
            SimpleTestStage(20, "阶段B"),
            SimpleTestStage(30, "阶段C"),
        ]
        config = PipelineConfig(
            only_stage="阶段B",
            validate_dependencies=False,
        )
        pipeline = Pipeline(stages, config)
        
        result = await pipeline.execute(mock_ctx, mock_engine)
        
        assert len(result.stage_results) == 1
        assert result.stage_results[0].stage_name == "阶段B"
        assert not stages[0].executed
        assert stages[1].executed
        assert not stages[2].executed

    async def test_internal_request_budget_stage_is_not_cancelled_by_stage_timeout(
        self,
        mock_ctx,
        mock_engine,
    ):
        stage = InternallyBudgetedTestStage()
        pipeline = Pipeline(
            [stage],
            PipelineConfig(stage_timeout=0.001, validate_dependencies=False),
        )

        result = await pipeline.execute(mock_ctx, mock_engine)

        assert result.success
        assert stage.completed is True

    async def test_business_stage_still_uses_generic_stage_timeout(
        self,
        mock_ctx,
        mock_engine,
    ):
        stage = SlowBusinessTestStage()
        pipeline = Pipeline(
            [stage],
            PipelineConfig(stage_timeout=0.001, validate_dependencies=False),
        )

        result = await pipeline.execute(mock_ctx, mock_engine)

        assert result.success is False
        assert stage.cancelled is True
        assert isinstance(result.stage_results[0].error, asyncio.TimeoutError)


async def test_all_remote_request_stages_declare_internal_budget_ownership() -> None:
    remote_request_stages = [
        InitStage,
        FetchSpeciesStage,
        GeneActivationStage,
        AutoHybridizationStage,
        SpeciationStage,
        BuildReportStage,
        EmbeddingStage,
        EmbeddingPluginsStage,
        EcologicalRealismStage,
    ]

    assert BaseStage.uses_internal_request_budget is False
    assert all(
        stage_type.uses_internal_request_budget is True
        for stage_type in remote_request_stages
    )


@pytest.mark.parametrize("stage_type", [SpeciationStage, BuildReportStage])
async def test_ai_stage_has_no_local_wait_for_wrapper(
    stage_type: type[BaseStage],
) -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(stage_type.execute)))
    wait_for_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
        and node.func.attr == "wait_for"
    ]

    assert wait_for_calls == []


class TestPipelineBuilder:
    """PipelineBuilder 测试"""
    
    def test_build_pipeline(self):
        """测试构建流水线"""
        stages = [SimpleTestStage(10, "测试")]
        
        pipeline = (
            PipelineBuilder()
            .add_stages(stages)
            .continue_on_error(True)
            .log_timing(True)
            .build()
        )
        
        assert pipeline is not None
        assert len(pipeline.stages) == 1


# ============================================================================
# StageLoader Tests
# ============================================================================

class TestStageLoader:
    """StageLoader 测试"""
    
    def test_list_available_stages(self):
        """测试列出可用阶段"""
        loader = StageLoader()
        stages = loader.list_available_stages()
        
        assert len(stages) > 0
        assert "init" in stages
        assert "parse_pressures" in stages
    
    def test_load_stages_for_mode(self):
        """测试按模式加载阶段"""
        loader = StageLoader()
        
        for mode in AVAILABLE_MODES:
            try:
                stages = loader.load_stages_for_mode(mode, validate=False)
                # 验证返回的是阶段列表
                assert isinstance(stages, list)
                # 验证每个阶段都有 execute 方法
                for stage in stages:
                    assert hasattr(stage, 'execute')
            except Exception as e:
                pytest.fail(f"加载模式 {mode} 失败: {e}")
    
    def test_get_dependency_graph(self):
        """测试获取依赖图"""
        loader = StageLoader()
        graph = loader.get_dependency_graph("standard")
        
        assert "Stage 依赖关系图" in graph


# ============================================================================
# Stage Registry Tests
# ============================================================================

class TestStageRegistry:
    """StageRegistry 测试"""
    
    def test_registered_stages(self):
        """测试 GPU-only 架构下的核心阶段注册"""
        expected_stages = [
            "init",
            "parse_pressures",
            "map_evolution",
            "fetch_species",
            "tensor_ecology",
            "population_update",
        ]
        legacy_stages = [
            "preliminary_mortality",
            "migration",
            "final_mortality",
        ]

        for stage_name in expected_stages:
            stage_class = stage_registry.get(stage_name)
            assert stage_class is not None, f"阶段 {stage_name} 未注册"

        for stage_name in legacy_stages:
            assert stage_registry.get(stage_name) is None, f"旧阶段 {stage_name} 不应注册"
    
    def test_create_stage_instance(self):
        """测试创建阶段实例"""
        stage = stage_registry.create_stage("init")
        
        assert stage is not None
        assert hasattr(stage, 'execute')
        assert stage.name == "回合初始化"


# ============================================================================
# Default Stages Tests
# ============================================================================

class TestDefaultStages:
    """默认阶段测试"""
    
    def test_get_default_stages(self):
        """测试获取默认阶段列表"""
        stages = get_default_stages()
        
        assert len(stages) > 0
        # 验证按顺序排列
        orders = [s.order for s in stages]
        assert orders == sorted(orders)
    
    def test_default_stages_have_execute(self):
        """测试默认阶段都有 execute 方法"""
        stages = get_default_stages()
        
        for stage in stages:
            assert callable(getattr(stage, 'execute', None)), \
                f"阶段 {stage.name} 没有 execute 方法"

