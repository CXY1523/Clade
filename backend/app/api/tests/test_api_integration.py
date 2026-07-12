"""
API 集成测试 - 验证路由和依赖注入

测试目标：
1. 验证新路由 (api/router.py) 与 Depends 注入正确工作
2. 验证 USE_LEGACY_ROUTES=true 时仍能启动
3. 验证 ConfigService 缓存和配置加载
4. 验证 ResourceManager 注入
5. 验证 SessionManager 状态管理
"""

import sys

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


# ============================================================================
# 容器和服务 Mock
# ============================================================================

@pytest.fixture
def mock_container():
    """创建模拟服务容器"""
    container = MagicMock()
    
    # ConfigService
    container.config_service = MagicMock()
    container.config_service.get_ui_config = MagicMock(return_value=MagicMock(
        food_web=MagicMock(),
        ecology_balance=MagicMock(),
        mortality=MagicMock(),
        speciation=MagicMock(),
        resource_system=MagicMock(),
    ))
    container.config_service.get_ecology_balance = MagicMock(return_value=MagicMock())
    container.config_service.get_mortality = MagicMock(return_value=MagicMock())
    container.config_service.get_speciation = MagicMock(return_value=MagicMock())
    
    # ResourceManager
    container.resource_manager = MagicMock()
    container.resource_manager.get_snapshot = MagicMock(return_value=MagicMock(
        total_npp=1000.0,
        avg_npp=100.0,
        overgrazing_tiles=0,
    ))
    
    # Repositories
    container.species_repository = MagicMock()
    container.species_repository.list_species = MagicMock(return_value=[])
    container.species_repository.get_by_lineage = MagicMock(return_value=None)
    
    container.environment_repository = MagicMock()
    container.history_repository = MagicMock()
    container.history_repository.list_turns = MagicMock(return_value=[])
    
    container.genus_repository = MagicMock()
    
    # Services
    container.save_manager = MagicMock()
    container.save_manager.list_saves = MagicMock(return_value=[])
    
    container.export_service = MagicMock()
    container.model_router = MagicMock()
    container.embedding_service = MagicMock()
    container.species_generator = MagicMock()
    
    # SimulationEngine
    container.simulation_engine = MagicMock()
    container.simulation_engine.turn_counter = 0
    
    # EnvironmentSystem
    container.environment_system = MagicMock()
    
    return container


@pytest.fixture
def mock_session():
    """创建模拟会话管理器"""
    session = MagicMock()
    session.session_id = "test-session-12345678"
    session.is_running = False
    session.current_save_name = None
    session.autosave_counter = 0
    session.pressure_queue = []
    session.get_pending_events = MagicMock(return_value=[])
    session.push_event = MagicMock()
    session.set_running = MagicMock()
    session.pop_pressure = MagicMock(return_value=None)
    return session


# ============================================================================
# 新路由测试
# ============================================================================

class TestNewRouterIntegration:
    """测试新版拆分路由 (api/router.py)"""
    
    @pytest.fixture
    def client(self, mock_container, mock_session):
        """创建使用新路由的测试客户端
        
        现在依赖注入使用 app.state，所以需要直接设置 app.state
        """
        from ...main import app
        
        # 设置 app.state（模拟 lifespan 的行为）
        app.state.container = mock_container
        app.state.session = mock_session
        
        yield TestClient(app)
        
        # 清理
        if hasattr(app.state, 'container'):
            del app.state.container
        if hasattr(app.state, 'session'):
            del app.state.session
    
    def test_health_check(self, client):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_api_health_check(self, client, mock_session):
        """测试 API 健康检查端点"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "session_id" in data
    
    def test_species_list(self, client, mock_container):
        """测试物种列表端点"""
        mock_container.species_repository.list_species.return_value = []
        response = client.get("/api/species/list")
        assert response.status_code == 200
    
    def test_saves_list(self, client, mock_container):
        """测试存档列表端点"""
        mock_container.save_manager.list_saves.return_value = []
        response = client.get("/api/saves/list")
        assert response.status_code == 200
        assert response.json() == []
    
    def test_history_list(self, client, mock_container):
        """测试历史记录端点"""
        mock_container.history_repository.list_turns.return_value = []
        response = client.get("/api/history")
        assert response.status_code == 200

    def test_get_config_redacts_provider_key(self, client, mock_container):
        from ...models.config import ProviderConfig, UIConfig

        mock_container.config_service.get_ui_config.return_value = UIConfig(
            providers={
                "main": ProviderConfig(
                    id="main", name="Main", api_key="sk-secret"
                )
            }
        )

        response = client.get("/api/config/ui")

        assert response.status_code == 200
        body = response.json()
        assert body["providers"]["main"]["api_key"] == ""
        assert body["providers"]["main"]["api_key_configured"] is True
        assert "sk-secret" not in response.text

    def test_post_config_validation_error_does_not_echo_nested_key(self, client):
        secret = "sk-validation-secret"

        response = client.post(
            "/api/config/ui",
            json={
                "config": {
                    "providers": [
                        {"id": "main", "name": "Main", "api_key": secret}
                    ]
                }
            },
        )

        assert response.status_code == 422
        assert response.json() == {"detail": "Invalid configuration payload"}
        assert secret not in response.text

    def test_post_config_non_object_body_does_not_echo_body(self, client):
        secret = "sk-non-object-secret"

        response = client.post("/api/config/ui", json=secret)

        assert response.status_code == 422
        assert response.json() == {"detail": "Invalid configuration payload"}
        assert secret not in response.text

    @pytest.mark.parametrize(
        ("body", "secret"),
        [
            (b"null", None),
            (b"", None),
            (b'{"api_key":"sk-bad-json-secret"', "sk-bad-json-secret"),
        ],
        ids=["null", "empty", "invalid-json"],
    )
    def test_post_config_unparseable_body_uses_safe_422(
        self, client, body, secret
    ):
        response = client.post(
            "/api/config/ui",
            content=body,
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 422
        assert response.json() == {"detail": "Invalid configuration payload"}
        assert '"input"' not in response.text
        assert '"ctx"' not in response.text
        if secret:
            assert secret not in response.text

    def test_post_config_openapi_documents_update_envelope(self, client):
        document = client.get("/openapi.json").json()
        request_body = document["paths"]["/api/config/ui"]["post"]["requestBody"]
        schema = request_body["content"]["application/json"]["schema"]

        def local_definition_refs(value):
            if isinstance(value, dict):
                reference = value.get("$ref")
                if isinstance(reference, str) and reference.startswith("#/$defs/"):
                    yield reference
                for nested in value.values():
                    yield from local_definition_refs(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from local_definition_refs(nested)

        assert request_body["required"] is True
        assert set(schema.get("properties", {})) >= {
            "config",
            "clear_provider_api_keys",
        }
        assert schema.get("title") != "Raw Request"
        assert list(local_definition_refs(schema)) == []
        config_schema = schema["properties"]["config"]
        assert set(config_schema.get("properties", {})) >= {
            "providers",
            "capability_routes",
        }

    def test_openapi_schema_inlining_rejects_recursive_definitions(self):
        from ..analytics import _inline_local_schema_definitions

        recursive_schema = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "child": {"$ref": "#/$defs/Node"},
                    },
                },
            },
            "$ref": "#/$defs/Node",
        }

        with pytest.raises(ValueError, match="Recursive schema reference"):
            _inline_local_schema_definitions(recursive_schema)

    def test_openapi_schema_inlining_allows_sibling_definition_reuse(self):
        from ..analytics import _inline_local_schema_definitions

        schema = {
            "$defs": {"Value": {"type": "string"}},
            "$ref": "#/$defs/Value",
            "not": {"$ref": "#/$defs/Value"},
        }

        assert _inline_local_schema_definitions(schema) == {
            "allOf": [{"type": "string"}],
            "not": {"type": "string"},
        }

    def test_openapi_schema_inlining_removes_definitions_from_ref_root(self):
        from ..analytics import _inline_local_schema_definitions

        schema = {
            "$defs": {"Value": {"type": "string"}},
            "$ref": "#/$defs/Value",
            "title": "Value",
        }

        expanded = _inline_local_schema_definitions(schema)

        assert expanded == {
            "allOf": [{"type": "string"}],
            "title": "Value",
        }

    @pytest.mark.parametrize(
        "content_type",
        [
            "text/plain",
            None,
            "application/*+json",
            "application/ +json",
        ],
        ids=["text-plain", "missing", "wildcard-json", "blank-subtype-json"],
    )
    def test_post_config_rejects_non_json_media_type(
        self, client, mock_container, content_type
    ):
        from ...models.config import UIConfig

        mock_container.config_service.get_ui_config.return_value = UIConfig()
        mock_container.settings.ui_config_path = "data/test-settings.json"
        mock_container.environment_repository.save_ui_config.side_effect = (
            lambda _path, config: config
        )
        runtime_routes = MagicMock()
        headers = {"content-type": content_type} if content_type else {}

        with (
            patch("app.api.analytics.configure_model_router"),
            patch.dict(sys.modules, {"app.api.routes": runtime_routes}),
        ):
            response = client.post(
                "/api/config/ui",
                content=b'{"config":{"providers":{}}}',
                headers=headers,
            )

        assert response.status_code == 422
        assert response.json() == {"detail": "Invalid configuration payload"}
        assert "providers" not in response.text
        mock_container.environment_repository.save_ui_config.assert_not_called()

    @pytest.mark.parametrize(
        "content_type",
        [
            "application/json; charset=utf-8",
            "application/problem+json; charset=utf-8",
            "Application/Problem+Json; Charset=UTF-8",
        ],
        ids=["application-json", "structured-json", "mixed-case-structured-json"],
    )
    def test_post_config_accepts_json_media_type_with_parameters(
        self, client, mock_container, content_type
    ):
        from ...models.config import UIConfig

        mock_container.config_service.get_ui_config.return_value = UIConfig()
        mock_container.settings.ui_config_path = "data/test-settings.json"
        mock_container.environment_repository.save_ui_config.side_effect = (
            lambda _path, config: config
        )
        runtime_routes = MagicMock()

        with (
            patch("app.api.analytics.configure_model_router"),
            patch.dict(sys.modules, {"app.api.routes": runtime_routes}),
        ):
            response = client.post(
                "/api/config/ui",
                content=b'{"config":{"providers":{}}}',
                headers={"content-type": content_type},
            )

        assert response.status_code == 200
        mock_container.environment_repository.save_ui_config.assert_called_once()

    def test_post_config_preserves_empty_key(self, client, mock_container):
        from ...models.config import ProviderConfig, UIConfig

        current = UIConfig(
            providers={
                "main": ProviderConfig(
                    id="main", name="Main", api_key="sk-secret"
                )
            }
        )
        mock_container.config_service.get_ui_config.return_value = current
        mock_container.settings.ui_config_path = "data/test-settings.json"
        mock_container.environment_repository.save_ui_config.side_effect = (
            lambda _path, config: config
        )
        runtime_routes = MagicMock()
        refresh_order = []
        mock_container.config_service.invalidate_cache.side_effect = (
            lambda: refresh_order.append("cache")
        )
        runtime_routes.apply_ui_config.side_effect = (
            lambda _saved: refresh_order.append("apply")
        )
        runtime_routes.simulation_engine.reload_configs.side_effect = (
            lambda _configs: refresh_order.append("reload")
        )

        with (
            patch(
                "app.api.analytics.configure_model_router",
                side_effect=lambda *_args: refresh_order.append("configure"),
            ) as configure_router,
            patch.dict(sys.modules, {"app.api.routes": runtime_routes}),
        ):
            response = client.post(
                "/api/config/ui",
                json={
                    "config": {
                        "providers": {
                            "main": {"id": "main", "name": "Main", "api_key": ""}
                        }
                    },
                    "clear_provider_api_keys": [],
                },
            )

        assert response.status_code == 200
        saved = mock_container.environment_repository.save_ui_config.call_args.args[1]
        assert saved.providers["main"].api_key == "sk-secret"
        configure_router.assert_called_once_with(
            saved,
            mock_container.model_router,
            mock_container.embedding_service,
            mock_container.settings,
        )
        runtime_routes.apply_ui_config.assert_called_once_with(saved)
        runtime_routes.simulation_engine.reload_configs.assert_called_once()
        mock_container.config_service.invalidate_cache.assert_called_once_with()
        assert refresh_order == ["cache", "configure", "apply", "reload"]
        assert "sk-secret" not in response.text

    def test_post_config_save_failure_does_not_refresh_runtime(
        self, client, mock_container
    ):
        from ...models.config import UIConfig

        mock_container.config_service.get_ui_config.return_value = UIConfig()
        mock_container.settings.ui_config_path = "data/test-settings.json"
        mock_container.environment_repository.save_ui_config.side_effect = RuntimeError(
            "save failed"
        )
        runtime_routes = MagicMock()

        with (
            patch("app.api.analytics.configure_model_router") as configure_router,
            patch.dict(sys.modules, {"app.api.routes": runtime_routes}),
            pytest.raises(RuntimeError, match="save failed"),
        ):
            client.post(
                "/api/config/ui",
                json={"config": {"providers": {}}, "clear_provider_api_keys": []},
            )

        mock_container.config_service.invalidate_cache.assert_not_called()
        configure_router.assert_not_called()
        runtime_routes.apply_ui_config.assert_not_called()
        runtime_routes.simulation_engine.reload_configs.assert_not_called()

    def test_post_config_cache_error_does_not_stop_runtime_refresh(
        self, client, mock_container, caplog
    ):
        from ...models.config import ProviderConfig, UIConfig

        current = UIConfig(
            providers={
                "main": ProviderConfig(
                    id="main", name="Main", api_key="sk-secret"
                )
            }
        )
        mock_container.config_service.get_ui_config.return_value = current
        mock_container.settings.ui_config_path = "data/test-settings.json"
        mock_container.environment_repository.save_ui_config.side_effect = (
            lambda _path, config: config
        )
        runtime_routes = MagicMock()
        refresh_order = []

        class CacheRefreshError(RuntimeError):
            pass

        def fail_cache():
            refresh_order.append("cache")
            raise CacheRefreshError("cache failed for sk-secret")

        mock_container.config_service.invalidate_cache.side_effect = fail_cache
        runtime_routes.apply_ui_config.side_effect = (
            lambda _saved: refresh_order.append("apply")
        )
        runtime_routes.simulation_engine.reload_configs.side_effect = (
            lambda _configs: refresh_order.append("reload")
        )

        with (
            patch(
                "app.api.analytics.configure_model_router",
                side_effect=lambda *_args: refresh_order.append("configure"),
            ),
            patch.dict(sys.modules, {"app.api.routes": runtime_routes}),
        ):
            response = client.post(
                "/api/config/ui",
                json={
                    "config": {
                        "providers": {
                            "main": {"id": "main", "name": "Main", "api_key": ""}
                        }
                    },
                    "clear_provider_api_keys": [],
                },
            )

        assert response.status_code == 200
        assert refresh_order == ["cache", "configure", "apply", "reload"]
        assert "CacheRefreshError" in caplog.text
        assert "sk-secret" not in response.text
        assert "sk-secret" not in caplog.text

    def test_post_config_runtime_errors_do_not_log_provider_key(
        self, client, mock_container, caplog
    ):
        from ...models.config import ProviderConfig, UIConfig

        current = UIConfig(
            providers={
                "main": ProviderConfig(
                    id="main", name="Main", api_key="sk-secret"
                )
            }
        )
        mock_container.config_service.get_ui_config.return_value = current
        mock_container.settings.ui_config_path = "data/test-settings.json"
        mock_container.environment_repository.save_ui_config.side_effect = (
            lambda _path, config: config
        )
        runtime_routes = MagicMock()
        refresh_order = []
        mock_container.config_service.invalidate_cache.side_effect = (
            lambda: refresh_order.append("cache")
        )

        def fail_refresh(stage):
            def fail(*_args):
                refresh_order.append(stage)
                raise RuntimeError(f"{stage} failed for sk-secret")

            return fail

        runtime_routes.apply_ui_config.side_effect = fail_refresh("apply")
        runtime_routes.simulation_engine.reload_configs.side_effect = fail_refresh(
            "reload"
        )

        with (
            patch(
                "app.api.analytics.configure_model_router",
                side_effect=fail_refresh("configure"),
            ),
            patch.dict(sys.modules, {"app.api.routes": runtime_routes}),
        ):
            response = client.post(
                "/api/config/ui",
                json={
                    "config": {
                        "providers": {
                            "main": {"id": "main", "name": "Main", "api_key": ""}
                        }
                    },
                    "clear_provider_api_keys": [],
                },
            )

        assert response.status_code == 200
        assert refresh_order == ["cache", "configure", "apply", "reload"]
        assert caplog.text.count("RuntimeError") == 3
        assert "sk-secret" not in response.text
        assert "sk-secret" not in caplog.text

    def test_api_connection_uses_stored_credentials_without_echoing_them(
        self, client, mock_container, caplog
    ):
        from ...models.config import ProviderConfig, UIConfig

        mock_container.config_service.get_ui_config.return_value = UIConfig(
            providers={
                "main": ProviderConfig(
                    id="main",
                    name="Main",
                    base_url="https://stored.example/v1",
                    api_key="sk-secret",
                )
            }
        )
        upstream_response = MagicMock(
            status_code=401,
            text="upstream echoed sk-secret",
        )
        http_client = MagicMock()
        http_client.__enter__.return_value = http_client
        http_client.get.return_value = upstream_response

        with patch("httpx.Client", return_value=http_client):
            response = client.post(
                "/api/config/test-api", json={"provider_id": "main"}
            )

        assert response.status_code == 200
        assert response.json()["success"] is False
        http_client.get.assert_called_once_with(
            "https://stored.example/v1/models",
            headers={"Authorization": "Bearer sk-secret"},
        )
        assert "sk-secret" not in response.text
        assert "sk-secret" not in caplog.text

    def test_fetch_models_uses_stored_credentials(self, client, mock_container):
        from ...models.config import ProviderConfig, UIConfig

        mock_container.config_service.get_ui_config.return_value = UIConfig(
            providers={
                "main": ProviderConfig(
                    id="main",
                    name="Main",
                    base_url="https://stored.example/v1",
                    api_key="sk-secret",
                )
            }
        )
        upstream_response = MagicMock()
        upstream_response.status_code = 200
        upstream_response.json.return_value = {"data": [{"id": "model-1"}]}
        http_client = MagicMock()
        http_client.__enter__.return_value = http_client
        http_client.get.return_value = upstream_response

        with patch("httpx.Client", return_value=http_client):
            response = client.post(
                "/api/config/fetch-models", json={"provider_id": "main"}
            )

        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "models": [{"id": "model-1", "name": "model-1"}],
        }
        http_client.get.assert_called_once_with(
            "https://stored.example/v1/models",
            headers={"Authorization": "Bearer sk-secret"},
        )
        assert "sk-secret" not in response.text

    def test_fetch_models_does_not_echo_key_from_exception(
        self, client, mock_container, caplog
    ):
        from ...models.config import UIConfig

        mock_container.config_service.get_ui_config.return_value = UIConfig()
        http_client = MagicMock()
        http_client.__enter__.return_value = http_client
        http_client.get.side_effect = RuntimeError("request rejected for sk-unsaved")

        with patch("httpx.Client", return_value=http_client):
            response = client.post(
                "/api/config/fetch-models",
                json={
                    "base_url": "https://new.example/v1",
                    "api_key": "sk-unsaved",
                },
            )

        assert response.status_code == 200
        assert response.json()["success"] is False
        assert "sk-unsaved" not in response.text
        assert "sk-unsaved" not in caplog.text


# 旧路由兼容性测试已移除 - api/routes.py 已删除


# ============================================================================
# ConfigService 契约测试
# ============================================================================

class TestConfigServiceContract:
    """ConfigService 缓存和配置加载契约测试"""
    
    @pytest.fixture
    def config_service(self):
        """创建 ConfigService 实例"""
        from ...core.config_service import ConfigService
        from ...core.config import get_settings
        
        settings = get_settings()
        return ConfigService(settings)
    
    def test_config_service_initialization(self, config_service):
        """测试 ConfigService 初始化"""
        assert config_service is not None
    
    def test_config_service_caching(self, config_service):
        """测试配置缓存行为"""
        # 首次调用应该加载配置
        config1 = config_service.get_ui_config()
        
        # 第二次调用应该返回缓存的配置
        config2 = config_service.get_ui_config()
        
        # 应该是同一个对象（缓存）
        assert config1 is config2
    
    def test_config_service_ecology_balance(self, config_service):
        """测试生态平衡配置获取"""
        from ...models.config import EcologyBalanceConfig
        
        config = config_service.get_ecology_balance()
        
        assert isinstance(config, EcologyBalanceConfig)
    
    def test_config_service_mortality(self, config_service):
        """测试死亡率配置获取"""
        from ...models.config import MortalityConfig
        
        config = config_service.get_mortality()
        
        assert isinstance(config, MortalityConfig)


# ============================================================================
# ResourceManager 契约测试
# ============================================================================

class TestResourceManagerContract:
    """ResourceManager 注入契约测试"""
    
    def test_resource_manager_via_dependency(self, mock_container):
        """测试 ResourceManager 通过依赖注入获取"""
        from ...main import app
        
        # 设置 app.state
        app.state.container = mock_container
        
        client = TestClient(app)
        
        # 通过 API 访问（间接测试依赖注入）
        # ResourceManager 应该可以通过容器获取
        assert mock_container.resource_manager is not None
        
        # 清理
        del app.state.container
    
    def test_resource_manager_from_container_directly(self):
        """测试直接从容器获取 ResourceManager"""
        from ...core.container import ServiceContainer
        from ...services.ecology.resource_manager import ResourceManager
        
        container = ServiceContainer()
        
        # 容器应该能提供 ResourceManager
        manager = container.resource_manager
        assert isinstance(manager, ResourceManager)


# ============================================================================
# SessionManager 契约测试
# ============================================================================

class TestSessionManagerContract:
    """SessionManager 状态管理契约测试"""
    
    def test_session_manager_initialization(self):
        """测试 SessionManager 初始化"""
        from ...core.session import SimulationSessionManager
        
        session = SimulationSessionManager()
        assert session.is_running == False
        assert session.current_save_name is None
    
    def test_session_manager_running_state(self):
        """测试运行状态管理"""
        from ...core.session import SimulationSessionManager
        
        session = SimulationSessionManager()
        
        session.set_running(True)
        assert session.is_running == True
        
        session.set_running(False)
        assert session.is_running == False
    
    def test_session_manager_save_name(self):
        """测试存档名称管理"""
        from ...core.session import SimulationSessionManager
        
        session = SimulationSessionManager()
        
        session.set_save_name("test_save")
        assert session.current_save_name == "test_save"
    
    def test_session_manager_event_queue(self):
        """测试事件队列管理"""
        from ...core.session import SimulationSessionManager
        
        session = SimulationSessionManager()
        
        session.push_event("test", "Test message", "category")
        events = session.get_pending_events(max_count=10)
        
        assert len(events) == 1
        assert events[0]["type"] == "test"


# ============================================================================
# 依赖注入契约测试
# ============================================================================

class TestDependencyInjectionContract:
    """依赖注入契约测试
    
    现在 get_container/get_session 需要 Request 参数并从 app.state 获取。
    """
    
    def test_get_container_dependency(self, mock_container):
        """测试 get_container 从 app.state 获取容器"""
        from ..dependencies import get_container
        from ...main import app
        
        # 设置 app.state
        app.state.container = mock_container
        
        # 创建 mock request
        mock_request = MagicMock()
        mock_request.app = app
        
        container = get_container(mock_request)
        assert container is mock_container
        
        # 清理
        del app.state.container
    
    def test_get_session_dependency(self, mock_session):
        """测试 get_session 从 app.state 获取会话"""
        from ..dependencies import get_session
        from ...main import app
        
        # 设置 app.state
        app.state.session = mock_session
        
        # 创建 mock request
        mock_request = MagicMock()
        mock_request.app = app
        
        session = get_session(mock_request)
        assert session is mock_session
        
        # 清理
        del app.state.session
    
    def test_require_not_running_passes(self, mock_session, mock_container):
        """测试 require_not_running 在未运行时通过"""
        from ..dependencies import require_not_running
        from fastapi import HTTPException
        from ...main import app
        
        mock_session.is_running = False
        app.state.session = mock_session
        app.state.container = mock_container
        
        # 创建 mock request
        mock_request = MagicMock()
        mock_request.app = app
        
        # 不应该抛出异常
        require_not_running(mock_request, mock_session)
        
        # 清理
        del app.state.session
        del app.state.container
    
    def test_require_not_running_fails(self, mock_session, mock_container):
        """测试 require_not_running 在运行时失败"""
        from ..dependencies import require_not_running
        from fastapi import HTTPException
        from ...main import app
        
        mock_session.is_running = True
        app.state.session = mock_session
        app.state.container = mock_container
        
        # 创建 mock request
        mock_request = MagicMock()
        mock_request.app = app
        
        with pytest.raises(HTTPException) as exc_info:
            require_not_running(mock_request, mock_session)
        
        assert exc_info.value.status_code == 400
        
        # 清理
        del app.state.session
        del app.state.container
    
    def test_require_save_loaded_passes(self, mock_session, mock_container):
        """测试 require_save_loaded 在有存档时通过"""
        from ..dependencies import require_save_loaded
        from ...main import app
        
        mock_session.current_save_name = "test_save"
        app.state.session = mock_session
        app.state.container = mock_container
        
        # 创建 mock request
        mock_request = MagicMock()
        mock_request.app = app
        
        save_name = require_save_loaded(mock_request, mock_session)
        assert save_name == "test_save"
        
        # 清理
        del app.state.session
        del app.state.container
    
    def test_require_save_loaded_fails(self, mock_session, mock_container):
        """测试 require_save_loaded 在无存档时失败"""
        from ..dependencies import require_save_loaded
        from fastapi import HTTPException
        from ...main import app
        
        mock_session.current_save_name = None
        app.state.session = mock_session
        app.state.container = mock_container
        
        # 创建 mock request
        mock_request = MagicMock()
        mock_request.app = app
        
        with pytest.raises(HTTPException) as exc_info:
            require_save_loaded(mock_request, mock_session)
        
        assert exc_info.value.status_code == 400
        
        # 清理
        del app.state.session
        del app.state.container


# ============================================================================
# 配置加载函数契约测试
# ============================================================================

class TestConfigInjectionContract:
    """Configuration injection contract tests
    
    These tests verify that:
    1. _load_*_config functions have been removed (no implicit container access)
    2. Services receive config via constructor injection
    3. Container provides configs when creating services
    """
    
    def test_no_load_config_functions_in_tile_mortality(self):
        """Verify _load_*_config functions removed from tile_based_mortality"""
        from ...simulation import tile_based_mortality
        
        # These functions should NOT exist
        assert not hasattr(tile_based_mortality, '_load_ecology_config')
        assert not hasattr(tile_based_mortality, '_load_mortality_config')
        assert not hasattr(tile_based_mortality, '_load_speciation_config')
    
    def test_no_load_config_function_in_species(self):
        """Verify _load_ecology_balance_config removed from species.py"""
        from ...simulation import species
        
        assert not hasattr(species, '_load_ecology_balance_config')
    
    def test_no_load_config_function_in_food_web(self):
        """Verify _load_food_web_config removed from food_web_manager"""
        from ...services.species import food_web_manager
        
        assert not hasattr(food_web_manager, '_load_food_web_config')
    
    def test_no_load_config_function_in_speciation(self):
        """Verify _load_speciation_config removed from speciation"""
        from ...services.species import speciation
        
        assert not hasattr(speciation, '_load_speciation_config')
    
    def test_no_load_config_function_in_resource_manager(self):
        """Verify _load_resource_config removed from resource_manager"""
        from ...services.ecology import resource_manager
        
        assert not hasattr(resource_manager, '_load_resource_config')
    
    def test_simulation_engine_accepts_configs(self):
        """Verify SimulationEngine accepts configs via constructor"""
        from ...simulation.engine import SimulationEngine
        import inspect
        
        sig = inspect.signature(SimulationEngine.__init__)
        params = list(sig.parameters.keys())
        
        # Should have 'configs' parameter
        assert 'configs' in params
    
    def test_container_provides_engine_configs(self):
        """Verify container has get_engine_configs method"""
        from ...core.container import ServiceContainer
        
        container = ServiceContainer()
        assert hasattr(container, 'get_engine_configs')
        
        # Should return dict with required keys
        configs = container.get_engine_configs()
        assert isinstance(configs, dict)
        assert 'ecology' in configs
        assert 'mortality' in configs
        assert 'speciation' in configs
        assert 'food_web' in configs


# ============================================================================
# 服务类配置注入契约测试
# ============================================================================

class TestServiceConfigInjectionContract:
    """服务类配置注入契约测试 - 验证服务通过构造函数接收配置，无静默回退"""
    
    def test_tile_mortality_engine_accepts_config(self):
        """测试 TileBasedMortalityEngine 接受配置注入"""
        from ...simulation.tile_based_mortality import TileBasedMortalityEngine
        from ...models.config import EcologyBalanceConfig, MortalityConfig, SpeciationConfig
        
        # 创建自定义配置
        ecology_config = EcologyBalanceConfig()
        mortality_config = MortalityConfig()
        speciation_config = SpeciationConfig()
        
        # 应该能接受配置参数
        engine = TileBasedMortalityEngine(
            batch_limit=50,
            ecology_config=ecology_config,
            mortality_config=mortality_config,
            speciation_config=speciation_config,
        )
        
        # 验证配置被正确存储
        assert engine._ecology_config is ecology_config
        assert engine._mortality_config is mortality_config
        assert engine._speciation_config is speciation_config
    
    def test_tile_mortality_engine_warns_without_config(self, caplog):
        """测试 TileBasedMortalityEngine 未提供配置时发出警告"""
        import logging
        from ...simulation.tile_based_mortality import TileBasedMortalityEngine
        
        with caplog.at_level(logging.WARNING):
            engine = TileBasedMortalityEngine()
        
        # 应该有警告日志
        assert "未注入" in caplog.text or "使用默认值" in caplog.text
    
    def test_tile_mortality_engine_reload_config(self, mock_container):
        """测试 TileBasedMortalityEngine 热加载配置"""
        from ...simulation.tile_based_mortality import TileBasedMortalityEngine
        from ...models.config import EcologyBalanceConfig, MortalityConfig, SpeciationConfig
        
        # 使用配置创建引擎（避免警告）
        engine = TileBasedMortalityEngine(
            ecology_config=EcologyBalanceConfig(),
            mortality_config=MortalityConfig(),
            speciation_config=SpeciationConfig(),
        )
        
        # 新的 reload_config 接受配置参数而不是从容器获取
        new_ecology = EcologyBalanceConfig(competition_base_coefficient=0.75)
        new_mortality = MortalityConfig()
        new_speciation = SpeciationConfig()
        
        engine.reload_config(
            ecology_config=new_ecology,
            mortality_config=new_mortality,
            speciation_config=new_speciation,
        )
        
        # 验证配置已更新
        assert engine._ecology_config.competition_base_coefficient == 0.75
    
    def test_food_web_manager_accepts_config(self):
        """测试 FoodWebManager 接受配置注入"""
        from ...services.species.food_web_manager import FoodWebManager
        from ...models.config import FoodWebConfig
        
        config = FoodWebConfig()
        manager = FoodWebManager(config=config)
        
        assert manager._config is config
    
    def test_food_web_manager_warns_without_config(self, caplog):
        """测试 FoodWebManager 未提供配置时发出警告"""
        import logging
        from ...services.species.food_web_manager import FoodWebManager
        
        with caplog.at_level(logging.WARNING):
            manager = FoodWebManager()
        
        # 应该有警告日志
        assert "未注入" in caplog.text or "使用默认值" in caplog.text
    
    def test_food_web_manager_reload_config(self, mock_container):
        """测试 FoodWebManager 热加载配置"""
        from ...services.species.food_web_manager import FoodWebManager
        from ...models.config import FoodWebConfig
        
        # 使用配置创建管理器（避免警告）
        manager = FoodWebManager(config=FoodWebConfig())
        
        # 新的 reload_config 接受配置参数而不是从容器获取
        new_config = FoodWebConfig(min_prey_count_t2=5)
        manager.reload_config(config=new_config)
        
        # 验证配置已更新
        assert manager._config.min_prey_count_t2 == 5
    
    def test_speciation_service_accepts_config(self):
        """测试 SpeciationService 接受配置注入"""
        from ...services.species.speciation import SpeciationService
        from ...models.config import SpeciationConfig
        
        config = SpeciationConfig()
        router = MagicMock()
        
        service = SpeciationService(router=router, config=config)
        
        assert service._config is config
    
    def test_speciation_service_warns_without_config(self, caplog):
        """测试 SpeciationService 未提供配置时发出警告"""
        import logging
        from ...services.species.speciation import SpeciationService
        
        router = MagicMock()
        
        with caplog.at_level(logging.WARNING):
            service = SpeciationService(router=router)
        
        # 应该有警告日志
        assert "未注入" in caplog.text or "使用默认值" in caplog.text
    
    def test_speciation_service_reload_config(self, mock_container):
        """测试 SpeciationService 热加载配置"""
        from ...services.species.speciation import SpeciationService
        from ...models.config import SpeciationConfig
        
        router = MagicMock()
        # 使用配置创建服务（避免警告）
        service = SpeciationService(router=router, config=SpeciationConfig())
        
        # 新的 reload_config 接受配置参数而不是从容器获取
        new_config = SpeciationConfig(base_speciation_rate=0.45)
        service.reload_config(config=new_config)
        
        # 验证配置已更新
        assert service._config.base_speciation_rate == 0.45



