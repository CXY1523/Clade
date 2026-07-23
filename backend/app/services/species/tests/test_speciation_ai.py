from types import SimpleNamespace

from .. import speciation as speciation_module
from ..speciation import SpeciationService
from ..speciation_ai import (
    build_batch_payload,
    generate_rule_based_fallback,
    normalize_ai_content,
)


class _NamingHints:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.sample_counts: list[int] = []

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def generate_naming_prompt(self, samples_per_category: int) -> str:
        self.sample_counts.append(samples_per_category)
        return "NAMING HINTS"


def _build_payload(
    monkeypatch,
    *,
    entries: list[dict],
    turn_index: int = 7,
    map_changes=None,
    major_events=None,
    is_plant=lambda _parent: False,
    get_stage_name=lambda stage: f"stage-{stage}",
    get_milestone_hints=lambda _parent: "MILESTONES",
    format_competition_context=lambda _parent, _parents: "COMPETITION",
    get_evolution_hint=lambda _code: None,
    naming_hint_generator=None,
    summarize_map_changes=lambda changes: f"MAP:{changes}",
    summarize_major_events=lambda events: f"EVENTS:{events}",
    get_tradeoff_ratio=lambda: 0.7,
) -> tuple[dict, _NamingHints]:
    monkeypatch.setattr(
        "app.simulation.constants.get_time_config",
        lambda turn: {
            "era_name": f"era-{turn}",
            "years_per_turn": 12_345,
            "evolution_guide": "guide",
        },
    )
    naming_hints = naming_hint_generator or _NamingHints()
    result = build_batch_payload(
        entries=entries,
        average_pressure=0.75,
        pressure_summary="PRESSURE",
        map_changes=["raised"] if map_changes is None else map_changes,
        major_events=["impact"] if major_events is None else major_events,
        turn_index=turn_index,
        is_plant=is_plant,
        get_stage_name=get_stage_name,
        get_milestone_hints=get_milestone_hints,
        format_competition_context=format_competition_context,
        get_evolution_hint=get_evolution_hint,
        naming_hint_generator=naming_hints,
        summarize_map_changes=summarize_map_changes,
        summarize_major_events=summarize_major_events,
        get_tradeoff_ratio=get_tradeoff_ratio,
    )
    return result, naming_hints


def test_batch_payload_preserves_animal_defaults_time_naming_and_order(
    monkeypatch,
) -> None:
    parent = SimpleNamespace(abstract_traits={})
    summary_calls: list[tuple[str, list]] = []

    payload, naming_hints = _build_payload(
        monkeypatch,
        entries=[
            {
                "payload": {
                    "parent_lineage": "PARENT",
                    "latin_name": "Animalia testus",
                    "common_name": "Test animal",
                    "habitat_type": "shore",
                    "traits": "curly {braces}",
                },
                "ctx": {"parent": parent, "new_code": "CHILD"},
            }
        ],
        turn_index=-3,
        summarize_map_changes=lambda changes: summary_calls.append(
            ("map", changes)
        )
        or "MAP SUMMARY",
        summarize_major_events=lambda events: summary_calls.append(
            ("events", events)
        )
        or "EVENT SUMMARY",
        get_tradeoff_ratio=lambda: 0.65,
    )

    assert list(payload) == [
        "average_pressure",
        "pressure_summary",
        "map_changes_summary",
        "major_events_summary",
        "major_events",
        "species_list",
        "batch_size",
        "time_context",
        "naming_hints",
        "max_increase",
        "single_max",
        "era_caps",
        "tradeoff_ratio",
        "parent_summary",
        "trigger_context",
    ]
    assert summary_calls == [
        ("map", ["raised"]),
        ("events", ["impact"]),
        ("events", ["impact"]),
    ]
    assert payload["time_context"] == (
        "\n=== ⏳ 时间尺度上下文 (Chronos Flow) ===\n"
        "当前地质年代：era-0\n"
        "时间流逝速度：12,345 年/回合\n"
        "演化指导原则：guide\n"
    )
    assert naming_hints.seeds == [
        abs(hash("batch--3-1")) % 1_000_000_007
    ]
    assert naming_hints.sample_counts == [2]
    assert payload["naming_hints"] == "NAMING HINTS"
    assert payload["batch_size"] == 1
    assert payload["tradeoff_ratio"] == 0.65
    assert payload["era_caps"] == "依时代上限"
    assert payload["parent_summary"] == payload["species_list"]
    assert payload["trigger_context"] == "PRESSURE"
    assert "【物种 1】🦎动物" in payload["species_list"]
    assert "- 生物类群: protist" in payload["species_list"]
    assert "- 营养级: T2.0（允许范围：1.5-2.5）" in payload["species_list"]
    assert "- 描述: curly {braces}" in payload["species_list"]
    assert "- 现有器官: 无已记录的器官系统" in payload["species_list"]
    assert "- 地理隔离: 否" in payload["species_list"]


def test_batch_payload_preserves_plant_context_parent_list_and_ai_slice(
    monkeypatch,
) -> None:
    plant = SimpleNamespace(
        life_form_stage=2,
        growth_form="moss",
        abstract_traits={
            "光合效率": 4.25,
            "保水能力": 5.0,
            "ignored": 99.0,
        },
    )
    other_parent = SimpleNamespace(abstract_traits={})
    competition_calls: list[tuple[object, list[object]]] = []
    hint_codes: list[str | None] = []

    def evolution_hint(code: str | None) -> dict | None:
        hint_codes.append(code)
        if code == "PLANT":
            return {"ai_directions": ["a", "b", "c", "d", "e", "ignored"]}
        return None

    payload, _ = _build_payload(
        monkeypatch,
        entries=[
            {
                "payload": {
                    "parent_lineage": "PLANT",
                    "trait_budget_summary": "PLANT BUDGET",
                },
                "ctx": {"parent": plant, "new_code": "PLANT-CHILD"},
            },
            {
                "payload": {"parent_lineage": "ANIMAL"},
                "ctx": {"parent": other_parent, "new_code": "ANIMAL-CHILD"},
            },
        ],
        is_plant=lambda parent: parent is plant,
        get_stage_name=lambda stage: f"named-{stage}",
        get_milestone_hints=lambda parent: (
            "PLANT MILESTONES" if parent is plant else "wrong"
        ),
        format_competition_context=lambda parent, parents: competition_calls.append(
            (parent, parents)
        )
        or "PLANT COMPETITION",
        get_evolution_hint=evolution_hint,
    )

    assert competition_calls == [(plant, [plant, other_parent])]
    assert hint_codes == ["PLANT", "ANIMAL"]
    assert "【🌱植物演化信息】" in payload["species_list"]
    assert "当前阶段: 2 (named-2)" in payload["species_list"]
    assert "生长形式: moss" in payload["species_list"]
    assert "光合效率=4.2, 保水能力=5.0" in payload["species_list"]
    assert "ignored" not in payload["species_list"]
    assert "PLANT MILESTONES" in payload["species_list"]
    assert "PLANT COMPETITION" in payload["species_list"]
    assert "建议方向: a, b, c, d, e" in payload["species_list"]
    assert "ignored" not in payload["species_list"]
    assert payload["era_caps"] == "PLANT BUDGET"


def test_batch_payload_preserves_empty_batch_defaults(monkeypatch) -> None:
    ratio_calls: list[None] = []

    payload, naming_hints = _build_payload(
        monkeypatch,
        entries=[],
        map_changes=[],
        major_events=[],
        get_tradeoff_ratio=lambda: ratio_calls.append(None) or 0.8,
    )

    assert payload["map_changes_summary"] == "无显著地形变化"
    assert payload["major_events_summary"] == "无重大事件"
    assert payload["major_events"] == "无重大事件"
    assert payload["species_list"] == ""
    assert payload["batch_size"] == 0
    assert payload["era_caps"] == "依时代上限"
    assert payload["tradeoff_ratio"] == 0.8
    assert ratio_calls == [None]
    assert naming_hints.seeds == [
        abs(hash("batch-7-0")) % 1_000_000_007
    ]


def test_service_batch_payload_delegate_preserves_bound_overrides(
    monkeypatch,
) -> None:
    captured: dict = {}
    sentinel = {"delegated": True}

    def fake_build_batch_payload(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        speciation_module, "build_batch_payload", fake_build_batch_payload
    )
    service = object.__new__(SpeciationService)
    service.naming_hint_generator = _NamingHints()
    service.tradeoff_calculator = SimpleNamespace(tradeoff_ratio=0.91)
    service.tradeoff_ratio = 0.12
    service.get_evolution_hint = lambda code: {"code": code}
    service._summarize_map_changes = lambda changes: f"custom-map:{changes}"
    service._summarize_major_events = lambda events: f"custom-events:{events}"

    result = service._build_batch_payload(
        entries=[],
        average_pressure=0.5,
        pressure_summary="summary",
        map_changes=[],
        major_events=[],
        turn_index=4,
    )

    assert result is sentinel
    assert captured["entries"] == []
    assert captured["average_pressure"] == 0.5
    assert captured["pressure_summary"] == "summary"
    assert captured["map_changes"] == []
    assert captured["major_events"] == []
    assert captured["turn_index"] == 4
    assert captured["get_evolution_hint"]("X") == {"code": "X"}
    assert captured["summarize_map_changes"](["x"]) == "custom-map:['x']"
    assert captured["summarize_major_events"](["x"]) == "custom-events:['x']"
    assert captured["naming_hint_generator"] is service.naming_hint_generator
    assert captured["get_tradeoff_ratio"]() == 0.91


def _fallback_parent(**overrides):
    values = {
        "latin_name": "Testus parentus",
        "common_name": "亲本",
        "habitat_type": "marine",
        "trophic_level": 2.5,
        "diet_type": "carnivore",
        "prey_species": ("PREY-A", "PREY-B"),
        "prey_preferences": {"PREY-A": 0.7},
        "morphology_stats": {
            "body_length_cm": 20.0,
            "body_weight_g": 300.0,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rule_fallback_preserves_deterministic_default_budget_and_copies() -> None:
    parent = _fallback_parent()
    callback_calls: list[tuple] = []

    def preprocess_rules(**kwargs):
        callback_calls.append(("preprocess", kwargs))
        return {
            "evolution_direction": "活动强化型",
            "direction_description": "快速迁移",
            "suggested_increases": ["速度", "随机选择", "未使用"],
            "suggested_decreases": ["防御"],
        }

    def generate_name(**kwargs):
        callback_calls.append(("name", kwargs))
        return "疾岸兽"

    result = generate_rule_based_fallback(
        parent=parent,
        new_code="NEW.123",
        survivors=4321,
        speciation_type="地理隔离",
        average_pressure=0.82,
        environment_pressure={"temperature": 0.4},
        turn_index=17,
        preprocess_rules=preprocess_rules,
        generate_background_species_name=generate_name,
    )

    assert callback_calls[0] == (
        "preprocess",
        {
            "parent_species": parent,
            "offspring_index": 1,
            "total_offspring": 1,
            "environment_pressure": {"temperature": 0.4},
            "pressure_context": "地理隔离",
            "turn_index": 17,
        },
    )
    assert callback_calls[1][0] == "name"
    assert callback_calls[1][1]["parent"] is parent
    assert callback_calls[1][1]["evolution_direction"] == "活动强化型"
    assert callback_calls[1][1]["habitat_type"] == "marine"
    assert callback_calls[1][1]["trophic_level"] == 2.5
    assert result == {
        "latin_name": "Testus cursor_123",
        "common_name": "疾岸兽",
        "description": (
            "疾岸兽是从亲本分化而来的海洋肉食性物种。"
            "在地理隔离的选择压力下，该物种发展出快速迁移的演化策略。"
            "主要适应性变化包括速度增强、防御降低。"
        ),
        "habitat_type": "marine",
        "trophic_level": 2.5,
        "diet_type": "carnivore",
        "prey_species": ["PREY-A", "PREY-B"],
        "prey_preferences": {"PREY-A": 0.7},
        "key_innovations": ["地理隔离适应", "速度强化"],
        "trait_changes": {"速度": "+1.5", "防御": "-0.9"},
        "morphology_changes": {
            "body_length_cm": 0.9863118604476898,
            "body_weight_g": 0.9551457055184757,
        },
        "event_description": "因地理隔离从亲本分化，采用活动强化型策略",
        "speciation_type": "地理隔离",
        "reason": "在地理隔离条件下，通过快速迁移实现自然选择",
        "organ_evolution": [],
        "_is_rule_fallback": True,
        "_evolution_direction": "活动强化型",
    }
    assert result["prey_species"] is not parent.prey_species
    assert result["prey_preferences"] is not parent.prey_preferences


def test_rule_fallback_preserves_budget_organ_and_random_draw_order() -> None:
    parent = _fallback_parent()
    budget = SimpleNamespace(
        total_increase_allowed=2.0,
        total_decrease_required=1.0,
        single_trait_max=1.0,
    )
    organs = [
        SimpleNamespace(category="locomotion", current_stage=1, max_target_stage=2),
        SimpleNamespace(category="sensory", current_stage=2, max_target_stage=4),
    ]

    result = generate_rule_based_fallback(
        parent=parent,
        new_code="ORG.457",
        survivors=100,
        speciation_type="竞争分化",
        average_pressure=0.5,
        preprocess_rules=lambda **_kwargs: {
            "evolution_direction": "防御特化型",
            "direction_description": "结构防御",
            "suggested_increases": ["护甲", "耐力"],
            "suggested_decreases": ["速度", "繁殖"],
            "_trait_budget": budget,
            "_organ_constraints": organs,
        },
        generate_background_species_name=lambda **_kwargs: "甲感兽",
    )

    assert result["latin_name"] == "Testus coriaceus_457"
    assert result["trait_changes"] == {
        "护甲": "+0.9",
        "耐力": "+0.9",
        "速度": "-0.4",
        "繁殖": "-0.5",
    }
    assert result["organ_evolution"] == [
        {
            "category": "sensory",
            "current_stage": 2,
            "target_stage": 3,
            "description": "感觉器官发展至功能完善",
        }
    ]
    assert result["morphology_changes"] == {
        "body_length_cm": 1.1220982527500307,
        "body_weight_g": 1.4065023615055423,
    }
    assert result["description"].endswith("形态上，感觉器官发展至功能完善。")
    assert result["key_innovations"] == [
        "竞争分化适应",
        "感觉器官发展至功能完善",
        "护甲强化",
    ]


def test_rule_fallback_preserves_empty_defaults_and_fresh_environment() -> None:
    parent = _fallback_parent(
        latin_name=None,
        common_name=None,
        habitat_type=None,
        trophic_level=1.0,
        diet_type=None,
        prey_species=None,
        prey_preferences=None,
        morphology_stats=None,
    )
    environments: list[dict[str, float]] = []

    result = generate_rule_based_fallback(
        parent=parent,
        new_code="EMPTY.789",
        survivors=0,
        speciation_type="自然分化",
        average_pressure=0.0,
        preprocess_rules=lambda **kwargs: environments.append(
            kwargs["environment_pressure"]
        )
        or {},
        generate_background_species_name=lambda **_kwargs: "无名兽",
    )

    assert environments == [{"temperature": 0, "humidity": 0}]
    assert result["latin_name"] == "Species evolutus_789"
    assert result["prey_species"] == []
    assert result["prey_preferences"] == {}
    assert result["trait_changes"] == {}
    assert result["organ_evolution"] == []
    assert result["morphology_changes"] == {
        "body_length_cm": 1.0362646606813137,
        "body_weight_g": 1.1609732302348752,
    }
    assert result["description"] == (
        "无名兽是从None分化而来的未知杂食性物种。"
        "在自然分化的选择压力下，该物种发展出自然选择的演化策略。"
    )
    assert result["key_innovations"] == ["自然分化适应"]


def test_service_rule_fallback_delegate_preserves_bound_overrides(
    monkeypatch,
) -> None:
    captured: dict = {}
    sentinel = {"delegated": True}

    def fake_generate_rule_based_fallback(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        speciation_module,
        "generate_rule_based_fallback",
        fake_generate_rule_based_fallback,
    )
    service = object.__new__(SpeciationService)
    service.rules = SimpleNamespace(preprocess=lambda **kwargs: ("rules", kwargs))
    service._generate_background_species_name = lambda **kwargs: ("name", kwargs)
    parent = _fallback_parent()

    result = service._generate_rule_based_fallback(
        parent=parent,
        new_code="NEW",
        survivors=123,
        speciation_type="test",
        average_pressure=0.4,
        environment_pressure={"humidity": 0.2},
        turn_index=9,
    )

    assert result is sentinel
    assert captured["parent"] is parent
    assert captured["new_code"] == "NEW"
    assert captured["survivors"] == 123
    assert captured["speciation_type"] == "test"
    assert captured["average_pressure"] == 0.4
    assert captured["environment_pressure"] == {"humidity": 0.2}
    assert captured["turn_index"] == 9
    assert captured["preprocess_rules"](value=1) == (
        "rules",
        {"value": 1},
    )
    assert captured["generate_background_species_name"](value=2) == (
        "name",
        {"value": 2},
    )


def test_ai_normalization_preserves_non_dict_identity() -> None:
    content = ["not", "a", "dict"]

    assert normalize_ai_content(content) is content


def test_ai_normalization_skips_truthy_existing_trait_changes() -> None:
    content = {
        "trait_changes": {"speed": 1.0},
        "innovations": [{"name": "wing", "gains": {"armor": 2.0}}],
    }

    assert normalize_ai_content(content) is content
    assert content == {
        "trait_changes": {"speed": 1.0},
        "innovations": [{"name": "wing", "gains": {"armor": 2.0}}],
    }


def test_ai_normalization_aggregates_mixed_gains_and_deduplicates_names() -> None:
    content = {
        "trait_changes": {},
        "innovations": [
            {"name": "wing", "gains": {"speed": "+1.5", "armor": "bad"}},
            {"name": "wing", "gains": {"speed": 2, "social": None}},
            "invalid",
            {"name": "shell", "gains": ["not", "a", "dict"]},
        ],
    }

    assert normalize_ai_content(content) is content
    assert content["trait_changes"] == {"speed": 3.5}
    assert content["key_innovations"] == ["wing", "shell"]


def test_ai_normalization_preserves_existing_key_innovations() -> None:
    existing = ["existing"]
    content = {
        "key_innovations": existing,
        "innovations": [
            {"name": "wing", "gains": {"speed": 1.0}},
        ],
    }

    normalize_ai_content(content)

    assert content["trait_changes"] == {"speed": 1.0}
    assert content["key_innovations"] is existing
    assert content["key_innovations"] == ["existing"]


def test_ai_normalization_adds_names_even_without_valid_gains() -> None:
    content = {
        "innovations": [
            {"name": "shell", "gains": {"armor": object()}},
        ],
    }

    normalize_ai_content(content)

    assert "trait_changes" not in content
    assert content["key_innovations"] == ["shell"]


def test_service_ai_normalization_delegate_matches_direct_function() -> None:
    service = object.__new__(SpeciationService)
    service_content = {
        "innovations": [{"name": "wing", "gains": {"speed": "+1.0"}}]
    }
    direct_content = {
        "innovations": [{"name": "wing", "gains": {"speed": "+1.0"}}]
    }

    assert service._normalize_ai_content(service_content) == normalize_ai_content(
        direct_content
    )
