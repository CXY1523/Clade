from types import SimpleNamespace

from .. import speciation as speciation_module
from ..speciation import SpeciationService
from ..speciation_ai import build_batch_payload, normalize_ai_content


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
