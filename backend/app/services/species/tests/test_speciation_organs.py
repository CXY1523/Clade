from copy import deepcopy
from types import SimpleNamespace

import pytest

from ..plant_evolution import (
    PLANT_ORGANS,
    PLANT_ORGAN_CATEGORIES,
    plant_evolution_service,
)
from ..speciation import SpeciationService
from ..speciation_organs import (
    get_complexity_constraints,
    infer_complexity_by_rules,
    process_plant_organ_changes,
    update_capabilities,
    validate_gradual_evolution,
)


def test_capabilities_convert_legacy_labels_preserve_unknown_and_deduplicate() -> None:
    parent = SimpleNamespace(
        capabilities=["photosynthesis", "光合作用", "custom", "venom"]
    )

    result = update_capabilities(parent, {})

    assert set(result) == {"光合作用", "custom", "毒素"}


def test_capabilities_derive_labels_from_each_active_organ_category() -> None:
    parent = SimpleNamespace(capabilities=[])
    organs = {
        "locomotion": {"type": "flagellum", "is_active": True},
        "sensory": {"type": "compound eye", "is_active": True},
        "metabolic": {"type": "chloroplast", "is_active": True},
        "digestive": {"type": "gut", "is_active": True},
        "defense": {"type": "shell", "is_active": True},
    }

    result = update_capabilities(parent, organs)

    assert set(result) == {
        "鞭毛运动",
        "感光",
        "视觉",
        "光合作用",
        "消化",
        "盔甲",
    }


def test_capabilities_skip_inactive_organs_and_preserve_keyword_priority() -> None:
    parent = SimpleNamespace(capabilities=[])
    organs = {
        "locomotion": {"type": "cilia leg fin", "is_active": True},
        "sensory": {"type": "eye", "is_active": False},
        "defense": {"type": "shell spine toxin", "is_active": True},
    }
    original = deepcopy(organs)

    result = update_capabilities(parent, organs)

    assert set(result) == {"纤毛运动", "盔甲"}
    assert organs == original


def test_service_capability_delegate_matches_direct_function() -> None:
    parent = SimpleNamespace(capabilities=["swimming"])
    organs = {
        "sensory": {"type": "chemoreceptor", "is_active": True},
    }
    service = object.__new__(SpeciationService)

    assert set(service._update_capabilities(parent, organs)) == set(
        update_capabilities(parent, organs)
    )


def _complexity_species(
    *,
    description: str = "",
    common_name: str = "",
    organs: dict | None = None,
    body_length_cm: float = 0.01,
) -> SimpleNamespace:
    return SimpleNamespace(
        description=description,
        common_name=common_name,
        organs=organs or {},
        morphology_stats={"body_length_cm": body_length_cm},
    )


def test_complexity_rules_preserve_keyword_priority_and_prokaryote_guard() -> None:
    assert infer_complexity_by_rules(
        _complexity_species(description="细菌与哺乳动物的对照描述")
    ) == 5
    assert infer_complexity_by_rules(
        _complexity_species(description="具有线粒体的细菌", body_length_cm=0.01)
    ) == 1


@pytest.mark.parametrize(
    ("organs", "body_length_cm", "expected"),
    [
        ({f"organ_{index}": {} for index in range(5)}, 0.01, 4),
        ({f"organ_{index}": {} for index in range(3)}, 0.01, 3),
        ({"active": {}, "inactive": {"is_active": False}}, 0.2, 2),
        ({"active": {}}, 0.1, 1),
    ],
)
def test_complexity_rules_count_only_active_organs(
    organs: dict, body_length_cm: float, expected: int
) -> None:
    assert infer_complexity_by_rules(
        _complexity_species(organs=organs, body_length_cm=body_length_cm)
    ) == expected


@pytest.mark.parametrize(
    ("body_length_cm", "expected"),
    [
        (0.0009, 0),
        (0.001, 1),
        (0.1, 2),
        (1.0, 3),
        (10.0, 4),
    ],
)
def test_complexity_rules_preserve_body_length_boundaries(
    body_length_cm: float, expected: int
) -> None:
    assert infer_complexity_by_rules(
        _complexity_species(body_length_cm=body_length_cm)
    ) == expected


def test_complexity_constraints_preserve_prokaryote_and_eukaryote_rules() -> None:
    assert get_complexity_constraints("complexity_0") == {
        "origin_type": "prokaryote",
        "hard_forbidden": [
            "真核鞭毛",
            "纤毛",
            "线粒体",
            "叶绿体",
            "细胞核",
            "内质网",
            "高尔基体",
        ],
        "max_organ_stage": 4,
    }
    assert get_complexity_constraints("complexity_5") == {
        "origin_type": "eukaryote",
        "hard_forbidden": [],
        "max_organ_stage": 4,
    }
    assert get_complexity_constraints("unknown") == {
        "origin_type": "eukaryote",
        "hard_forbidden": [],
        "max_organ_stage": 4,
    }

    with pytest.raises(ValueError):
        get_complexity_constraints("complexity_invalid")


def test_service_complexity_delegates_match_direct_functions() -> None:
    species = _complexity_species(common_name="水母")
    service = object.__new__(SpeciationService)

    assert service._infer_complexity_by_rules(species) == infer_complexity_by_rules(
        species
    )
    assert service._get_complexity_constraints(
        "complexity_0"
    ) == get_complexity_constraints("complexity_0")


def test_gradual_validation_empty_input_skips_constraint_lookup() -> None:
    def unexpected_lookup(_domain: str) -> dict:
        raise AssertionError("empty input must not read constraints")

    assert validate_gradual_evolution([], {}, "complexity_1", unexpected_lookup) == (
        True,
        [],
    )


def test_gradual_validation_skips_non_dictionary_items() -> None:
    assert validate_gradual_evolution(
        [None, "bad"], {}, "complexity_1", get_complexity_constraints
    ) == (True, [])


def test_gradual_validation_applies_parent_stage_and_jump_rules_in_place() -> None:
    change = {
        "category": "sensory",
        "action": "enhance",
        "current_stage": 1,
        "target_stage": 4,
        "structure_name": "眼",
    }

    valid, result = validate_gradual_evolution(
        [change],
        {"sensory": {"evolution_stage": 2}},
        "complexity_1",
        get_complexity_constraints,
    )

    assert valid is True
    assert result == [change]
    assert result[0] is change
    assert change["current_stage"] == 2
    assert change["target_stage"] == 3


def test_gradual_validation_starts_new_organs_at_stage_one() -> None:
    change = {
        "category": "defense",
        "action": "initiate",
        "current_stage": 0,
        "target_stage": 4,
        "structure_name": "壳",
    }

    assert validate_gradual_evolution(
        [change], {}, "complexity_1", get_complexity_constraints
    ) == (True, [change])
    assert change["target_stage"] == 1


def test_gradual_validation_filters_prokaryote_forbidden_structures() -> None:
    forbidden = {
        "category": "metabolic",
        "action": "initiate",
        "target_stage": 1,
        "structure_name": "线粒体",
    }
    allowed = {
        "category": "defense",
        "action": "initiate",
        "target_stage": 1,
        "structure_name": "细胞壁",
    }

    assert validate_gradual_evolution(
        [forbidden, allowed], {}, "complexity_0", get_complexity_constraints
    ) == (True, [allowed])


def test_gradual_validation_converts_missing_parent_enhancement() -> None:
    change = {
        "category": "sensory",
        "action": "enhance",
        "current_stage": 3,
        "target_stage": 4,
        "structure_name": "眼",
    }

    assert validate_gradual_evolution(
        [change], {}, "complexity_1", get_complexity_constraints
    ) == (True, [change])
    assert change == {
        "category": "sensory",
        "action": "initiate",
        "current_stage": 0,
        "target_stage": 1,
        "structure_name": "眼",
    }


def test_gradual_validation_uses_constraint_callback() -> None:
    seen = []
    change = {
        "category": "defense",
        "action": "initiate",
        "current_stage": 0,
        "target_stage": 4,
        "structure_name": "壳",
    }

    def constraints(domain: str) -> dict:
        seen.append(domain)
        return {
            "origin_type": "eukaryote",
            "hard_forbidden": [],
            "max_organ_stage": 1,
        }

    assert validate_gradual_evolution(
        [change], {}, "custom_domain", constraints
    ) == (True, [change])
    assert seen == ["custom_domain"]
    assert change["target_stage"] == 1


def test_gradual_validation_keeps_only_first_three_valid_changes() -> None:
    changes = [
        {
            "category": str(index),
            "action": "initiate",
            "target_stage": 0,
            "structure_name": str(index),
        }
        for index in range(4)
    ]

    assert validate_gradual_evolution(
        changes, {}, "complexity_1", get_complexity_constraints
    ) == (True, changes[:3])


def test_service_gradual_validation_delegate_matches_direct_function() -> None:
    changes = [
        {
            "category": "sensory",
            "action": "enhance",
            "current_stage": 0,
            "target_stage": 4,
            "structure_name": "眼",
        }
    ]
    parent_organs = {"sensory": {"evolution_stage": 2}}
    service_changes = deepcopy(changes)
    direct_changes = deepcopy(changes)
    service = object.__new__(SpeciationService)

    assert service._validate_gradual_evolution(
        service_changes, parent_organs, "complexity_1"
    ) == validate_gradual_evolution(
        direct_changes,
        parent_organs,
        "complexity_1",
        get_complexity_constraints,
    )
    assert service_changes == direct_changes


def _process_plant_changes(
    organs: dict,
    changes: list,
    parent: SimpleNamespace,
    turn_index: int,
) -> dict:
    return process_plant_organ_changes(
        organs,
        changes,
        parent,
        turn_index,
        plant_evolution_service,
        PLANT_ORGANS,
        PLANT_ORGAN_CATEGORIES,
    )


def test_plant_changes_empty_input_returns_same_generic_organs() -> None:
    organs = {"legacy": {"type": "legacy"}}
    parent = SimpleNamespace(life_form_stage=0, plant_organs=None)

    result = _process_plant_changes(organs, [], parent, 4)

    assert result is organs
    assert result == {
        "legacy": {"type": "legacy"},
        "_plant_organs": {},
    }


def test_plant_changes_add_reference_milestone_from_legacy_parameters() -> None:
    organs = {}
    parent = SimpleNamespace(life_form_stage=1, plant_organs=None)
    change = {
        "category": "photosynthetic",
        "change_type": "new",
        "organ_name": "叶绿体",
        "parameter": "efficiency",
        "delta": 9.0,
    }

    result = _process_plant_changes(organs, [change], parent, 7)
    stored = result["_plant_organs"]["photosynthetic"]["叶绿体"]

    assert stored == {
        "efficiency": 5.0,
        "min_stage": 0,
        "acquired_turn": 7,
        "is_custom": False,
        "milestone_required": True,
        "milestone_id": "first_eukaryote",
    }
    assert result["photosynthetic"] == {
        "type": "叶绿体",
        "parameters": stored.copy(),
        "evolution_stage": 4,
        "evolution_progress": 1.0,
        "is_active": True,
    }


def test_plant_changes_skip_unknown_and_stage_locked_categories() -> None:
    parent = SimpleNamespace(life_form_stage=0, plant_organs=None)
    changes = [
        {
            "category": "unknown",
            "change_type": "new",
            "organ_name": "未知",
            "parameters": {},
        },
        {
            "category": "root_system",
            "change_type": "new",
            "organ_name": "假根",
            "parameters": {"depth_cm": 1, "absorption": 1},
        },
    ]

    assert _process_plant_changes({}, changes, parent, 1) == {
        "_plant_organs": {}
    }


def test_plant_changes_enhance_and_degrade_shared_parent_records() -> None:
    photosynthetic = {"efficiency": 4.8, "min_stage": 0}
    protection = {"uv_resist": 1.0, "min_stage": 0}
    parent = SimpleNamespace(
        life_form_stage=3,
        plant_organs={
            "photosynthetic": {"自定义叶": photosynthetic},
            "protection": {"树脂层": protection},
        },
    )
    changes = [
        {
            "category": "photosynthetic",
            "change_type": "enhance",
            "organ_name": "自定义叶",
            "parameters": {"efficiency": 1.0},
        },
        {
            "category": "protection",
            "change_type": "degrade",
            "organ_name": "树脂层",
        },
    ]

    result = _process_plant_changes({}, changes, parent, 9)

    assert photosynthetic == {
        "efficiency": 5.0,
        "min_stage": 0,
        "modified_turn": 9,
    }
    assert protection == {
        "uv_resist": 1.0,
        "min_stage": 0,
        "is_degraded": True,
        "degraded_turn": 9,
    }
    assert result["photosynthetic"]["type"] == "自定义叶"
    assert "type" not in result["protection"]


def test_plant_changes_protect_milestones_and_keep_first_best_organ() -> None:
    first = {"efficiency": 2.0, "min_stage": 0}
    tied = {"efficiency": 2.0, "min_stage": 0}
    milestone = {"efficiency": 1.0, "min_stage": 0}
    parent = SimpleNamespace(
        life_form_stage=3,
        plant_organs={
            "photosynthetic": {
                "第一叶": first,
                "同值叶": tied,
                "叶绿体": milestone,
            }
        },
    )
    changes = [
        {
            "category": "photosynthetic",
            "change_type": "degrade",
            "organ_name": "叶绿体",
        }
    ]

    result = _process_plant_changes({}, changes, parent, 10)

    assert "is_degraded" not in milestone
    assert result["photosynthetic"]["type"] == "第一叶"


def test_service_plant_change_delegate_matches_direct_function() -> None:
    parent = SimpleNamespace(life_form_stage=1, plant_organs=None)
    changes = [
        {
            "category": "photosynthetic",
            "change_type": "new",
            "organ_name": "光合泡",
            "parameters": {"efficiency": 1.2},
        }
    ]
    service_organs = {}
    direct_organs = {}
    service = object.__new__(SpeciationService)

    assert service._process_plant_organ_changes(
        service_organs, deepcopy(changes), parent, 3
    ) == _process_plant_changes(
        direct_organs, deepcopy(changes), parent, 3
    )
    assert service_organs == direct_organs
