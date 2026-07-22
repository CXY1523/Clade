from copy import deepcopy
from types import SimpleNamespace

import pytest

from ..speciation import SpeciationService
from ..speciation_organs import (
    get_complexity_constraints,
    infer_complexity_by_rules,
    update_capabilities,
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
