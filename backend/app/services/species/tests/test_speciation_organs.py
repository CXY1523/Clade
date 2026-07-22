from copy import deepcopy
from types import SimpleNamespace

from ..speciation import SpeciationService
from ..speciation_organs import update_capabilities


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
