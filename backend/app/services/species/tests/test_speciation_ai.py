from ..speciation import SpeciationService
from ..speciation_ai import normalize_ai_content


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
