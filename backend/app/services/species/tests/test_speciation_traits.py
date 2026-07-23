from types import SimpleNamespace

from ..speciation import SpeciationService
from ..speciation_traits import validate_trait_changes


def _limits(
    *,
    total: float = 100.0,
    specialized: float = 20.0,
    base: float = 10.0,
) -> dict:
    return {
        "total": total,
        "specialized": specialized,
        "base": base,
    }


def _validate(
    old_traits: dict,
    new_traits: dict,
    *,
    limits: dict | None = None,
) -> tuple[bool, str]:
    return validate_trait_changes(
        old_traits,
        new_traits,
        2.5,
        lambda trophic_level: limits or _limits(),
    )


def test_trait_validation_rejects_net_gain_above_six_first() -> None:
    assert _validate({"speed": 1.0}, {"speed": 7.1}) == (
        False,
        "属性总和净增加6.1，超过上限6.0（建议≤3.0）",
    )


def test_trait_validation_rejects_total_above_trophic_limit() -> None:
    assert _validate(
        {"speed": 6.0, "armor": 5.0},
        {"speed": 6.0, "armor": 5.0},
        limits=_limits(total=10.0),
    ) == (
        False,
        "属性总和11.0超过营养级T2.5的上限10.0",
    )


def test_trait_validation_reports_first_specialized_trait_in_input_order() -> None:
    assert _validate(
        {"speed": 9.0, "armor": 10.0},
        {"speed": 9.0, "armor": 10.0},
        limits=_limits(specialized=8.0),
    ) == (
        False,
        "属性speed=9.0超过特化上限8.0",
    )


def test_trait_validation_rejects_more_than_two_traits_above_base_limit() -> None:
    traits = {"speed": 6.0, "armor": 6.0, "reproduction": 6.0}

    assert _validate(
        traits,
        traits,
        limits=_limits(base=5.0),
    ) == (
        False,
        "3个属性超过基础上限5.0，最多允许2个",
    )


def test_trait_validation_rejects_large_gain_without_a_decrease() -> None:
    assert _validate(
        {"speed": 1.0, "armor": 1.0},
        {"speed": 6.0, "armor": 1.0},
    ) == (
        False,
        "净增益5.0但无权衡代价（需要至少一项属性降低）",
    )


def test_trait_validation_preserves_exact_boundaries_and_compensating_decrease() -> None:
    assert _validate(
        {"speed": 1.0, "armor": 1.0},
        {"speed": 5.0, "armor": 1.0},
    ) == (True, "验证通过")
    assert _validate(
        {"speed": 1.0, "armor": 5.0},
        {"speed": 10.0, "armor": 1.0},
        limits=_limits(total=11.0, specialized=10.0, base=9.0),
    ) == (True, "验证通过")


def test_service_trait_validation_delegate_uses_bound_limit_calculator() -> None:
    calls = []
    service = object.__new__(SpeciationService)
    service.trophic_calculator = SimpleNamespace(
        get_attribute_limits=lambda trophic_level: (
            calls.append(trophic_level) or _limits()
        )
    )

    assert service._validate_trait_changes(
        {"speed": 2.0},
        {"speed": 3.0},
        3.5,
    ) == (True, "验证通过")
    assert calls == [3.5]
