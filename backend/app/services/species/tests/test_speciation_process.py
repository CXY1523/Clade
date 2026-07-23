from types import SimpleNamespace

from ..speciation_process import (
    generate_background_results,
    partition_speciation_entries,
)


def _entry(name: str, *, background: bool | None = None) -> dict:
    parent = SimpleNamespace(name=name)
    if background is not None:
        parent.is_background = background
    return {"name": name, "ctx": {"parent": parent}}


def test_entry_partition_preserves_flags_identity_order_and_inputs() -> None:
    background_one = _entry("background-one", background=True)
    active_one = _entry("active-one", background=False)
    missing_flag = _entry("missing-flag")
    background_two = _entry("background-two", background=True)
    deferred_one = _entry("deferred-one")
    deferred_two = _entry("deferred-two")
    entries = [
        background_one,
        active_one,
        missing_flag,
        background_two,
    ]
    deferred = [deferred_one, deferred_two]
    original_entries = list(entries)
    original_deferred = list(deferred)

    background, active, remaining = partition_speciation_entries(
        entries,
        deferred,
        max_deferred_requests=10,
        max_speciation_per_turn=3,
    )

    assert background == [background_one, background_two]
    assert active == [deferred_one, deferred_two, active_one]
    assert remaining == [missing_flag]
    assert background[0] is background_one
    assert active[0] is deferred_one
    assert remaining[0] is missing_flag
    assert entries == original_entries
    assert deferred == original_deferred
    assert active is not deferred
    assert remaining is not deferred


def test_entry_partition_truncates_pending_before_per_turn_slice() -> None:
    background = _entry("background", background=True)
    deferred_one = _entry("deferred-one")
    deferred_two = _entry("deferred-two")
    active_one = _entry("active-one")
    active_two = _entry("active-two")

    background_entries, active, remaining = partition_speciation_entries(
        [background, active_one, active_two],
        [deferred_one, deferred_two],
        max_deferred_requests=3,
        max_speciation_per_turn=1,
    )

    assert background_entries == [background]
    assert active == [deferred_one]
    assert remaining == [deferred_two, active_one]
    assert active_two not in active
    assert active_two not in remaining


def test_entry_partition_preserves_zero_limits_and_background_entries() -> None:
    background = _entry("background", background=True)

    assert partition_speciation_entries(
        [background, _entry("active")],
        [_entry("deferred")],
        max_deferred_requests=0,
        max_speciation_per_turn=0,
    ) == ([background], [], [])


def test_entry_partition_preserves_empty_lists() -> None:
    assert partition_speciation_entries(
        [],
        [],
        max_deferred_requests=5,
        max_speciation_per_turn=2,
    ) == ([], [], [])


def test_background_results_preserve_pressures_order_arguments_and_identity(
) -> None:
    entries = [
        {
            "ctx": {
                "parent": SimpleNamespace(common_name="Parent A"),
                "new_code": "CHILD-A",
                "population": 101,
                "speciation_type": "type-a",
            }
        },
        {
            "ctx": {
                "parent": SimpleNamespace(common_name="Parent B"),
                "new_code": "CHILD-B",
                "population": 202,
                "speciation_type": "type-b",
            }
        },
    ]
    pressures = [
        SimpleNamespace(category="temperature", intensity=0.2),
        SimpleNamespace(category="ignored-no-intensity"),
        SimpleNamespace(intensity=9.9),
        SimpleNamespace(category="temperature", intensity=0.8),
        SimpleNamespace(category="humidity", intensity=0.4),
    ]
    calls: list[dict] = []
    contents = [
        {"common_name": "Child A", "_evolution_direction": "direction-a"},
        {"common_name": "Child B"},
    ]

    def fallback(**kwargs):
        calls.append(kwargs)
        return contents[len(calls) - 1]

    results = generate_background_results(
        entries,
        pressures,
        average_pressure=3.5,
        turn_index=12,
        generate_rule_based_fallback=fallback,
    )

    assert results == [
        (entries[0], contents[0]),
        (entries[1], contents[1]),
    ]
    assert results[0][0] is entries[0]
    assert results[0][1] is contents[0]
    assert [call["new_code"] for call in calls] == ["CHILD-A", "CHILD-B"]
    assert calls[0] == {
        "parent": entries[0]["ctx"]["parent"],
        "new_code": "CHILD-A",
        "survivors": 101,
        "speciation_type": "type-a",
        "average_pressure": 3.5,
        "environment_pressure": {
            "temperature": 0.8,
            "humidity": 0.4,
        },
        "turn_index": 12,
    }
    assert calls[1]["parent"] is entries[1]["ctx"]["parent"]
    assert (
        calls[0]["environment_pressure"]
        is calls[1]["environment_pressure"]
    )


def test_background_results_preserve_empty_inputs() -> None:
    calls: list[dict] = []

    assert generate_background_results(
        [],
        None,
        average_pressure=0.0,
        turn_index=0,
        generate_rule_based_fallback=lambda **kwargs: calls.append(kwargs),
    ) == []
    assert calls == []
