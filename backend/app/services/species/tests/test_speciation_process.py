from types import SimpleNamespace

from ..speciation_process import partition_speciation_entries


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
