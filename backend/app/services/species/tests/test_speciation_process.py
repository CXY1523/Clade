from types import SimpleNamespace

import pytest

from .. import speciation_process as speciation_process_module
from ..speciation_process import (
    enhance_rule_fallback_descriptions,
    execute_active_ai_batches,
    generate_background_results,
    materialize_background_result,
    materialize_background_results,
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


@pytest.mark.asyncio
async def test_active_ai_batches_preserve_empty_input_without_callbacks() -> None:
    def unexpected(*_args, **_kwargs):
        pytest.fail("empty active batches must not invoke dependencies")

    assert await execute_active_ai_batches(
        [],
        average_pressure=0.0,
        pressure_summary="",
        map_changes=[],
        major_events=[],
        turn_index=0,
        stream_callback=None,
        build_batch_payload=unexpected,
        call_batch_ai=unexpected,
        parse_batch_results=unexpected,
        staggered_gather=unexpected,
    ) == []


@pytest.mark.asyncio
async def test_active_ai_batches_preserve_stable_batches_and_arguments() -> None:
    entries = [{"name": f"entry-{idx}"} for idx in range(5)]
    callback = lambda *_args: None
    calls: list[tuple] = []
    gather_kwargs: dict = {}

    def build_payload(
        batch_entries,
        average_pressure,
        pressure_summary,
        map_changes,
        major_events,
        turn_index,
    ):
        calls.append(("build", batch_entries))
        assert average_pressure == 2.5
        assert pressure_summary == "pressure"
        assert map_changes == ["map"]
        assert major_events == ["event"]
        assert turn_index == 9
        return {"batch": batch_entries}

    async def call_batch(payload, stream_callback, batch_entries):
        calls.append(("call", batch_entries))
        assert payload == {"batch": batch_entries}
        assert stream_callback is callback
        return {"raw": batch_entries}

    def parse_batch(raw, batch_entries):
        calls.append(("parse", batch_entries))
        assert raw == {"raw": batch_entries}
        return [f"parsed-{entry['name']}" for entry in batch_entries]

    async def gather(coroutines, **kwargs):
        gather_kwargs.update(kwargs)
        return [await coroutine for coroutine in coroutines]

    results = await execute_active_ai_batches(
        entries,
        average_pressure=2.5,
        pressure_summary="pressure",
        map_changes=["map"],
        major_events=["event"],
        turn_index=9,
        stream_callback=callback,
        build_batch_payload=build_payload,
        call_batch_ai=call_batch,
        parse_batch_results=parse_batch,
        staggered_gather=gather,
    )

    batches = [
        entries[0:2],
        entries[2:4],
        entries[4:5],
    ]
    assert calls == [
        ("build", batches[0]),
        ("call", batches[0]),
        ("parse", batches[0]),
        ("build", batches[1]),
        ("call", batches[1]),
        ("parse", batches[1]),
        ("build", batches[2]),
        ("call", batches[2]),
        ("parse", batches[2]),
    ]
    assert gather_kwargs == {
        "interval": 1.5,
        "max_concurrent": 20,
        "task_name": "分化批次",
        "event_callback": callback,
    }
    assert results == [f"parsed-entry-{idx}" for idx in range(5)]


@pytest.mark.asyncio
async def test_active_ai_batches_repeat_batch_exception_per_entry() -> None:
    entries = [{"name": f"entry-{idx}"} for idx in range(5)]
    batch_error = RuntimeError("first batch failed")

    def build_payload(batch_entries, *_args):
        return {"batch": batch_entries}

    async def call_batch(payload, _stream_callback, _batch_entries):
        if payload["batch"][0] is entries[0]:
            raise batch_error
        return payload

    def parse_batch(payload, batch_entries):
        return [entry["name"] for entry in batch_entries]

    async def gather(coroutines, **_kwargs):
        gathered = []
        for coroutine in coroutines:
            try:
                gathered.append(await coroutine)
            except Exception as exc:
                gathered.append(exc)
        return gathered

    results = await execute_active_ai_batches(
        entries,
        average_pressure=1.0,
        pressure_summary="pressure",
        map_changes=[],
        major_events=[],
        turn_index=3,
        stream_callback=None,
        build_batch_payload=build_payload,
        call_batch_ai=call_batch,
        parse_batch_results=parse_batch,
        staggered_gather=gather,
    )

    assert results == [
        batch_error,
        batch_error,
        "entry-2",
        "entry-3",
        "entry-4",
    ]
    assert results[0] is batch_error
    assert results[1] is batch_error


@pytest.mark.asyncio
async def test_fallback_description_enhancement_preserves_empty_noop() -> None:
    class UnexpectedEnhancer:
        def queue_for_enhancement(self, **_kwargs):
            pytest.fail("empty fallback list must not queue work")

        async def process_queue_async(self, **_kwargs):
            pytest.fail("empty fallback list must not process work")

    pending: list[tuple] = []

    await enhance_rule_fallback_descriptions(
        pending,
        description_enhancer=UnexpectedEnhancer(),
        upsert_species=lambda _species: pytest.fail(
            "empty fallback list must not persist work"
        ),
    )

    assert pending == []


@pytest.mark.asyncio
async def test_fallback_description_enhancement_preserves_order_and_args(
) -> None:
    species_one = object()
    species_two = object()
    parent_one = object()
    parent_two = object()
    enhanced_one = object()
    enhanced_two = object()
    pending = [
        (species_one, parent_one, "type-one"),
        (species_two, parent_two, "type-two"),
    ]
    queued: list[dict] = []
    process_kwargs: dict = {}
    persisted: list[object] = []

    class Enhancer:
        def queue_for_enhancement(self, **kwargs):
            queued.append(kwargs)

        async def process_queue_async(self, **kwargs):
            process_kwargs.update(kwargs)
            return [enhanced_one, enhanced_two]

    await enhance_rule_fallback_descriptions(
        pending,
        description_enhancer=Enhancer(),
        upsert_species=persisted.append,
    )

    assert queued == [
        {
            "species": species_one,
            "parent": parent_one,
            "speciation_type": "type-one",
            "is_hybrid": False,
        },
        {
            "species": species_two,
            "parent": parent_two,
            "speciation_type": "type-two",
            "is_hybrid": False,
        },
    ]
    assert process_kwargs == {
        "max_items": 20,
        "timeout_per_item": 25.0,
    }
    assert persisted == [enhanced_one, enhanced_two]
    assert pending == []


@pytest.mark.asyncio
async def test_fallback_description_enhancement_swallows_failure_and_clears(
) -> None:
    pending = [(object(), object(), "type-one")]
    processing_error = RuntimeError("enhancement failed")

    class FailingEnhancer:
        def queue_for_enhancement(self, **_kwargs):
            return None

        async def process_queue_async(self, **_kwargs):
            raise processing_error

    await enhance_rule_fallback_descriptions(
        pending,
        description_enhancer=FailingEnhancer(),
        upsert_species=lambda _species: pytest.fail(
            "failed enhancement must not persist"
        ),
    )

    assert pending == []


def test_background_result_materialization_preserves_callbacks_and_state(
) -> None:
    parent = SimpleNamespace(common_name="Parent", lineage_code="PARENT")
    created = SimpleNamespace(
        common_name="Created",
        created_turn=8,
        genus_code="GENUS",
    )
    stored = SimpleNamespace(
        common_name="Stored",
        created_turn=8,
        genus_code="GENUS",
    )
    entry = {
        "ctx": {
            "parent": parent,
            "new_code": "CHILD",
            "population": 123,
            "speciation_type": "adaptive",
            "assigned_tiles": {4, 5},
        }
    }
    raw_content = {"common_name": "Raw"}
    validated_content = {"common_name": "Validated"}
    calls: list[tuple] = []
    queued: list[tuple] = []
    persisted: list[object] = []
    timestamp = object()
    lineage_event = object()
    branching_event = object()

    class GeneLibrary:
        def inherit_dormant_genes(self, parent_arg, child_arg, genus_arg):
            calls.append(("inherit_genes", parent_arg, child_arg, genus_arg))

    gene_library = GeneLibrary()

    class GeneOwner:
        @property
        def gene_library_service(self):
            calls.append(("get_gene_library",))
            return gene_library

    class GenusRepository:
        def get_by_code(self, code):
            calls.append(("get_genus", code))
            return "GENUS-OBJECT"

    def validate_and_fix(content, parent_arg, preprocess_result):
        calls.append(
            ("validate", content, parent_arg, preprocess_result)
        )
        return validated_content

    def create_species(**kwargs):
        calls.append(("create", kwargs))
        return created

    def upsert_species(species):
        calls.append(
            ("upsert", species, getattr(species, "is_background", None))
        )
        persisted.append(species)
        return stored

    def inherit_habitat_distribution(**kwargs):
        calls.append(("inherit_habitat", kwargs))

    def try_speciation_breakthrough(species, turn):
        calls.append(("breakthrough", species, turn))
        return "activated-gene"

    def lineage_event_factory(**kwargs):
        calls.append(("lineage_factory", kwargs))
        return lineage_event

    def log_lineage_event(event):
        calls.append(("log_event", event))

    def utcnow():
        calls.append(("utcnow",))
        return timestamp

    def branching_event_factory(**kwargs):
        calls.append(("branching_factory", kwargs))
        return branching_event

    result = materialize_background_result(
        entry,
        raw_content,
        turn_index=8,
        average_pressure=3.25,
        validate_and_fix=validate_and_fix,
        create_species=create_species,
        rule_fallback_species=queued,
        random_uniform=lambda low, high: (
            calls.append(("random", low, high)) or 0.4
        ),
        inherit_habitat_distribution=inherit_habitat_distribution,
        gene_library_service_owner=GeneOwner(),
        genus_repository=GenusRepository(),
        try_speciation_breakthrough=try_speciation_breakthrough,
        upsert_species=upsert_species,
        log_lineage_event=log_lineage_event,
        lineage_event_factory=lineage_event_factory,
        branching_event_factory=branching_event_factory,
        utcnow=utcnow,
    )

    assert result is branching_event
    assert created.is_background is True
    assert queued == [(stored, parent, "adaptive")]
    assert persisted == [created, stored, stored]
    assert [call[0] for call in calls] == [
        "validate",
        "create",
        "upsert",
        "random",
        "inherit_habitat",
        "get_gene_library",
        "get_gene_library",
        "get_genus",
        "get_gene_library",
        "inherit_genes",
        "upsert",
        "breakthrough",
        "upsert",
        "lineage_factory",
        "log_event",
        "utcnow",
        "branching_factory",
    ]
    assert calls[1][1] == {
        "parent": parent,
        "new_code": "CHILD",
        "survivors": 123,
        "turn_index": 8,
        "ai_payload": validated_content,
        "average_pressure": 3.25,
        "speciation_type": "adaptive",
    }
    assert calls[4][1] == {
        "parent": parent,
        "child": stored,
        "turn_index": 8,
        "assigned_tiles": {4, 5},
        "reproduction_bonus": 0.4,
    }
    assert calls[13][1] == {
        "lineage_code": "CHILD",
        "event_type": "speciation",
        "payload": {"parent": "PARENT", "turn": 8},
    }
    assert calls[16][1] == {
        "parent_lineage": "PARENT",
        "new_lineage": "CHILD",
        "description": "Parent在压力3.2条件下分化出CHILD（背景物种）",
        "timestamp": timestamp,
        "reason": "Parent种群在演化压力下发生生态位分化",
    }


def test_background_result_materialization_preserves_optional_noop_branches(
) -> None:
    parent = SimpleNamespace(common_name="Parent", lineage_code="PARENT")
    child = SimpleNamespace(
        common_name="Child",
        created_turn=1,
        genus_code=None,
    )
    upserts: list[object] = []
    genus_lookups: list[str] = []
    entry = {
        "ctx": {
            "parent": parent,
            "new_code": "CHILD",
            "population": 5,
            "speciation_type": "type",
        }
    }

    result = materialize_background_result(
        entry,
        {},
        turn_index=1,
        average_pressure=0.0,
        validate_and_fix=lambda content, *_args, **_kwargs: content,
        create_species=lambda **_kwargs: child,
        rule_fallback_species=[],
        random_uniform=lambda _low, _high: 0.3,
        inherit_habitat_distribution=lambda **kwargs: (
            pytest.fail("assigned_tiles must default to an empty set")
            if kwargs["assigned_tiles"] != set()
            else None
        ),
        gene_library_service_owner=SimpleNamespace(),
        genus_repository=SimpleNamespace(
            get_by_code=lambda code: genus_lookups.append(code)
        ),
        try_speciation_breakthrough=lambda _species, _turn: None,
        upsert_species=lambda species: (
            upserts.append(species) or species
        ),
        log_lineage_event=lambda _event: None,
        lineage_event_factory=lambda **kwargs: kwargs,
        branching_event_factory=lambda **kwargs: kwargs,
        utcnow=lambda: "now",
    )

    assert upserts == [child]
    assert genus_lookups == []
    assert result["timestamp"] == "now"


def test_background_results_materialization_preserves_loop_order(
    monkeypatch,
) -> None:
    first = ({"name": "entry-one"}, {"name": "content-one"})
    second = ({"name": "entry-two"}, {"name": "content-two"})
    calls: list[tuple] = []

    def materialize_one(entry, content, **kwargs):
        calls.append((entry, content, kwargs))
        return f"event-{entry['name']}"

    monkeypatch.setattr(
        speciation_process_module,
        "materialize_background_result",
        materialize_one,
    )

    marker = object()
    existing_events = ["existing-active-event"]
    result = materialize_background_results(
        [first, second],
        result_events=existing_events,
        marker=marker,
    )

    assert result is None
    assert existing_events == [
        "existing-active-event",
        "event-entry-one",
        "event-entry-two",
    ]
    assert calls == [
        (first[0], first[1], {"marker": marker}),
        (second[0], second[1], {"marker": marker}),
    ]


def test_background_results_materialization_preserves_partial_append_on_error(
    monkeypatch,
) -> None:
    first = ({"name": "entry-one"}, {"name": "content-one"})
    second = ({"name": "entry-two"}, {"name": "content-two"})
    processing_error = RuntimeError("second entry failed")

    def materialize_one(entry, _content, **_kwargs):
        if entry is second[0]:
            raise processing_error
        return "event-entry-one"

    monkeypatch.setattr(
        speciation_process_module,
        "materialize_background_result",
        materialize_one,
    )
    existing_events = ["existing-active-event"]

    with pytest.raises(RuntimeError) as exc_info:
        materialize_background_results(
            [first, second],
            result_events=existing_events,
        )

    assert exc_info.value is processing_error
    assert existing_events == [
        "existing-active-event",
        "event-entry-one",
    ]
