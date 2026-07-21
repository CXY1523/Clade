"""Repository-isolation coverage for the gene-flow stage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from ...models.genus import Genus
from ...models.species import Species
from ..context import SimulationContext
from ..stages import GeneFlowStage


class _InMemoryGenusRepository:
    def __init__(self, genus: Genus) -> None:
        self.genus = genus
        self.requested_codes: list[str] = []

    def get_by_code(self, code: str) -> Genus | None:
        self.requested_codes.append(code)
        return self.genus if self.genus.code == code else None


class _RecordingGeneFlowService:
    def __init__(self) -> None:
        self.calls: list[tuple[Genus, list[Species]]] = []

    def apply_gene_flow(self, genus: Genus, species: list[Species]) -> int:
        self.calls.append((genus, species))
        return 0


def _species(lineage_code: str) -> Species:
    return Species(
        lineage_code=lineage_code,
        latin_name=f"{lineage_code.title()} testus",
        common_name=lineage_code.lower(),
        description="Gene-flow repository test species",
        morphology_stats={},
        abstract_traits={},
        hidden_traits={},
        ecological_vector=[],
        genus_code="GEN-TEST",
        updated_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
    )


def test_gene_flow_stage_uses_injected_genus_repository() -> None:
    genus = Genus(
        code="GEN-TEST",
        name_latin="Testus",
        name_common="test genus",
    )
    repository = _InMemoryGenusRepository(genus)
    gene_flow_service = _RecordingGeneFlowService()
    species_batch = [_species("SP-A"), _species("SP-B")]
    context = SimulationContext(species_batch=species_batch)
    engine = SimpleNamespace(gene_flow_service=gene_flow_service)

    asyncio.run(
        GeneFlowStage(genus_repository=repository).execute(context, engine)
    )

    assert repository.requested_codes == ["GEN-TEST"]
    assert gene_flow_service.calls == [(genus, species_batch)]
    assert context.gene_flow_count == 0
