from types import SimpleNamespace

from ..speciation import SpeciationService
from ..speciation_dormant_genes import summarize_dormant_genes


def _species_with(dormant_genes: dict) -> SimpleNamespace:
    return SimpleNamespace(dormant_genes=dormant_genes)


def test_dormant_gene_summary_reports_empty_library() -> None:
    assert summarize_dormant_genes(_species_with({})) == "无休眠基因"


def test_dormant_gene_summary_reports_no_available_genes() -> None:
    species = _species_with(
        {
            "traits": {"已激活特质": {"activated": True}},
            "organs": {
                "成熟器官": {"activated": True, "development_stage": 3}
            },
        }
    )

    assert summarize_dormant_genes(species) == "无可激活基因"


def test_dormant_gene_summary_preserves_scoring_and_text() -> None:
    species = _species_with(
        {
            "traits": {
                "耐寒增强": {
                    "potential_value": 10,
                    "pressure_types": ["cold"],
                    "dominance": "dominant",
                },
                "普通代谢": {
                    "potential_value": 15,
                    "pressure_types": ["competition"],
                    "dominance": "recessive",
                },
                "有害甲": {"mutation_effect": "harmful"},
                "有害乙": {"mutation_effect": "lethal"},
            },
            "organs": {
                "厚甲": {
                    "pressure_types": ["cold"],
                    "development_stage": 1,
                    "organ_data": {"category": "defense"},
                }
            },
        }
    )

    assert summarize_dormant_genes(species, ["cold"], 8) == (
        "推荐特质: ⭐耐寒增强[显](10), 普通代谢[隐](15)\n"
        "推荐器官: 厚甲(defense)[初级]\n"
        "⚠️ 遗传负荷: 有害甲, 有害乙 (避免激活)\n"
        "(⭐=匹配当前压力优先激活; [显]=显性易表达; "
        "[隐]=隐性需高压激活)"
    )


def test_speciation_service_keeps_dormant_gene_summary_method() -> None:
    species = _species_with({})
    service = object.__new__(SpeciationService)

    assert service._summarize_dormant_genes(species) == summarize_dormant_genes(
        species
    )
