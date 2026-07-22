import copy
import logging
from types import SimpleNamespace

from ..speciation import SpeciationService
from ..speciation_dormant_genes import (
    process_ai_activated_genes,
    summarize_dormant_genes,
)


def _species_with(dormant_genes: dict) -> SimpleNamespace:
    return SimpleNamespace(dormant_genes=dormant_genes)


def _activation_species() -> SimpleNamespace:
    return SimpleNamespace(
        common_name="测试物种",
        abstract_traits={},
        organs={},
        dormant_genes={
            "traits": {
                "耐寒增强": {
                    "activated": False,
                    "potential_value": 12.0,
                    "dominance": "recessive",
                },
                "有害甲": {
                    "activated": False,
                    "mutation_effect": "harmful",
                },
            },
            "organs": {
                "厚甲": {
                    "activated": False,
                    "development_stage": 1,
                    "organ_data": {
                        "category": "defense",
                        "type": "shell",
                        "parameters": {"hardness": 2.0},
                    },
                }
            },
        },
    )


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


def test_ai_activated_genes_preserve_trait_and_organ_mutations() -> None:
    species = _activation_species()

    result = process_ai_activated_genes(
        species, ["耐寒", "有害甲", "厚甲"], turn_index=9
    )

    assert result == 2
    assert species.abstract_traits == {"耐寒增强": 3.0}
    assert species.dormant_genes["traits"]["耐寒增强"] == {
        "activated": True,
        "potential_value": 12.0,
        "dominance": "recessive",
        "activation_turn": 9,
        "expressed_value": 3.0,
    }
    assert species.dormant_genes["traits"]["有害甲"] == {
        "activated": False,
        "mutation_effect": "harmful",
    }
    assert species.dormant_genes["organs"]["厚甲"] == {
        "activated": True,
        "development_stage": 2,
        "stage_start_turn": 9,
        "activation_turn": 9,
        "organ_data": {
            "category": "defense",
            "type": "shell",
            "parameters": {"hardness": 2.0},
        },
    }
    assert species.organs == {
        "defense": {
            "type": "shell",
            "parameters": {
                "hardness": 2.0,
                "efficiency_modifier": 0.60,
            },
            "acquired_turn": 9,
            "is_active": True,
            "maturity": 0.60,
            "development_stage": 2,
        }
    }


def test_ai_activated_genes_preserve_warning_logger_category(caplog) -> None:
    species = _activation_species()

    with caplog.at_level(
        logging.WARNING, logger="app.services.species.speciation"
    ):
        result = process_ai_activated_genes(species, ["有害甲"], 9)

    assert result == 0
    assert [
        (record.name, record.getMessage()) for record in caplog.records
    ] == [
        (
            "app.services.species.speciation",
            "[AI基因激活] 阻止激活有害突变: 有害甲",
        )
    ]


def test_speciation_service_keeps_ai_activated_genes_method() -> None:
    direct_species = _activation_species()
    service_species = copy.deepcopy(direct_species)
    service = object.__new__(SpeciationService)

    direct_result = process_ai_activated_genes(
        direct_species, ["耐寒", "厚甲"], 9
    )
    service_result = service._process_ai_activated_genes(
        service_species, ["耐寒", "厚甲"], 9
    )

    assert service_result == direct_result
    assert vars(service_species) == vars(direct_species)
