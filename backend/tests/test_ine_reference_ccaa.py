from app.data.ine_reference import (
    AUTONOMOUS_COMMUNITY_DISPLAY_NAMES,
    AUTONOMOUS_COMMUNITY_PROVINCES,
    CCAA_POPULATION,
    PROVINCE_POPULATION,
    resolve_autonomous_community,
)


class TestAutonomousCommunityProvinces:
    def test_every_province_referenced_exists_in_province_population(self):
        for ccaa, provinces in AUTONOMOUS_COMMUNITY_PROVINCES.items():
            for province in provinces:
                assert province in PROVINCE_POPULATION, f"{province} (de {ccaa}) no está en PROVINCE_POPULATION"

    def test_every_ccaa_has_a_display_name(self):
        assert set(AUTONOMOUS_COMMUNITY_PROVINCES) == set(AUTONOMOUS_COMMUNITY_DISPLAY_NAMES)

    def test_canarias_has_its_two_real_provinces(self):
        assert set(AUTONOMOUS_COMMUNITY_PROVINCES["canarias"]) == {"las palmas", "santa cruz de tenerife"}


class TestCcaaPopulation:
    def test_every_ccaa_has_a_population(self):
        assert set(CCAA_POPULATION) == set(AUTONOMOUS_COMMUNITY_PROVINCES)

    def test_canarias_population_is_sum_of_its_provinces(self):
        expected = PROVINCE_POPULATION["las palmas"] + PROVINCE_POPULATION["santa cruz de tenerife"]
        assert CCAA_POPULATION["canarias"] == expected

    def test_single_province_ccaa_population_matches_its_province(self):
        assert CCAA_POPULATION["madrid"] == PROVINCE_POPULATION["madrid"]
        assert CCAA_POPULATION["asturias"] == PROVINCE_POPULATION["asturias"]


class TestResolveAutonomousCommunity:
    def test_english_name_resolves(self):
        assert resolve_autonomous_community("Canary Islands") == "canarias"

    def test_spanish_name_with_accents_resolves(self):
        assert resolve_autonomous_community("Canarias") == "canarias"
        assert resolve_autonomous_community("País Vasco") == "pais vasco"
        assert resolve_autonomous_community("Cataluña") == "cataluna"

    def test_alternative_spanish_and_english_variants_resolve_to_same_key(self):
        assert resolve_autonomous_community("Principado de Asturias") == "asturias"
        assert resolve_autonomous_community("Basque Country") == "pais vasco"
        assert resolve_autonomous_community("Catalonia") == "cataluna"
        assert resolve_autonomous_community("Community of Madrid") == "madrid"

    def test_unrecognized_region_returns_none(self):
        assert resolve_autonomous_community("Ile-de-France") is None
        assert resolve_autonomous_community("desconocido") is None
