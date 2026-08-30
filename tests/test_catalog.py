from semantic_layer_benchmark.catalog import COUNTRIES, generate_catalog


def test_catalog_has_requested_size_per_country() -> None:
    records = generate_catalog(20)
    assert len(records) == 20 * len(COUNTRIES)
    for country in COUNTRIES:
        assert sum(record.country == country for record in records) == 20


def test_catalog_contains_benchmark_ground_truth() -> None:
    keys = {record.key for record in generate_catalog(10)}
    assert "argentina_gold_sales_daily" in keys
    assert "mexico_gold_customer_360" in keys
    assert "united_states_gold_inventory_position" in keys


def test_context_contains_semantics_and_metadata() -> None:
    record = generate_catalog(10)[0]
    text = record.context()
    assert record.key in text
    assert record.country_name in text
    assert record.layer in text
    assert record.domain in text
