from semantic_layer_benchmark.catalog import COUNTRIES, generate_catalog, write_catalog


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


def test_core_catalog_contains_explicit_lineage_relationships() -> None:
    records = {record.key: record for record in generate_catalog(10)}
    sales = records["argentina_gold_sales_daily"]
    assert "DERIVED_FROM argentina_silver_orders_enriched" in sales.relationships
    assert "Relationships:" in sales.context()


def test_write_catalog_replaces_stale_knowledge_base_documents(tmp_path) -> None:
    documents = tmp_path / "knowledge-base-documents"
    documents.mkdir()
    (documents / "stale.md").write_text("stale", encoding="utf-8")

    records = generate_catalog(10)
    write_catalog(records, tmp_path)

    generated = list(documents.glob("*.md"))
    assert len(generated) == len(records)
    assert not (documents / "stale.md").exists()
