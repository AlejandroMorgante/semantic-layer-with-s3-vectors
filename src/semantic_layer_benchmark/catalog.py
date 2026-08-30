from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

COUNTRIES = {
    "argentina": "Argentina",
    "mexico": "Mexico",
    "united_states": "United States",
}

LAYERS = ("bronze", "silver", "gold")
DOMAINS = (
    "sales",
    "customers",
    "inventory",
    "marketing",
    "logistics",
    "finance",
    "stores",
    "products",
    "operations",
    "risk",
)

CORE_TABLES = (
    (
        "gold_sales_daily",
        "gold",
        "sales",
        "Certified daily commercial sales facts with gross sales, discounts, returns, net "
        "sales, units, currency, fiscal periods, stores, channels, and products.",
        [
            "business_date",
            "fiscal_quarter",
            "store_id",
            "product_id",
            "gross_sales",
            "net_sales",
            "units_sold",
        ],
    ),
    (
        "gold_customer_360",
        "gold",
        "customers",
        "Certified customer 360 view with active status, lifecycle segment, acquisition "
        "channel, loyalty tier, purchase frequency, and lifetime value.",
        [
            "customer_id",
            "active_flag",
            "customer_segment",
            "loyalty_tier",
            "lifetime_value",
        ],
    ),
    (
        "gold_inventory_position",
        "gold",
        "inventory",
        "Current certified inventory position by product and location, including available "
        "stock, reserved units, safety stock, and out-of-stock indicators.",
        [
            "snapshot_at",
            "location_id",
            "product_id",
            "available_units",
            "out_of_stock_flag",
        ],
    ),
    (
        "gold_marketing_performance",
        "gold",
        "marketing",
        "Certified campaign performance with spend, impressions, conversions, attributed "
        "revenue, return on ad spend, channel, and campaign dimensions.",
        [
            "campaign_id",
            "channel",
            "spend",
            "conversions",
            "attributed_revenue",
            "return_on_ad_spend",
        ],
    ),
    (
        "gold_logistics_delivery",
        "gold",
        "logistics",
        "Certified delivery performance facts with promised and actual delivery dates, delay "
        "duration, carrier, destination, and on-time delivery rate.",
        [
            "shipment_id",
            "carrier_id",
            "promised_date",
            "delivered_date",
            "delay_days",
            "on_time_flag",
        ],
    ),
    (
        "gold_finance_profitability",
        "gold",
        "finance",
        "Certified financial profitability facts with revenue, cost of goods sold, gross "
        "margin, operating expenses, and operating profit by fiscal period.",
        [
            "fiscal_period",
            "revenue",
            "cost_of_goods_sold",
            "gross_margin",
            "operating_profit",
        ],
    ),
    (
        "bronze_pos_transactions",
        "bronze",
        "sales",
        "Raw immutable point-of-sale transaction events as received from source systems, "
        "including original payload identifiers and ingestion timestamps.",
        ["source_event_id", "raw_payload", "source_system", "ingested_at"],
    ),
    (
        "silver_orders_enriched",
        "silver",
        "sales",
        "Cleaned, deduplicated, and enriched order records before business aggregation, with "
        "normalized customer, product, payment, and fulfillment attributes.",
        [
            "order_id",
            "customer_id",
            "product_id",
            "order_status",
            "payment_status",
            "fulfillment_status",
        ],
    ),
    (
        "gold_store_performance",
        "gold",
        "stores",
        "Certified store scorecard with revenue, transactions, comparable-store growth, "
        "conversion, labor productivity, format, and region.",
        [
            "store_id",
            "fiscal_period",
            "revenue",
            "transactions",
            "comparable_store_growth",
        ],
    ),
    (
        "gold_product_performance",
        "gold",
        "products",
        "Certified product performance with sales, units, margin, returns, category, brand, "
        "and product hierarchy attributes.",
        [
            "product_id",
            "category",
            "brand",
            "net_sales",
            "gross_margin",
            "return_rate",
        ],
    ),
)


@dataclass(frozen=True)
class TableRecord:
    key: str
    country: str
    country_name: str
    layer: str
    domain: str
    description: str
    columns: list[str]

    def context(self) -> str:
        return (
            f"Table: {self.key}\n"
            f"Country: {self.country_name}\n"
            f"Medallion layer: {self.layer}\n"
            f"Business domain: {self.domain}\n"
            f"Description: {self.description}\n"
            f"Important columns: {', '.join(self.columns)}"
        )


def generate_catalog(tables_per_country: int = 100) -> list[TableRecord]:
    if tables_per_country < len(CORE_TABLES):
        raise ValueError(f"tables_per_country must be at least {len(CORE_TABLES)}")

    records: list[TableRecord] = []
    for country, country_name in COUNTRIES.items():
        country_records = [
            TableRecord(
                key=f"{country}_{suffix}",
                country=country,
                country_name=country_name,
                layer=layer,
                domain=domain,
                description=description,
                columns=columns,
            )
            for suffix, layer, domain, description, columns in CORE_TABLES
        ]

        number = 1
        while len(country_records) < tables_per_country:
            layer = LAYERS[(number - 1) % len(LAYERS)]
            domain = DOMAINS[(number - 1) % len(DOMAINS)]
            country_records.append(
                TableRecord(
                    key=f"{country}_{layer}_{domain}_activity_{number:03d}",
                    country=country,
                    country_name=country_name,
                    layer=layer,
                    domain=domain,
                    description=(
                        f"{layer.title()} {domain} operational dataset for {country_name}. "
                        f"Contains activity records for internal process {number:03d}; it is "
                        "not the certified aggregate for general business reporting."
                    ),
                    columns=[
                        "record_id",
                        "event_timestamp",
                        "source_system",
                        f"{domain}_status",
                        "updated_at",
                    ],
                )
            )
            number += 1
        records.extend(country_records)
    return records


def write_catalog(records: list[TableRecord], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "catalog.json"
    markdown_path = output_dir / "catalog.md"
    json_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = ["# Data platform table catalog", ""]
    for record in records:
        markdown.extend([f"## {record.key}", "", record.context(), ""])
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    return json_path, markdown_path


def read_catalog(path: Path) -> list[TableRecord]:
    return [TableRecord(**record) for record in json.loads(path.read_text(encoding="utf-8"))]
