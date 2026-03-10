"""Тесты для dataset/schema_augmentation.py."""

from unittest.mock import MagicMock

from table_unifier.dataset.schema_augmentation import (
    apply_schema_injection,
    augment_schema,
    generate_column_synonyms,
)


class TestGenerateColumnSynonyms:
    def test_returns_list(self):
        client = MagicMock()
        client.generate.return_value = "Product Name\nItem Title\nGoods Label\n"
        result = generate_column_synonyms(client, "title", n=3)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == "Product Name"

    def test_truncates_to_n(self):
        client = MagicMock()
        client.generate.return_value = "a\nb\nc\nd\ne\nf\n"
        result = generate_column_synonyms(client, "col", n=2)
        assert len(result) == 2

    def test_empty_response(self):
        client = MagicMock()
        client.generate.return_value = ""
        result = generate_column_synonyms(client, "col", n=3)
        assert result == []

    def test_strips_whitespace(self):
        client = MagicMock()
        client.generate.return_value = "  Name  \n  Title  \n"
        result = generate_column_synonyms(client, "col", n=5)
        assert result == ["Name", "Title"]


class TestAugmentSchema:
    def test_generates_for_all_columns(self):
        client = MagicMock()
        client.generate.return_value = "syn1\nsyn2\n"
        result = augment_schema(client, ["title", "brand"], n_variants=2)
        assert "title" in result
        assert "brand" in result
        assert len(result["title"]) == 2
        assert client.generate.call_count == 2


class TestApplySchemaInjection:
    def test_basic_injection(self):
        synonym_map = {
            "title": ["Product Name", "Item Title"],
            "brand": ["Manufacturer", "Company"],
        }
        result = apply_schema_injection(["title", "brand"], synonym_map, variant_idx=0)
        assert result == {"title": "Product Name", "brand": "Manufacturer"}

    def test_variant_idx(self):
        synonym_map = {"title": ["A", "B", "C"]}
        result = apply_schema_injection(["title"], synonym_map, variant_idx=2)
        assert result["title"] == "C"

    def test_variant_idx_clamp(self):
        synonym_map = {"title": ["A"]}
        result = apply_schema_injection(["title"], synonym_map, variant_idx=10)
        assert result["title"] == "A"

    def test_missing_column_returns_original(self):
        result = apply_schema_injection(["unknown"], {}, variant_idx=0)
        assert result["unknown"] == "unknown"
