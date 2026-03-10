"""Тесты для dataset/embedding_generation.py (serialize_row, COLUMN_DESCRIPTION_PROMPT)."""

import numpy as np
import pandas as pd
import pytest

from table_unifier.dataset.embedding_generation import serialize_row


class TestSerializeRow:
    def test_basic(self):
        row = pd.Series({"id": 1, "title": "iPhone", "brand": "Apple"})
        result = serialize_row(row)
        assert "title: iPhone" in result
        assert "brand: Apple" in result
        assert "id" not in result

    def test_custom_columns(self):
        row = pd.Series({"id": 1, "title": "Phone", "brand": "X", "price": 100})
        result = serialize_row(row, columns=["title", "price"])
        assert "title: Phone" in result
        assert "price: 100" in result
        assert "brand" not in result

    def test_nan_skipped(self):
        row = pd.Series({"id": 1, "title": "Phone", "brand": None})
        result = serialize_row(row)
        assert "brand" not in result

    def test_empty_string_skipped(self):
        row = pd.Series({"id": 1, "title": "Phone", "brand": "  "})
        result = serialize_row(row)
        assert "brand" not in result

    def test_separator(self):
        row = pd.Series({"id": 1, "a": "x", "b": "y"})
        result = serialize_row(row)
        assert " | " in result

    def test_all_nan(self):
        row = pd.Series({"id": 1, "a": None, "b": float("nan")})
        result = serialize_row(row)
        assert result == ""
