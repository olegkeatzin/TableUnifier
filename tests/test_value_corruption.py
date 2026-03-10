"""Тесты для dataset/value_corruption.py."""

import random

import pandas as pd
import pytest

from table_unifier.dataset.value_corruption import (
    add_typo,
    change_format,
    corrupt_dataframe,
    corrupt_row,
    corrupt_value,
    drop_tokens,
)


class TestAddTypo:
    def test_empty_string(self):
        assert add_typo("") == ""

    def test_single_char(self):
        assert add_typo("a") == "a"

    def test_preserves_length(self):
        text = "hello world"
        result = add_typo(text, char_prob=1.0)
        assert len(result) == len(text)

    def test_deterministic_with_seed(self):
        random.seed(42)
        r1 = add_typo("Naruto Shippuden", char_prob=0.5)
        random.seed(42)
        r2 = add_typo("Naruto Shippuden", char_prob=0.5)
        assert r1 == r2

    def test_zero_prob_no_change(self):
        assert add_typo("hello", char_prob=0.0) == "hello"


class TestChangeFormat:
    def test_year_format(self):
        assert change_format("2014") == "'14"

    def test_year_format_2023(self):
        assert change_format("2023") == "'23"

    def test_large_number(self):
        result = change_format("1234.5")
        assert "," in result or result == "1234.5"

    def test_small_number_unchanged(self):
        assert change_format("42") == "42"

    def test_non_numeric_unchanged(self):
        assert change_format("hello") == "hello"

    def test_whitespace_handling(self):
        assert change_format("  2014  ") == "'14"


class TestDropTokens:
    def test_single_word_preserved(self):
        assert drop_tokens("hello") == "hello"

    def test_all_drop_keeps_first(self):
        random.seed(42)
        result = drop_tokens("a b c d e", drop_prob=1.0)
        assert result == "a"

    def test_zero_drop_no_change(self):
        assert drop_tokens("hello world test", drop_prob=0.0) == "hello world test"

    def test_result_subset_of_original(self):
        random.seed(0)
        original = "one two three four five"
        result = drop_tokens(original, drop_prob=0.5)
        for word in result.split():
            assert word in original.split()


class TestCorruptValue:
    def test_returns_string(self):
        result = corrupt_value("test")
        assert isinstance(result, str)

    def test_no_corruption_possible(self):
        # Вероятности дают prob > 1 — всегда typo
        result = corrupt_value("abc", typo_prob=1.0, format_prob=0.0, drop_prob=0.0)
        assert isinstance(result, str)
        assert len(result) == 3


class TestCorruptRow:
    def test_returns_series(self):
        row = pd.Series({"a": "hello", "b": "world"})
        result = corrupt_row(row, corruption_prob=0.5)
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b"]

    def test_no_corruption(self):
        row = pd.Series({"a": "hello", "b": "world"})
        result = corrupt_row(row, corruption_prob=0.0)
        assert result["a"] == "hello"
        assert result["b"] == "world"


class TestCorruptDataframe:
    def test_shape_preserved(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]})
        result = corrupt_dataframe(df)
        assert result.shape == df.shape

    def test_no_corruption(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
        result = corrupt_dataframe(df, row_prob=0.0)
        pd.testing.assert_frame_equal(result, df)

    def test_original_unchanged(self):
        df = pd.DataFrame({"a": ["hello", "world"], "b": [10, 20]})
        original = df.copy()
        corrupt_dataframe(df, row_prob=1.0, cell_prob=1.0)
        pd.testing.assert_frame_equal(df, original)
