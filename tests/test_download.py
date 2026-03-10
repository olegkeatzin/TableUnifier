"""Тесты для dataset/download.py (без реальной сети)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from table_unifier.dataset.download import (
    DATASETS,
    download_dataset,
    load_dataset,
    load_local_dataset,
    _read_magellan_csv,
    _detect_id_column,
    _split_labeled,
)


class TestDatasets:
    def test_known_datasets(self):
        assert "beer" in DATASETS
        assert "electronics" in DATASETS
        assert "restaurants1" in DATASETS
        assert len(DATASETS) >= 24

    def test_dataset_has_required_keys(self):
        for name, info in DATASETS.items():
            assert "url" in info, f"{name} missing url"
            assert "table_a" in info, f"{name} missing table_a"
            assert "table_b" in info, f"{name} missing table_b"

    def test_unknown_dataset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Неизвестный датасет"):
            download_dataset("NonExistent", tmp_path)


class TestReadMagellanCsv:
    def test_skips_comment_lines(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("# metadata line\nid,val\n1,a\n2,b", encoding="latin-1")
        df = _read_magellan_csv(csv)
        assert len(df) == 2
        assert list(df.columns) == ["id", "val"]

    def test_normal_csv(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("id,val\n1,a", encoding="latin-1")
        df = _read_magellan_csv(csv)
        assert len(df) == 1


class TestDetectIdColumn:
    def test_finds_id(self):
        df = pd.DataFrame({"id": [1], "val": ["a"]})
        assert _detect_id_column(df) == "id"

    def test_finds_underscore_id(self):
        df = pd.DataFrame({"_id": [1], "val": ["a"]})
        assert _detect_id_column(df) == "_id"

    def test_fallback_to_first(self):
        df = pd.DataFrame({"my_key": [1], "val": ["a"]})
        assert _detect_id_column(df) == "my_key"


class TestLoadDataset:
    def test_loads_with_known_name(self, tmp_path):
        csv_dir = tmp_path / "beer" / "csv_files"
        csv_dir.mkdir(parents=True)
        pd.DataFrame({"id": [0, 1], "name": ["a", "b"]}).to_csv(
            csv_dir / "beer_advocate.csv", index=False,
        )
        pd.DataFrame({"id": [0, 1], "name": ["x", "y"]}).to_csv(
            csv_dir / "rate_beer.csv", index=False,
        )
        pd.DataFrame({
            "ltable.id": [0], "rtable.id": [0], "gold": [1],
        }).to_csv(csv_dir / "labeled_data.csv", index=False)

        result = load_dataset(csv_dir, name="beer")
        assert "tableA" in result
        assert "tableB" in result
        assert "labeled_data" in result
        assert "train" in result
        assert "valid" in result
        assert "test" in result

    def test_splits_have_label_columns(self, tmp_path):
        csv_dir = tmp_path / "beer" / "csv_files"
        csv_dir.mkdir(parents=True)
        pd.DataFrame({"id": [0, 1], "v": ["a", "b"]}).to_csv(
            csv_dir / "beer_advocate.csv", index=False,
        )
        pd.DataFrame({"id": [0, 1], "v": ["x", "y"]}).to_csv(
            csv_dir / "rate_beer.csv", index=False,
        )
        pd.DataFrame({
            "ltable.id": [0, 1, 0, 1, 0],
            "rtable.id": [0, 1, 1, 0, 0],
            "gold": [1, 1, 0, 0, 1],
        }).to_csv(csv_dir / "labeled_data.csv", index=False)

        result = load_dataset(csv_dir, name="beer")
        for split in ("train", "valid", "test"):
            assert "ltable_id" in result[split].columns
            assert "rtable_id" in result[split].columns
            assert "label" in result[split].columns

    def test_auto_detects_name_from_parent(self, tmp_path):
        csv_dir = tmp_path / "beer" / "csv_files"
        csv_dir.mkdir(parents=True)
        pd.DataFrame({"id": [0]}).to_csv(csv_dir / "beer_advocate.csv", index=False)
        pd.DataFrame({"id": [0]}).to_csv(csv_dir / "rate_beer.csv", index=False)

        result = load_dataset(csv_dir)  # no name= argument
        assert "tableA" in result
        assert "tableB" in result

    def test_fallback_all_csvs(self, tmp_path):
        csv_dir = tmp_path / "unknown" / "csv_files"
        csv_dir.mkdir(parents=True)
        pd.DataFrame({"id": [0]}).to_csv(csv_dir / "foo.csv", index=False)
        pd.DataFrame({"id": [0]}).to_csv(csv_dir / "bar.csv", index=False)

        result = load_dataset(csv_dir)
        assert "foo" in result or "bar" in result

    def test_empty_dir(self, tmp_path):
        result = load_dataset(tmp_path)
        assert result == {}

    def test_renames_id_column(self, tmp_path):
        csv_dir = tmp_path / "beer" / "csv_files"
        csv_dir.mkdir(parents=True)
        pd.DataFrame({"_id": [0, 1], "v": ["a", "b"]}).to_csv(
            csv_dir / "beer_advocate.csv", index=False,
        )
        pd.DataFrame({"ID": [0, 1], "v": ["x", "y"]}).to_csv(
            csv_dir / "rate_beer.csv", index=False,
        )

        result = load_dataset(csv_dir, name="beer")
        assert "id" in result["tableA"].columns
        assert "id" in result["tableB"].columns


class TestLoadLocalDataset:
    def test_loads_three_files(self, tmp_path):
        df_a = pd.DataFrame({"id": [0], "title": ["x"]})
        df_b = pd.DataFrame({"id": [0], "title": ["y"]})
        df_l = pd.DataFrame({"ltable_id": [0], "rtable_id": [0], "label": [1]})

        df_a.to_csv(tmp_path / "a.csv", index=False)
        df_b.to_csv(tmp_path / "b.csv", index=False)
        df_l.to_csv(tmp_path / "l.csv", index=False)

        result = load_local_dataset(
            tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "l.csv",
        )
        assert "tableA" in result
        assert "tableB" in result
        assert "labels" in result


class TestDownloadDataset:
    def test_skips_existing(self, tmp_path):
        csv_dir = tmp_path / "beer" / "csv_files"
        csv_dir.mkdir(parents=True)
        (csv_dir / "beer_advocate.csv").write_text("id,val\n1,a")

        result = download_dataset("beer", tmp_path)
        assert result == csv_dir

    @patch("table_unifier.dataset.download.urllib.request.urlretrieve")
    @patch("table_unifier.dataset.download.tarfile.open")
    def test_downloads_and_extracts(self, mock_taropen, mock_urlretrieve, tmp_path):
        mock_tf = MagicMock()
        mock_taropen.return_value.__enter__ = MagicMock(return_value=mock_tf)
        mock_taropen.return_value.__exit__ = MagicMock(return_value=False)

        # Simulate that extractall creates the csv_files dir
        def fake_extract(path):
            csv_dir = tmp_path / "beer" / "csv_files"
            csv_dir.mkdir(parents=True, exist_ok=True)
            (csv_dir / "beer_advocate.csv").write_text("id,v\n1,a")

        mock_tf.extractall = fake_extract

        # urlretrieve side-effect: create the tar.gz file
        def fake_download(url, dest):
            Path(dest).write_bytes(b"dummy")

        mock_urlretrieve.side_effect = fake_download

        result = download_dataset("beer", tmp_path)
        mock_urlretrieve.assert_called_once()
        assert result == tmp_path / "beer" / "csv_files"
