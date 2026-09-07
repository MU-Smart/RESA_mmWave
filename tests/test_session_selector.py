"""Local contract tests for the notebook session-catalog helper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tools import session_selector as selector


class SessionSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = pd.DataFrame(
            [
                {
                    "session_id": "raw-a",
                    "dataset_id": "ds-a",
                    "quality_tier": "silver",
                    "label_status": "raw",
                    "split": "none",
                    "processed_path": "",
                },
                {
                    "session_id": "train-a",
                    "dataset_id": "ds-a",
                    "quality_tier": "gold",
                    "label_status": "curated",
                    "split": "train",
                    "processed_path": "gs://bucket/curated/sessions/gold/ds-a/train-a.tar.gz",
                },
                {
                    "session_id": "holdout-b",
                    "dataset_id": "ds-b",
                    "quality_tier": "gold",
                    "label_status": "curated",
                    "split": "holdout",
                    "processed_path": "gs://bucket/curated/sessions/gold/ds-b/holdout-b.tar.gz",
                },
            ]
        )

    def test_selects_curated_rows_and_excludes_splits(self) -> None:
        selected = selector.select_sessions(
            self.catalog, split_exclude=("holdout",), dataset_ids=("ds-a",)
        )
        self.assertEqual(selected["session_id"].tolist(), ["train-a"])

    def test_can_select_raw_intake_rows_without_a_processed_path(self) -> None:
        selected = selector.select_sessions(
            self.catalog,
            quality_tier=("silver",),
            label_status=("raw",),
            require_processed_path=False,
        )
        self.assertEqual(selected["session_id"].tolist(), ["raw-a"])

    def test_admission_updates_an_existing_row_and_preserves_metadata(self) -> None:
        admitted = selector.admit_session(
            self.catalog,
            "raw-a",
            "gold",
            "curated",
            "gs://bucket/curated/sessions/gold/ds-a/raw-a.tar.gz",
            notes="passed admission",
        )
        row = admitted.loc[admitted["session_id"] == "raw-a"].iloc[0]
        self.assertEqual(row["dataset_id"], "ds-a")
        self.assertEqual(row["quality_tier"], "gold")
        self.assertEqual(row["label_status"], "curated")
        self.assertEqual(row["notes"], "passed admission")

    def test_catalog_sync_uses_gcloud_storage_cp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "nested" / "catalog.csv"
            with patch("tools.session_selector.subprocess.run") as run:
                selector.sync_catalog_from_gcs(local_path, "gs://bucket/catalog.csv")
            run.assert_called_once_with(
                ["gcloud", "storage", "cp", "gs://bucket/catalog.csv", str(local_path)],
                check=True,
                text=True,
            )
            self.assertTrue(local_path.parent.exists())

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "catalog.csv"
            local_path.write_text("session_id\\nexample\\n")
            with patch("tools.session_selector.subprocess.run") as run:
                selector.sync_catalog_to_gcs(local_path, "gs://bucket/catalog.csv")
            run.assert_called_once_with(
                ["gcloud", "storage", "cp", str(local_path), "gs://bucket/catalog.csv"],
                check=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
