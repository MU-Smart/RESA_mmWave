"""Contract tests for the refactored Branch 3 BEV-label generator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from branch3.branch3_label_bev import generate_session_labels


class Branch3LabelBEVTests(unittest.TestCase):
    def test_writes_one_polar_label_per_csv_frame_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session_dir = Path(temp_dir) / "session_example"
            session_dir.mkdir()
            pd.DataFrame(
                [
                    {"x": 0.0, "y": 2.0, "z": 0.0, "snr": 8.0, "bucket_4class": "structure", "radar_frame_num": 1},
                    {"x": 0.2, "y": 3.0, "z": 0.0, "snr": 8.0, "bucket_4class": "structure", "radar_frame_num": 2},
                ]
            ).to_csv(session_dir / "labeled_radar_points_v4_fused.csv", index=False)

            self.assertEqual(generate_session_labels(session_dir), 2)
            meta = json.loads((session_dir / "bev_label_meta.json").read_text())
            self.assertEqual(meta["frame_ids"], [1, 2])
            self.assertEqual(meta["n_range_bins"], 128)
            self.assertEqual(meta["n_az_bins"], 128)
            self.assertEqual(np.load(session_dir / "bev_labels" / "000001.npy").shape, (128, 128))


if __name__ == "__main__":
    unittest.main()
