"""Shared data and constants for E2E tests."""

from pathlib import Path
import pytest
from edansa.io import TIMESTAMP_INFILE_FORMAT

# Define paths relative to the project root
# Assuming tests run from the project root or discover paths correctly
TESTS_DIR = Path(
    __file__).parent.parent  # Go up two levels from test_e2e/test_data.py
ASSETS_DIR = TESTS_DIR / "assets"
EDANSA_DIR = ASSETS_DIR / "31m2plxv-V1"
MODEL_INFO_DIR = EDANSA_DIR / "model_info"
# Define the base directory for reference CSVs
EXPECTED_CSV_BASE_DIR = EDANSA_DIR / "10s_csv"

MODEL_PT = MODEL_INFO_DIR / "best_model_370_val_f1_min=0.8028.pt"
MODEL_CONFIG = MODEL_INFO_DIR / "model_config.json"
MODEL_ID = "31m2plxv-V1"

# List of tuples: (rel_audio_path, region, loc, year, filename, duration, rec_id, start_dt_str, rel_expected_csv_path)
# Shared across multiple E2E tests
AUDIO_TEST_CASES = [
    # ANWR
    ("anwr/47/2022/S4A10341_20220802_235902.flac", "anwr", "47", "2022",
     "S4A10341_20220802_235902.flac", 58, "S4A10341",
     "2022-08-02T23:59:02.000000", "anwr/47/2022/2022-08-02_23-59-02_pred.csv"),
    pytest.param("anwr/47/2022/S4A10341_20220802_225908.flac",
                 "anwr",
                 "47",
                 "2022",
                 "S4A10341_20220802_225908.flac",
                 3597,
                 "S4A10341",
                 "2022-08-02T22:59:08.000000",
                 "anwr/47/2022/2022-08-02_22-59-08_pred.csv",
                 marks=pytest.mark.slow,
                 id="anwr_long"),  # ~1 hour file
    # Dalton
    ("dalton/10/2023/S4A10407_20230906_033155.flac", "dalton", "10", "2023",
     "S4A10407_20230906_033155.flac", 180, "S4A10407",
     "2023-09-06T03:31:55.000000", "dalton/10/2023/2023-09-06_03-31-55_pred.csv"
    ),
    ("dalton/05/2023/S4A10422_20230831_120000.flac", "dalton", "05", "2023",
     "S4A10422_20230831_120000.flac", 180, "S4A10422",
     "2023-08-31T12:00:00.000000",
     "dalton/05/2023/2023-08-31_12-00-00_pred.csv"),
    ("dalton/04/2023/S4A10291_20230606_025958.flac", "dalton", "04", "2023",
     "S4A10291_20230606_025958.flac", 180, "S4A10291",
     "2023-06-06T02:59:58.000000",
     "dalton/04/2023/2023-06-06_02-59-58_pred.csv"),
    # Dempster
    ("dempster/20/2023/S4A10444_20230818_012000.wav", "dempster", "20", "2023",
     "S4A10444_20230818_012000.wav", 300, "S4A10444",
     "2023-08-18T01:20:00.000000",
     "dempster/20/2023/2023-08-18_01-20-00_pred.csv"),
]
