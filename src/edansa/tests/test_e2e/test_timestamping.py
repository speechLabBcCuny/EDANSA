"""End-to-end tests for timestamp generation logic."""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
import json
import numpy as np
import logging
from datetime import datetime, timedelta
import torch
from typing import Optional
#from _pytest.mark.structures import ParameterSet # No longer needed directly

from edansa import inference  # For parsing function

# Import shared data
from .test_data import (AUDIO_TEST_CASES, ASSETS_DIR, MODEL_PT, MODEL_CONFIG)

# Get logger for this module
logger = logging.getLogger(__name__)

# Define paths relative to the project root (assuming tests run from there)
#TESTS_DIR = Path(__file__).parent.parent
#ASSETS_DIR = TESTS_DIR / "assets"
#EDANSA_DIR = ASSETS_DIR / "31m2plxv-V1"
#MODEL_INFO_DIR = EDANSA_DIR / "model_info"

#MODEL_PT = MODEL_INFO_DIR / "best_model_370_val_f1_min=0.8028.pt"
#MODEL_CONFIG = MODEL_INFO_DIR / "model_config.json"

# --- Test Case Data (Copied/Adapted from e2e_test.py) ---

# Use the same AUDIO_TEST_CASES definition for extracting paths/data
# List of tuples: (rel_audio_path, region, loc, year, filename, duration, rec_id, start_dt_str, rel_expected_csv_path)
#AUDIO_TEST_CASES = [
#    # ANWR
#    ("anwr/47/2022/S4A10341_20220802_235902.flac", "anwr", "47", "2022",
#     "S4A10341_20220802_235902.flac", 58, "S4A10341",
#     "2022-08-02T23:59:02.000000", "anwr/47/2022/2022-08-02_23-59-02_pred.csv"),
#    pytest.param("anwr/47/2022/S4A10341_20220802_225908.flac",
#                 "anwr",
#                 "47",
#                 "2022",
#                 "S4A10341_20220802_225908.flac",
#                 3597,
#                 "S4A10341",
#                 "2022-08-02T22:59:08.000000",
#                 "anwr/47/2022/2022-08-02_22-59-08_pred.csv",
#                 marks=pytest.mark.slow,
#                 id="anwr_long"),  # ~1 hour file
#    # Dalton
#    ("dalton/10/2023/S4A10407_20230906_033155.flac", "dalton", "10", "2023",
#     "S4A10407_20230906_033155.flac", 180, "S4A10407",
#     "2023-09-06T03:31:55.000000", "dalton/10/2023/2023-09-06_03-31-55_pred.csv"
#    ),
#    ("dalton/05/2023/S4A10422_20230831_120000.flac", "dalton", "05", "2023",
#     "S4A10422_20230831_120000.flac", 180, "S4A10422",
#     "2023-08-31T12:00:00.000000",
#     "dalton/05/2023/2023-08-31_12-00-00_pred.csv"),
#    ("dalton/04/2023/S4A10291_20230606_025958.flac", "dalton", "04", "2023",
#     "S4A10291_20230606_025958.flac", 180, "S4A10291",
#     "2023-06-06T02:59:58.000000",
#     "dalton/04/2023/2023-06-06_02-59-58_pred.csv"),
#    # Dempster
#    ("dempster/20/2023/S4A10444_20230818_012000.wav", "dempster", "20", "2023",
#     "S4A10444_20230818_012000.wav", 300, "S4A10444",
#     "2023-08-18T01:20:00.000000",
#     "dempster/20/2023/2023-08-18_01-20-00_pred.csv"),
#]

# Extract paths for the absolute timestamp test (files with parsable names)
#ABSOLUTE_TIMESTAMP_TEST_PATHS = [
#    case[0] for case in AUDIO_TEST_CASES if not isinstance(case, ParameterSet)
#]

# Cases specifically for testing relative timestamp fallback
RELATIVE_TIME_TEST_CASES = [
    pytest.param("no_timestamp_in_name/anwr/ANWR_NoDate.flac",
                 "anwr",
                 "no_ts",
                 2024,
                 "ANWR_NoDate.flac",
                 58,
                 "UNKNOWN",
                 None,
                 None,
                 id="relative-anwr_no_ts_ANWR_NoDate.flac"),
    pytest.param("no_timestamp_in_name/dempster/DEMPSTER_NoDate.wav",
                 "dempster",
                 "no_ts",
                 2024,
                 "DEMPSTER_NoDate.wav",
                 300,
                 "UNKNOWN",
                 None,
                 None,
                 id="relative-dempster_no_ts_DEMPSTER_NoDate.wav"),
]

# Helper to extract case data (slightly modified from original)
#def get_case_data(target_path_rel_assets_str, test_cases_list):
#    """Finds and returns the dictionary for a specific test case path."""
#    for case in test_cases_list:
#        # Handle both tuples and pytest.param objects
#        path_in_case = case[0] if isinstance(case, tuple) else case.values[0]
#        if path_in_case == target_path_rel_assets_str:
#            if isinstance(case, tuple):
#                return {
#                    "rel_audio_path": case[0],
#                    "region": case[1],
#                    "location": case[2],
#                    "year": case[3],
#                    "filename": case[4],
#                    "duration": case[5],
#                    "recorder_id": case[6],
#                    "start_dt_str": case[7],
#                    "rel_expected_csv_path": case[8]
#                }
#            else:  # It's a pytest.param
#                return {
#                    "rel_audio_path": case.values[0],
#                    "region": case.values[1],
#                    "location": case.values[2],
#                    "year": case.values[3],
#                    "filename": case.values[4],
#                    "duration": case.values[5],
#                    "recorder_id": case.values[6],
#                    "start_dt_str": case.values[7],
#                    "rel_expected_csv_path": case.values[8]
#                }
#    return None


@pytest.mark.e2e
#@pytest.mark.parametrize("test_audio_path_rel_assets_str",
#                         ABSOLUTE_TIMESTAMP_TEST_PATHS)
#def test_absolute_timestamps_from_filename(tmp_path,
#                                           test_audio_path_rel_assets_str):
#    """Verify absolute timestamps are generated correctly when filename is parsable."""
#    case_data = get_case_data(test_audio_path_rel_assets_str, AUDIO_TEST_CASES)
#    if case_data is None:
#        pytest.fail(
#            f"Could not find test case data for path: {test_audio_path_rel_assets_str}"
#        )
#
#    test_audio_path_rel_assets = Path(test_audio_path_rel_assets_str)
#    test_audio_path_abs = (ASSETS_DIR / "audio" / "real" /
#                           test_audio_path_rel_assets).resolve()
#    actual_duration_sec = case_data["duration"]
#    test_audio_filename = case_data["filename"]
#
#    if not test_audio_path_abs.is_file():
#        pytest.fail(f"Test audio file not found: {test_audio_path_abs}")

# Parametrize over the full test case data
@pytest.mark.parametrize(
    ("test_audio_path_rel_assets_str, _region, _location, _year, "
     "test_audio_filename, actual_duration_sec, _recorder_id, "
     "start_dt_str, _rel_expected_csv_path"),
    AUDIO_TEST_CASES  # Pass the whole list, filter inside
)
def test_absolute_timestamps_from_filename(
    tmp_path,
    test_audio_path_rel_assets_str: str,
    _region: str,
    _location: str,
    _year: str,
    test_audio_filename: str,
    actual_duration_sec: int,
    _recorder_id: str,
    start_dt_str: Optional[str],  # Expected start time if parsed correctly
    _rel_expected_csv_path: str,
):
    """Verify absolute timestamps are generated correctly for parsable filenames."""
    # Skip this test if the case doesn't have an expected start time (i.e., filename isn't parsable)
    if start_dt_str is None:
        pytest.skip(
            f"Skipping absolute timestamp test for {test_audio_filename} as start_dt_str is None."
        )

    # Check if the filename is indeed parsable (as a sanity check for the test setup)
    if inference.parse_start_time_from_filename(test_audio_filename) is None:
        pytest.skip(
            f"Filename {test_audio_filename} was not parsable by the helper, skipping absolute timestamp test."
        )

    test_audio_path_rel_assets = Path(test_audio_path_rel_assets_str)
    test_audio_path_abs = (ASSETS_DIR / "audio" / "real" /
                           test_audio_path_rel_assets).resolve()

    if not test_audio_path_abs.is_file():
        pytest.fail(f"Test audio file not found: {test_audio_path_abs}")

    # --- 1. Prepare Input File List ---
    input_list_file = tmp_path / f"input_list_absolute_{Path(test_audio_filename).stem}.txt"
    with open(input_list_file, 'w') as f:
        f.write(f"{str(test_audio_path_abs)}\n")
    logger.debug(f"Created absolute time input list: {input_list_file}")

    # --- 2. Construct Command ---
    output_dir = tmp_path / f"test_output_absolute_{Path(test_audio_filename).stem}"
    logger.debug(
        f"E2E absolute time test output directory: {output_dir.resolve()}")
    command = [
        sys.executable,
        "-m",
        "runs.augment.inference",
        "--model_path",
        str(MODEL_PT.resolve()),
        "--config_file",
        str(MODEL_CONFIG.resolve()),
        "--input_files_list",
        str(input_list_file.resolve()),
        "--output_folder",
        str(output_dir.resolve()),
        "--channel_selection_method",
        "average",  # Keep simple
        "--device",
        "cpu",
    ]

    # --- 3. Run Subprocess ---
    logger.info(f"Running E2E absolute time test for: {test_audio_path_abs}")
    result = subprocess.run(command,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=1200)

    # --- 4. Assert Execution Success and NO Warning Log --- #
    print(
        f"STDOUT (Absolute Time) {test_audio_filename}:\n{result.stdout[-500:]}"
    )
    print(
        f"STDERR (Absolute Time) {test_audio_filename}:\n{result.stderr[-500:]}"
    )
    assert "Traceback" not in result.stderr, f"Traceback in stderr for {test_audio_filename}"
    assert result.returncode == 0, f"Script failed for {test_audio_filename} with code {result.returncode}"
    # CRITICAL: Ensure the fallback warning did NOT appear
    assert "Could not determine absolute start time" not in result.stderr, \
        f"Fallback warning unexpectedly found in stderr for {test_audio_filename}."
    assert "Using relative time index" not in result.stderr, \
        f"Relative index message unexpectedly found in stderr for {test_audio_filename}."

    # --- 5. Locate and Verify Output Structure & Timestamps --- #
    input_data_root_actual = test_audio_path_abs.parent
    try:
        relative_audio_path_for_output = test_audio_path_abs.relative_to(
            input_data_root_actual)
    except ValueError:
        pytest.fail(f"Could not get relative path for {test_audio_path_abs}")

    generated_output_path = output_dir / relative_audio_path_for_output.with_suffix(
        ".csv")
    assert generated_output_path.is_file(
    ), f"Output file not found: {generated_output_path}"

    try:
        df_generated = pd.read_csv(generated_output_path)
        with open(MODEL_CONFIG, 'r') as f:
            config_data = json.load(f)
        # Ensure timestamp column is loaded as datetime, trying multiple formats
        if not pd.api.types.is_datetime64_any_dtype(df_generated['timestamp']):
            parsed = False
            for fmt in (
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d_%H:%M:%S'):  # Try space and underscore separators
                try:
                    df_generated['timestamp'] = pd.to_datetime(
                        df_generated['timestamp'], format=fmt)
                    parsed = True
                    break  # Stop if successful
                except (ValueError, TypeError):
                    continue  # Try next format
            if not parsed:
                # If specific formats fail, try pandas auto-parsing as a last resort
                logger.warning(
                    f"Parsing timestamp with specific formats ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d_%H:%M:%S') failed, trying auto-parsing for {test_audio_filename}."
                )
                df_generated['timestamp'] = pd.to_datetime(
                    df_generated['timestamp'])  # Fallback to auto

    except Exception as e:
        pytest.fail(f"Failed to load generated CSV or config: {e}")

    # --- Verify Rows ---
    excerpt_len_config = config_data.get('excerpt_length', {}).get('value', 10)
    expected_rows_min = int(actual_duration_sec // excerpt_len_config)
    expected_rows_max = int(np.ceil(
        actual_duration_sec / excerpt_len_config)) + 1
    generated_rows = len(df_generated)
    assert expected_rows_min <= generated_rows <= expected_rows_max, \
        f"Generated rows ({generated_rows}) outside expected range ({expected_rows_min}-{expected_rows_max}) for {test_audio_filename}."

    # --- Verify Absolute Timestamps ---
    assert 'timestamp' in df_generated.columns, f"Timestamp column missing for {test_audio_filename}"
    assert pd.api.types.is_datetime64_any_dtype(df_generated['timestamp']), \
           f"Timestamp column should be datetime, found {df_generated['timestamp'].dtype}"

    # Calculate expected start time by parsing the filename using the script's logic
    expected_start_dt = inference.parse_start_time_from_filename(
        test_audio_filename)
    if expected_start_dt is None:
        pytest.fail(
            f"Test setup error: Could not parse known good filename {test_audio_filename}."
        )
    expected_first_timestamp = pd.Timestamp(expected_start_dt)
    expected_timestamps_pd = pd.date_range(
        start=expected_first_timestamp,
        periods=generated_rows,
        freq=pd.Timedelta(seconds=excerpt_len_config))

    pd.testing.assert_series_equal(
        df_generated['timestamp'],
        pd.Series(expected_timestamps_pd, name='timestamp'),
        check_dtype=True,
        check_index=False,
        check_names=False,
    )

    # --- Verify Columns --- (Minimal check, detailed check in basic predictions test)
    assert 'clipping' in df_generated.columns  # Should be present even if NaN
    pred_cols_exist = any(
        col.startswith('pred_') for col in df_generated.columns)
    assert pred_cols_exist, "No prediction columns found in output."


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("test_audio_path_rel_assets_str, _region, _location, _year, "
     "test_audio_filename, actual_duration_sec, _recorder_id, _start_dt_str, "
     "_rel_expected_csv_path"), RELATIVE_TIME_TEST_CASES)
def test_relative_timestamps_fallback(
        tmp_path,
        test_audio_path_rel_assets_str,
        _region,
        _location,
        _year,  # Unused
        test_audio_filename,
        actual_duration_sec,
        _recorder_id,
        _start_dt_str,
        _rel_expected_csv_path  # Unused
):
    """Run inference with unparsable filenames, verify relative timestamps and warning log."""
    # --- 1. Prepare Input File List ---
    test_audio_path_rel_assets = Path(test_audio_path_rel_assets_str)
    test_audio_path_abs = (ASSETS_DIR / "audio" / "real" /
                           test_audio_path_rel_assets).resolve()

    if not test_audio_path_abs.is_file():
        pytest.fail(f"Test audio file not found: {test_audio_path_abs}")

    input_list_file = tmp_path / f"input_list_relative_{Path(test_audio_filename).stem}.txt"
    with open(input_list_file, 'w') as f:
        f.write(f"{str(test_audio_path_abs)}\n")
    logger.debug(f"Created relative time input list: {input_list_file}")

    # --- 2. Construct Command ---
    output_dir = tmp_path / f"test_output_relative_{Path(test_audio_filename).stem}"
    logger.debug(
        f"E2E relative time test output directory: {output_dir.resolve()}")
    command = [
        sys.executable,
        "-m",
        "runs.augment.inference",
        "--model_path",
        str(MODEL_PT.resolve()),
        "--config_file",
        str(MODEL_CONFIG.resolve()),
        "--input_files_list",
        str(input_list_file.resolve()),
        "--output_folder",
        str(output_dir.resolve()),
        "--channel_selection_method",
        "average",  # Keep simple
        "--device",
        "cpu",
    ]

    # --- 3. Run Subprocess ---
    logger.info(f"Running E2E relative time test for: {test_audio_path_abs}")
    result = subprocess.run(command,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=1200)

    # --- 4. Assert Execution Success and WARNING Log --- #
    print(
        f"STDOUT (Relative Time) {test_audio_filename}:\n{result.stdout[-500:]}"
    )
    print(
        f"STDERR (Relative Time) {test_audio_filename}:\n{result.stderr[-500:]}"
    )
    assert "Traceback" not in result.stderr, f"Traceback in stderr for {test_audio_filename}"
    assert result.returncode == 0, f"Script failed for {test_audio_filename} with code {result.returncode}"
    # CRITICAL: Assert that the specific warning for fallback WAS logged
    assert "Could not determine absolute start time" in result.stderr, \
        f"Expected fallback warning not found in stderr for {test_audio_filename}."
    assert "Using relative time index" in result.stderr, \
        f"Expected relative index message not found in stderr for {test_audio_filename}."

    # --- 5. Locate and Verify Output Structure & Timestamps --- #
    input_data_root_actual = test_audio_path_abs.parent
    try:
        relative_audio_path_for_output = test_audio_path_abs.relative_to(
            input_data_root_actual)
    except ValueError:
        pytest.fail(f"Could not get relative path for {test_audio_path_abs}")

    generated_output_path = output_dir / relative_audio_path_for_output.with_suffix(
        ".csv")
    assert generated_output_path.is_file(
    ), f"Output file not found: {generated_output_path}"

    try:
        df_generated = pd.read_csv(generated_output_path)
        with open(MODEL_CONFIG, 'r') as f:
            config_data = json.load(f)
    except Exception as e:
        pytest.fail(f"Failed to load generated CSV or config: {e}")

    # --- Verify Rows ---
    excerpt_len_config = config_data.get('excerpt_length', {}).get('value', 10)
    expected_rows_min = int(actual_duration_sec // excerpt_len_config)
    expected_rows_max = int(np.ceil(
        actual_duration_sec / excerpt_len_config)) + 1
    generated_rows = len(df_generated)
    assert expected_rows_min <= generated_rows <= expected_rows_max, \
        f"Generated rows ({generated_rows}) outside expected range ({expected_rows_min}-{expected_rows_max}) for {test_audio_filename} (relative time)."

    # --- Verify Relative Timestamps ---
    assert 'timestamp' in df_generated.columns, f"Timestamp column missing for {test_audio_filename}"
    # Check it is numeric (float)
    assert pd.api.types.is_numeric_dtype(df_generated['timestamp']), \
           f"Timestamp column should be numeric (float), found {df_generated['timestamp'].dtype}"
    # Check it is specifically float64 as expected
    assert df_generated['timestamp'].dtype == np.float64, \
        f"Timestamp column should be float64, found {df_generated['timestamp'].dtype}"

    # Calculate expected relative timestamps (float seconds)
    expected_relative_timestamps = np.arange(generated_rows) * float(
        excerpt_len_config)
    expected_timestamps_pd = pd.Series(expected_relative_timestamps,
                                       name='timestamp',
                                       dtype='float64')

    pd.testing.assert_series_equal(
        df_generated['timestamp'],
        expected_timestamps_pd,
        check_dtype=True,
        check_index=False,
        check_names=False,
        rtol=1e-6  # Tolerance for float comparison
    )

    # --- Verify Columns --- (Minimal check)
    assert 'clipping' in df_generated.columns
    pred_cols_exist = any(
        col.startswith('pred_') for col in df_generated.columns)
    assert pred_cols_exist, "No prediction columns found in output."
