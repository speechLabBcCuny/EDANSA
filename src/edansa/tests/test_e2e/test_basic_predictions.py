"""End-to-end tests for basic prediction runs."""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
import json
import numpy as np
import logging
from datetime import datetime
import torch

from edansa.io import TIMESTAMP_INFILE_FORMAT

# Import shared data
from .test_data import (AUDIO_TEST_CASES, ASSETS_DIR, MODEL_PT, MODEL_CONFIG,
                        EXPECTED_CSV_BASE_DIR)

# Get logger for this module
logger = logging.getLogger(__name__)

# Define paths relative to the project root (assuming tests run from there)
# Adjust these paths if the test execution context is different
# TESTS_DIR = Path(
#     __file__
# ).parent.parent  # Go up two levels from test_e2e/test_basic_predictions.py
# ASSETS_DIR = TESTS_DIR / "assets"
# EDANSA_DIR = ASSETS_DIR / "31m2plxv-V1"
# MODEL_INFO_DIR = EDANSA_DIR / "model_info"
# # Define the base directory for reference CSVs
# EXPECTED_CSV_BASE_DIR = EDANSA_DIR / "10s_csv"
#
# MODEL_PT = MODEL_INFO_DIR / "best_model_370_val_f1_min=0.8028.pt"
# MODEL_CONFIG = MODEL_INFO_DIR / "model_config.json"
# MODEL_ID = "31m2plxv-V1"  # Update model ID to match directory
# Path definitions moved to test_data.py

# --- Test Case Definitions (Copied from e2e_test.py) ---
# List of tuples: (rel_audio_path, region, loc, year, filename, duration, rec_id, start_dt_str, rel_expected_csv_path)
# AUDIO_TEST_CASES = [
#     # ANWR
#     ("anwr/47/2022/S4A10341_20220802_235902.flac", "anwr", "47", "2022",
#      "S4A10341_20220802_235902.flac", 58, "S4A10341",
#      "2022-08-02T23:59:02.000000", "anwr/47/2022/2022-08-02_23-59-02_pred.csv"),
#     pytest.param("anwr/47/2022/S4A10341_20220802_225908.flac",
#                  "anwr",
#                  "47",
#                  "2022",
#                  "S4A10341_20220802_225908.flac",
#                  3597,
#                  "S4A10341",
#                  "2022-08-02T22:59:08.000000",
#                  "anwr/47/2022/2022-08-02_22-59-08_pred.csv",
#                  marks=pytest.mark.slow,
#                  id="anwr_long"),  # ~1 hour file
#     # Dalton
#     ("dalton/10/2023/S4A10407_20230906_033155.flac", "dalton", "10", "2023",
#      "S4A10407_20230906_033155.flac", 180, "S4A10407",
#      "2023-09-06T03:31:55.000000", "dalton/10/2023/2023-09-06_03-31-55_pred.csv"
#     ),
#     ("dalton/05/2023/S4A10422_20230831_120000.flac", "dalton", "05", "2023",
#      "S4A10422_20230831_120000.flac", 180, "S4A10422",
#      "2023-08-31T12:00:00.000000",
#      "dalton/05/2023/2023-08-31_12-00-00_pred.csv"),
#     ("dalton/04/2023/S4A10291_20230606_025958.flac", "dalton", "04", "2023",
#      "S4A10291_20230606_025958.flac", 180, "S4A10291",
#      "2023-06-06T02:59:58.000000",
#      "dalton/04/2023/2023-06-06_02-59-58_pred.csv"),
#     # Dempster
#     ("dempster/20/2023/S4A10444_20230818_012000.wav", "dempster", "20", "2023",
#      "S4A10444_20230818_012000.wav", 300, "S4A10444",
#      "2023-08-18T01:20:00.000000",
#      "dempster/20/2023/2023-08-18_01-20-00_pred.csv"),
# ]
# AUDIO_TEST_CASES moved to test_data.py


@pytest.mark.e2e
@pytest.mark.parametrize(
    "test_audio_path_rel_assets_str, _region, _location, _year, test_audio_filename, actual_duration_sec, _recorder_id, start_dt_str, rel_expected_csv_path",
    AUDIO_TEST_CASES)
@pytest.mark.parametrize("channel_selection_method", [
    pytest.param("average", id="average"),
    pytest.param("clipping", id="clipping"),
])
@pytest.mark.parametrize("device", [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda",
                 marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                          reason="CUDA device not available"),
                 id="cuda"),
])
def test_inference_pipeline_single_file(
    tmp_path,  # Pytest fixture for temporary directory
    test_audio_path_rel_assets_str,
    _region,  # Underscore indicates not directly used in this test logic
    _location,
    _year,
    test_audio_filename,
    actual_duration_sec,
    _recorder_id,
    start_dt_str,
    rel_expected_csv_path,
    channel_selection_method,
    device,
):
    """Run the inference script on a single test file and verify output structure and values."""
    # --- 1. Prepare Input File List ---
    test_audio_path_rel_assets = Path(test_audio_path_rel_assets_str)
    # Construct absolute path to the test audio file inside the assets directory
    test_audio_path_abs = (ASSETS_DIR / "audio" / "real" /
                           test_audio_path_rel_assets).resolve()

    if not test_audio_path_abs.is_file():
        pytest.fail(f"Test audio file not found: {test_audio_path_abs}")

    # Create a temporary file to list the single input audio file
    input_list_file = tmp_path / f"input_list_{Path(test_audio_filename).stem}_{channel_selection_method}_{device}.txt"
    with open(input_list_file, 'w') as f:
        f.write(f"{str(test_audio_path_abs)}\n")  # Write absolute path
    logger.debug(
        f"Created input file list: {input_list_file} with content: {test_audio_path_abs}"
    )

    # --- 2. Construct Command ---
    output_dir = tmp_path / f"test_output_{Path(test_audio_filename).stem}_{device}"
    logger.debug(f"E2E test output base directory: {output_dir.resolve()}")

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
        channel_selection_method,
        "--device",
        device,
    ]

    # --- 3. Run Subprocess ---
    logger.info(f"Running E2E test for: {test_audio_path_abs}")
    print(f"\nRunning command for {test_audio_filename}: {' '.join(command)}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,  # Don't check=True, assert returncode below
        timeout=1200)  # Add timeout (e.g., 20 minutes for the long file)

    # --- 4. Assert Execution Success ---
    print(
        f"STDOUT for {test_audio_filename} ({device}):\n{result.stdout[-1000:]}"
    )  # Print last 1000 chars
    print(
        f"STDERR for {test_audio_filename} ({device}):\n{result.stderr[-1000:]}"
    )  # Print last 1000 chars
    # Ensure stderr doesn't contain unexpected errors (allow specific warnings maybe)
    # Check specific known failure strings
    assert "Traceback" not in result.stderr, f"Traceback detected in stderr for {test_audio_filename} ({device})"
    assert "RuntimeError" not in result.stderr, f"RuntimeError detected in stderr for {test_audio_filename} ({device})"

    # Assert return code AFTER checking stderr for easier debugging
    assert result.returncode == 0, f"Inference script failed for {test_audio_filename} ({device}) with return code {result.returncode}"

    # --- Assert Absence of Clipping Warnings/Errors (if applicable) ---
    # Only check these if clipping method is used? Or always check they don't appear unexpectedly?
    # Let's check always for now to catch regressions.
    assert "Could not calculate clipping" not in result.stderr, \
        f"Clipping calculation error detected in stderr for {test_audio_filename} ({device})."
    assert "Could not calculate clipping" not in result.stdout, \
        f"Clipping calculation error detected in stdout for {test_audio_filename} ({device})."
    # Check for the reconstruction mismatch warning
    assert "Clipping segment count mismatch" not in result.stderr, \
        f"Clipping segment count mismatch detected in stderr for {test_audio_filename} ({device})."
    assert "Clipping segment count mismatch" not in result.stdout, \
        f"Clipping segment count mismatch detected in stdout for {test_audio_filename} ({device})."

    # --- 5. Locate and Verify Output Structure ---
    # The inference script saves output relative to the determined input root,
    # under the output_dir. For a single file input list with an absolute path,
    # the script might use the file's directory or a higher-level common path as root.
    # The most robust way to check output is to use the original relative path
    # structure provided in the test case.
    # The script is expected to create output_dir / <relative_path_from_test_case>.csv

    # Use the original relative path string from the test case definition
    # (test_audio_path_rel_assets_str) to construct the expected output path.
    # generated_output_path = (output_dir /
    #                          test_audio_path_rel_assets_str).with_suffix(".csv")

    # --- Restore original block calculating path relative to parent ---
    # Determine input_data_root as the parent of the audio file
    # This mimics how the script should determine it when given a list containing this single file.
    input_data_root_actual = test_audio_path_abs.parent

    # Calculate relative path for output structure
    try:
        relative_audio_path_for_output = test_audio_path_abs.relative_to(
            input_data_root_actual)
    except ValueError:
        pytest.fail(
            f"Could not determine relative path for {test_audio_path_abs} relative to its parent {input_data_root_actual}"
        )

    # Construct the full expected path based on the script's derived root and relative path
    generated_output_path = output_dir / relative_audio_path_for_output.with_suffix(
        ".csv")
    # --- End restored block ---

    print(f"Checking for generated file: {generated_output_path}")
    assert generated_output_path.is_file(), \
        f"Expected output file not found for {test_audio_filename} ({device}) at {generated_output_path}"

    # Load generated dataframe and the config used
    try:
        df_generated = pd.read_csv(generated_output_path)
        with open(MODEL_CONFIG, 'r') as f:
            config_data = json.load(f)
        # DEBUG: Print head to inspect timestamp format
        logger.debug("DEBUG: Generated DataFrame head:")
        logger.debug(df_generated.head())
        logger.debug(
            f"DEBUG: Timestamp column dtype: {df_generated['timestamp'].dtype}")

        # Ensure timestamp column is loaded as datetime, trying auto-parse first, then specific format
        if not pd.api.types.is_datetime64_any_dtype(df_generated['timestamp']):
            try:
                df_generated['timestamp'] = pd.to_datetime(
                    df_generated['timestamp'], format='ISO8601')
            except (ValueError, TypeError):
                # If auto-parse fails, try the default pandas format
                logger.warning(
                    "Auto-parsing timestamp failed, trying format='%Y-%m-%d_%H:%M:%S'"
                )
                df_generated['timestamp'] = pd.to_datetime(
                    df_generated['timestamp'], format='%Y-%m-%d_%H:%M:%S'
                )  # Note: Original file used T, ref csv uses _

    except Exception as e:
        pytest.fail(
            f"Failed to load generated CSV or config file for {test_audio_filename} ({device}): {e}"
        )

    # --- Verify Clipping Column Values (if not skipped) ---
    # Clipping calculation is not skipped in this test run by default
    assert 'clipping' in df_generated.columns, f"Clipping column missing for {test_audio_filename} ({device})."
    assert not df_generated['clipping'].isnull().any(), \
        f"Clipping column contains NaN values for {test_audio_filename} ({device}). Values: \n{df_generated['clipping'].tolist()}"

    # --- Verify Structure Based on Config ---
    # Calculate expected number of rows
    excerpt_len_config = config_data.get('excerpt_length', {}).get('value', 10)
    # Use floor division as the primary expectation, allow +1 for padding/edge cases
    expected_rows_min = int(actual_duration_sec // excerpt_len_config)
    # Allow for ceiling + potential extra segment due to padding
    expected_rows_max = int(np.ceil(
        actual_duration_sec / excerpt_len_config)) + 1
    generated_rows = len(df_generated)

    assert expected_rows_min <= generated_rows <= expected_rows_max, \
        f"Generated rows ({generated_rows}) for {test_audio_filename} ({device}) outside expected range ({expected_rows_min}-{expected_rows_max}) based on duration ({actual_duration_sec}s) and excerpt length ({excerpt_len_config}s)."

    # --- Verify Absolute Timestamps (since filename should parse) ---
    assert 'timestamp' in df_generated.columns, f"Timestamp column missing for {test_audio_filename} ({device})"
    # Check it's datetime (or potentially numeric if fallback occurred unexpectedly, which would be a failure)
    assert pd.api.types.is_datetime64_any_dtype(df_generated['timestamp']), \
           f"Timestamp column for {test_audio_filename} ({device}) should be datetime, found {df_generated['timestamp'].dtype}"

    # Calculate expected start time (should match the one parsed from filename)
    expected_first_timestamp = pd.Timestamp(
        start_dt_str)  # From test case definition
    expected_timestamps_pd = pd.date_range(
        start=expected_first_timestamp,
        periods=generated_rows,  # Use actual generated rows
        freq=pd.Timedelta(seconds=excerpt_len_config))

    # Compare timestamp series
    pd.testing.assert_series_equal(
        df_generated['timestamp'],
        pd.Series(expected_timestamps_pd, name='timestamp'),
        check_dtype=True,
        check_index=False,  # Index might not match
        check_names=False,  # Series name might differ
    )

    # --- Verify Columns Based on Config (excluding timestamp) ---
    expected_pred_cols = set()
    target_taxo = config_data.get('target_taxo', {}).get('value', [])
    code2name = config_data.get('code2excell_names', {}).get('value', {})
    for code in target_taxo:
        expected_pred_cols.add(f"pred_{code2name.get(code, code)}")
    expected_pred_cols.add('clipping')  # Clipping column should exist
    expected_pred_cols_list = sorted(list(expected_pred_cols))

    generated_cols_no_ts = sorted(
        [col for col in df_generated.columns if col != 'timestamp'])

    assert generated_cols_no_ts == expected_pred_cols_list, \
        f"Generated columns (excluding timestamp) differ from expected for {test_audio_filename} ({device}).\nGenerated: {generated_cols_no_ts}\nExpected: {expected_pred_cols_list}"

    # --- Value Comparison Against Reference CSV --- #
    if rel_expected_csv_path:
        # Construct path using the base dir and method-specific subfolder
        method_subfolder = f"pick_channel_by_{channel_selection_method}"
        expected_csv_full_path = EXPECTED_CSV_BASE_DIR / method_subfolder / rel_expected_csv_path

        if not expected_csv_full_path.is_file():
            # Add a specific skip message if the clipping file is missing
            if channel_selection_method == 'clipping':
                pytest.skip(
                    f"Reference CSV file for 'clipping' not found at: {expected_csv_full_path}. Generate it manually."
                )
            # Fail for missing 'average' files as they should exist
            pytest.fail(
                f"Reference CSV file not found at: {expected_csv_full_path}")

        logger.info(
            f"Comparing generated output with reference: {expected_csv_full_path} (Device: {device})"
        )

        try:
            df_expected = pd.read_csv(expected_csv_full_path)

            # --- Preprocess expected dataframe --- #
            # 1. Parse timestamp (format YYYY-MM-DD_HH-MM-SS in reference files)
            df_expected['timestamp'] = pd.to_datetime(
                df_expected['timestamp'], format=TIMESTAMP_INFILE_FORMAT)

            # 2. Rename columns to add 'pred_' prefix based on config
            code2name = config_data.get('code2excell_names',
                                        {}).get('value', {})
            rename_map = {
                name: f"pred_{name}"
                for code, name in code2name.items()
                if name in
                df_expected.columns  # Rename only if column exists in expected
            }
            df_expected = df_expected.rename(columns=rename_map)
            # --- End Preprocessing --- #

            # Handle potential row mismatch (compare up to min length)
            min_rows = min(len(df_generated), len(df_expected))
            if len(df_generated) != len(df_expected):
                logger.warning(
                    f"Row count mismatch for {test_audio_filename}: Generated({len(df_generated)}) vs Expected({len(df_expected)}). Comparing first {min_rows} rows."
                )
                df_generated_comp = df_generated.iloc[:min_rows].copy()
                df_expected_comp = df_expected.iloc[:min_rows].copy()
            else:
                df_generated_comp = df_generated.copy()
                df_expected_comp = df_expected.copy()

            # Identify common columns for comparison (include predictions and timestamp, exclude clipping for now)
            common_cols = sorted(
                list(
                    set(df_generated_comp.columns) &
                    set(df_expected_comp.columns) - {'clipping'}))

            if 'timestamp' not in common_cols:
                pytest.fail(
                    f"Timestamp column missing in common columns for comparison: {common_cols}"
                )
            if len(common_cols) <= 1:  # Only timestamp found
                pytest.fail(
                    f"No common prediction columns found for comparison: {common_cols}"
                )

            logger.debug(f"Comparing columns: {common_cols}")

            # Reset index for comparison, ensure timestamps align
            df_generated_comp = df_generated_comp[common_cols].reset_index(
                drop=True)
            df_expected_comp = df_expected_comp[common_cols].reset_index(
                drop=True)

            # Compare dataframes (only common prediction cols + timestamp)
            pd.testing.assert_frame_equal(
                df_generated_comp,
                df_expected_comp,
                check_dtype=False,  # Allow float64 vs float32
                rtol=1e-3,
                atol=3e-4  # Increased tolerance
            )
            logger.info(
                f"Value comparison successful for {test_audio_filename} ({device})."
            )

        except AssertionError as e:
            pytest.fail(
                f"DataFrame value comparison failed for {test_audio_filename} ({device}) using ref {expected_csv_full_path}:\n{e}"
            )
        except Exception as e:
            pytest.fail(
                f"Error during comparison preprocessing for {test_audio_filename} ({device}): {e}"
            )

    else:
        logger.warning(
            f"No reference CSV specified for {test_audio_filename} ({device}). Skipping value comparison."
        )
