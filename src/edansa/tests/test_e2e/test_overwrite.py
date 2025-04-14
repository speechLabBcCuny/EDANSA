"""End-to-end tests for file overwrite/skipping logic."""

import subprocess
import sys
from pathlib import Path
import pytest
import os
import logging
import time

# Import shared data from test_data.py
from .test_data import (ASSETS_DIR, MODEL_PT, MODEL_CONFIG, AUDIO_TEST_CASES)

# Get logger for this module
logger = logging.getLogger(__name__)


@pytest.mark.e2e
def test_inference_overwrite_and_skip(tmp_path):
    """Test the default skipping behavior and the --force_overwrite flag."""
    # --- Use a small, fixed test case --- #
    test_case = None
    for case in AUDIO_TEST_CASES:
        # Find the first non-parametrized test case (tuple)
        if isinstance(case, tuple):
            test_case = case
            break
    if test_case is None:
        pytest.fail("Could not find a standard test case in AUDIO_TEST_CASES")

    test_audio_path_rel_assets_str = test_case[0]
    test_audio_filename = test_case[4]
    device = "cpu"  # Use CPU for simplicity

    test_audio_path_rel_assets = Path(test_audio_path_rel_assets_str)
    test_audio_path_abs = (ASSETS_DIR / "audio" / "real" /
                           test_audio_path_rel_assets).resolve()

    if not test_audio_path_abs.is_file():
        pytest.fail(f"Test audio file not found: {test_audio_path_abs}")

    # --- Prepare Input File List --- #
    input_list_file = tmp_path / f"input_list_overwrite_{Path(test_audio_filename).stem}.txt"
    with open(input_list_file, 'w') as f:
        f.write(f"{str(test_audio_path_abs)}\n")
    logger.debug(
        f"Created input file list for overwrite test: {input_list_file}")

    # --- Prepare Output Dir and Paths --- #
    output_dir = tmp_path / f"test_output_overwrite_{Path(test_audio_filename).stem}"
    logger.debug(
        f"E2E overwrite test output base directory: {output_dir.resolve()}")

    # Calculate expected output path (similar logic to the main test)
    input_data_root_actual = test_audio_path_abs.parent
    try:
        relative_audio_path_for_output = test_audio_path_abs.relative_to(
            input_data_root_actual)
    except ValueError:
        pytest.fail(
            f"Could not determine relative path for {test_audio_path_abs} relative to its parent {input_data_root_actual}"
        )
    expected_output_path = output_dir / relative_audio_path_for_output.with_suffix(
        ".csv")

    # --- Command Base --- #
    command_base = [
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
        device,
    ]

    # --- 1. First Run (Generate Initial Output) --- #
    logger.info("Running first pass to generate initial output...")
    result_first = subprocess.run(command_base,
                                  capture_output=True,
                                  text=True,
                                  check=False)
    print(f"STDOUT (First Run):\n{result_first.stdout[-500:]}")
    print(f"STDERR (First Run):\n{result_first.stderr[-500:]}")
    assert result_first.returncode == 0, "First run failed"
    assert "Traceback" not in result_first.stderr, "Traceback in first run stderr"
    assert expected_output_path.is_file(
    ), "Output file not created after first run"
    mtime_first = os.path.getmtime(expected_output_path)
    logger.info(
        f"Output file created: {expected_output_path} (mtime: {mtime_first})")

    # Wait a second to ensure modification times can differ
    time.sleep(1.1)

    # --- 2. Second Run (Default Skip Behavior) --- #
    logger.info("Running second pass to test default skipping...")
    command_skip = command_base[:]  # Copy base command
    result_skip = subprocess.run(command_skip,
                                 capture_output=True,
                                 text=True,
                                 check=False)
    print(f"STDOUT (Skip Run):\n{result_skip.stdout[-500:]}")
    print(f"STDERR (Skip Run):\n{result_skip.stderr[-500:]}")
    assert result_skip.returncode == 0, "Skip run failed"
    assert "Traceback" not in result_skip.stderr, "Traceback in skip run stderr"

    # Check logs for skipping message
    skip_msg_found = False
    log_text_skip = result_skip.stdout + result_skip.stderr  # Combine stdout and stderr for checking
    # Use Path object in f-string for consistency
    if f"Skipping {test_audio_filename} as output {expected_output_path} already exists." in log_text_skip:
        skip_msg_found = True
    assert skip_msg_found, f"Expected skipping log message not found for {test_audio_filename} in combined logs"
    logger.info("Skipping log message found.")

    # Check modification time hasn't changed
    mtime_skip = os.path.getmtime(expected_output_path)
    assert mtime_skip == mtime_first, f"Output file modification time changed during skip run! ({mtime_skip} != {mtime_first})"
    logger.info(
        f"Output file modification time unchanged (mtime: {mtime_skip})")

    # Wait again
    time.sleep(1.1)

    # --- 3. Third Run (Force Overwrite) --- #
    logger.info("Running third pass to test --force-overwrite...")
    command_overwrite = command_base + ["--force_overwrite"]
    result_overwrite = subprocess.run(command_overwrite,
                                      capture_output=True,
                                      text=True,
                                      check=False)
    print(f"STDOUT (Overwrite Run):\n{result_overwrite.stdout[-500:]}")
    print(f"STDERR (Overwrite Run):\n{result_overwrite.stderr[-500:]}")
    assert result_overwrite.returncode == 0, "Overwrite run failed"
    assert "Traceback" not in result_overwrite.stderr, "Traceback in overwrite run stderr"

    # Check logs to ensure skipping message is NOT present
    skip_msg_found_overwrite = False
    log_text_overwrite = result_overwrite.stdout + result_overwrite.stderr  # Combine stdout and stderr
    if f"Skipping {test_audio_filename} as output {expected_output_path} already exists." in log_text_overwrite:
        skip_msg_found_overwrite = True
    assert not skip_msg_found_overwrite, f"Skipping log message unexpectedly found during --force-overwrite run"
    logger.info("Skipping log message correctly absent during overwrite run.")

    # Check modification time HAS changed
    mtime_overwrite = os.path.getmtime(expected_output_path)
    assert mtime_overwrite > mtime_first, f"Output file modification time did not change during overwrite run! ({mtime_overwrite} <= {mtime_first})"
    logger.info(
        f"Output file modification time updated (mtime: {mtime_overwrite})")
