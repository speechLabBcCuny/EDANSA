import subprocess
import sys
from pathlib import Path
import pytest
import os
import logging
import shutil
import time

# Import shared data from test_data.py
from .test_data import (ASSETS_DIR, MODEL_PT, MODEL_CONFIG, AUDIO_TEST_CASES)

# Get logger for this module
logger = logging.getLogger(__name__)


def _run_inference_command(command, test_name):
    """Helper function to run inference command and print output."""
    logger.info(f"Running command for {test_name}: {' '.join(command)}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,  # Don't check=True, assert returncode below
        timeout=600)  # Add timeout (e.g., 10 minutes)

    print(f"\nSTDOUT ({test_name}):\n{result.stdout[-1000:]}")
    print(f"STDERR ({test_name}):\n{result.stderr[-1000:]}")
    assert "Traceback" not in result.stderr, f"Traceback detected in stderr for {test_name}"
    assert result.returncode == 0, f"Inference script failed for {test_name} with return code {result.returncode}"
    return result


@pytest.mark.e2e
def test_inference_with_input_folder(tmp_path):
    """Test inference using the --input_folder argument."""
    # --- Select two test cases --- #
    test_cases = [
        case for case in AUDIO_TEST_CASES
        if isinstance(case, tuple) and "anwr" in case[0]
    ][0:1]  # First ANWR case
    test_cases.append([
        case for case in AUDIO_TEST_CASES
        if isinstance(case, tuple) and "dalton" in case[0]
    ][0])  # First Dalton case

    if len(test_cases) < 2:
        pytest.fail("Could not find two suitable test cases.")

    # --- Prepare Temporary Input Folder --- #
    input_folder = tmp_path / "input_folder_test"
    output_folder = tmp_path / "output_folder_test"
    output_folder.mkdir()

    expected_output_paths = []

    for i, case in enumerate(test_cases):
        rel_audio_path_str = case[0]
        abs_audio_path = (ASSETS_DIR / "audio" / "real" /
                          rel_audio_path_str).resolve()
        if not abs_audio_path.is_file():
            pytest.fail(f"Test audio file not found: {abs_audio_path}")

        # Create a subdirectory structure within the input folder
        sub_dir = input_folder / f"subdir_{i}" / Path(rel_audio_path_str).parent
        sub_dir.mkdir(parents=True, exist_ok=True)
        dest_path = sub_dir / Path(rel_audio_path_str).name
        shutil.copy2(abs_audio_path, dest_path)
        logger.debug(f"Copied {abs_audio_path.name} to {dest_path}")

        # Calculate expected output path relative to input_folder
        rel_output_path_from_input = dest_path.relative_to(input_folder)
        expected_output_paths.append(
            (output_folder / rel_output_path_from_input).with_suffix(".csv"))

    # --- Construct Command --- #
    command = [
        sys.executable,
        "-m",
        "runs.augment.inference",
        "--model_path",
        str(MODEL_PT.resolve()),
        "--config_file",
        str(MODEL_CONFIG.resolve()),
        "--input_folder",  # Use input folder arg
        str(input_folder.resolve()),
        "--output_folder",
        str(output_folder.resolve()),
        "--channel_selection_method",
        "average",  # Keep simple
        "--device",
        "cpu",
    ]

    # --- Run and Assert --- #
    _run_inference_command(command, "input_folder_test")

    # Verify output files exist
    for expected_path in expected_output_paths:
        print(f"Checking for generated file: {expected_path}")
        assert expected_path.is_file(), \
            f"Expected output file not found at {expected_path} for input_folder test"
        # Optional: Add basic check for content (e.g., not empty)
        assert os.path.getsize(
            expected_path) > 0, f"Output file {expected_path} is empty."

    logger.info("Input folder test successful.")


@pytest.mark.e2e
def test_inference_with_multi_file_list(tmp_path):
    """Test inference using --input_files_list with multiple absolute paths."""
    # --- Select two test cases --- #
    test_cases = [
        case for case in AUDIO_TEST_CASES
        if isinstance(case, tuple) and "anwr" in case[0]
    ][0:1]  # First ANWR case
    test_cases.append([
        case for case in AUDIO_TEST_CASES
        if isinstance(case, tuple) and "dalton" in case[0]
    ][0])  # First Dalton case

    if len(test_cases) < 2:
        pytest.fail("Could not find two suitable test cases.")

    # --- Prepare Input File List --- #
    input_list_file = tmp_path / "multi_file_list.txt"
    output_folder = tmp_path / "output_list_test"
    output_folder.mkdir()

    abs_audio_paths = []
    original_rel_paths = []  # Store original relative paths for output check
    for case in test_cases:
        rel_audio_path_str = case[0]
        abs_path = (ASSETS_DIR / "audio" / "real" /
                    rel_audio_path_str).resolve()
        if not abs_path.is_file():
            pytest.fail(f"Test audio file not found: {abs_path}")
        abs_audio_paths.append(str(abs_path))
        original_rel_paths.append(
            Path(rel_audio_path_str))  # Keep as Path object

    with open(input_list_file, 'w') as f:
        for path_str in abs_audio_paths:
            f.write(f"{path_str}\n")
    logger.debug(f"Created multi-file input list: {input_list_file}")

    # --- Determine Expected Common Root --- #
    # Since we used absolute paths from assets, the common root should be ASSETS_DIR/audio/real
    expected_input_root = (ASSETS_DIR / "audio" / "real").resolve()
    logger.info(f"Expected common input root: {expected_input_root}")

    # --- Construct Command --- #
    command = [
        sys.executable,
        "-m",
        "runs.augment.inference",
        "--model_path",
        str(MODEL_PT.resolve()),
        "--config_file",
        str(MODEL_CONFIG.resolve()),
        "--input_files_list",  # Use input list arg
        str(input_list_file.resolve()),
        "--output_folder",
        str(output_folder.resolve()),
        "--channel_selection_method",
        "average",  # Keep simple
        "--device",
        "cpu",
    ]

    # --- Run and Assert --- #
    _run_inference_command(command, "multi_file_list_test")

    # Verify output files exist based on original relative paths from common root
    for rel_path in original_rel_paths:
        expected_output_path = (output_folder / rel_path).with_suffix(".csv")
        print(f"Checking for generated file: {expected_output_path}")
        assert expected_output_path.is_file(), \
            f"Expected output file not found at {expected_output_path} for multi_file_list test"
        assert os.path.getsize(
            expected_output_path
        ) > 0, f"Output file {expected_output_path} is empty."

    logger.info("Multi-file list test successful.")
