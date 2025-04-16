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


@pytest.mark.e2e
def test_inference_with_corrupt_file(tmp_path):
    """Test inference handling of a corrupted audio file."""
    # --- Prepare Input File List with Corrupt File --- #
    input_list_file = tmp_path / "corrupt_file_list.txt"
    output_folder = tmp_path / "output_corrupt_test"
    output_folder.mkdir()

    # Path to the dummy corrupt file created earlier
    corrupt_file_path = (ASSETS_DIR / "audio" / "dummy" / "corrupt" /
                         "corrupt_dummy.wav").resolve()
    if not corrupt_file_path.is_file():
        pytest.fail(f"Dummy corrupt test file not found: {corrupt_file_path}")

    with open(input_list_file, 'w') as f:
        f.write(f"{str(corrupt_file_path)}\\n")
    logger.debug(f"Created corrupt file input list: {input_list_file}")

    # --- Determine Expected Input Root --- #
    # The common root should be the directory containing the dummy file
    expected_input_root = corrupt_file_path.parent.resolve()
    logger.info(
        f"Expected common input root for corrupt file: {expected_input_root}")

    # --- Construct Command --- #
    command = [
        sys.executable,
        "-m",
        "runs.augment.inference",  # Assuming this is the entry point module
        "--model_path",
        str(MODEL_PT.resolve()),
        "--config_file",
        str(MODEL_CONFIG.resolve()),
        "--input_files_list",  # Use input list arg
        str(input_list_file.resolve()),
        "--output_folder",
        str(output_folder.resolve()),
        "--channel_selection_method",
        "average",  # Method doesn't matter much here, as loading should fail
        "--device",
        "cpu",
    ]

    # --- Run and Assert --- #
    # We expect the script to run successfully (returncode 0) but log the error
    logger.info(f"Running command for corrupt_file_test: {' '.join(command)}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,  # Don't raise exception on non-zero exit
        timeout=600)

    print(f"\\nSTDOUT (corrupt_file_test):\\n{result.stdout[-1000:]}")
    print(f"STDERR (corrupt_file_test):\\n{result.stderr[-1000:]}")

    # The script should complete without crashing
    assert result.returncode == 0, f"Inference script failed unexpectedly for corrupt_file_test with return code {result.returncode}"
    assert "Traceback" not in result.stderr, f"Traceback detected in stderr for corrupt_file_test"

    # Verify failed_files.log exists and contains the corrupt file name
    failed_log_path = output_folder / "failed_files.log"
    assert failed_log_path.is_file(
    ), f"Expected {failed_log_path} was not created."

    with open(failed_log_path, 'r') as f:
        log_content = f.read()

    assert corrupt_file_path.name in log_content, \
        f"Corrupt file '{corrupt_file_path.name}' not found in {failed_log_path}. Content:\\n{log_content}"

    # Verify no predictions file was created for the corrupt file
    expected_output_path = (
        output_folder /
        corrupt_file_path.relative_to(expected_input_root)).with_suffix(".csv")
    assert not expected_output_path.exists(), \
        f"Output file {expected_output_path} was unexpectedly created for corrupt input."

    logger.info("Corrupt file handling test successful.")


@pytest.mark.e2e
def test_inference_with_zero_length_file(tmp_path):
    """Test inference handling of a zero-length audio file."""
    input_list_file = tmp_path / "zero_length_list.txt"
    output_folder = tmp_path / "output_zero_length_test"
    output_folder.mkdir()

    zero_len_file_path = (ASSETS_DIR / "audio" / "dummy" / "error_cases" /
                          "zero_length.wav").resolve()
    if not zero_len_file_path.is_file():
        pytest.fail(
            f"Dummy zero-length test file not found: {zero_len_file_path}")

    with open(input_list_file, 'w') as f:
        f.write(f"{str(zero_len_file_path)}\n")

    expected_input_root = zero_len_file_path.parent.resolve()

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
        str(output_folder.resolve()),
        "--device",
        "cpu",
    ]

    logger.info(f"Running command for zero_length_test: {' '.join(command)}")
    result = subprocess.run(command,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=600)

    print(f"\nSTDOUT (zero_length_test):\n{result.stdout[-1000:]}")
    print(f"STDERR (zero_length_test):\n{result.stderr[-1000:]}")

    assert result.returncode == 0, f"Inference script failed unexpectedly for zero_length_test with return code {result.returncode}"
    assert "Traceback" not in result.stderr, f"Traceback detected in stderr for zero_length_test"

    failed_log_path = output_folder / "failed_files.log"
    assert failed_log_path.is_file(
    ), f"Expected {failed_log_path} was not created."
    with open(failed_log_path, 'r') as f:
        log_content = f.read()
    assert zero_len_file_path.name in log_content, \
        f"Zero-length file '{zero_len_file_path.name}' not found in {failed_log_path}. Content:\n{log_content}"

    expected_output_path = (
        output_folder /
        zero_len_file_path.relative_to(expected_input_root)).with_suffix(".csv")
    assert not expected_output_path.exists(), \
        f"Output file {expected_output_path} was unexpectedly created for zero-length input."

    logger.info("Zero-length file handling test successful.")


@pytest.mark.e2e
def test_inference_with_very_short_file(tmp_path):
    """Test inference handling of a valid but very short audio file."""
    input_list_file = tmp_path / "very_short_list.txt"
    output_folder = tmp_path / "output_very_short_test"
    output_folder.mkdir()

    very_short_file_path = (ASSETS_DIR / "audio" / "dummy" / "error_cases" /
                            "very_short.wav").resolve()
    if not very_short_file_path.is_file():
        pytest.fail(
            f"Dummy very short test file not found: {very_short_file_path}")

    with open(input_list_file, 'w') as f:
        f.write(f"{str(very_short_file_path)}\n")

    expected_input_root = very_short_file_path.parent.resolve()

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
        str(output_folder.resolve()),
        "--device",
        "cpu",
    ]

    logger.info(f"Running command for very_short_test: {' '.join(command)}")
    result = _run_inference_command(
        command, "very_short_test")  # Use helper that checks return code

    # Verify NO failed_files.log was created (or is empty if created by prior tests in sequence)
    failed_log_path = output_folder / "failed_files.log"
    if failed_log_path.exists():
        with open(failed_log_path, 'r') as f:
            log_content = f.read()
            assert very_short_file_path.name not in log_content, \
                f"Very short file '{very_short_file_path.name}' was unexpectedly logged as failed in {failed_log_path}."

    # Verify predictions file WAS created
    expected_output_path = (
        output_folder / very_short_file_path.relative_to(expected_input_root)
    ).with_suffix(".csv")
    assert expected_output_path.is_file(), \
        f"Expected output file {expected_output_path} was not created for very short input."
    assert os.path.getsize(expected_output_path) > 0, \
        f"Output file {expected_output_path} is empty for very short input."

    logger.info("Very short file handling test successful.")
