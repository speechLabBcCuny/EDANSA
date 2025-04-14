"""Tests for audio loading functionalities."""

import pytest
import torch
import logging
from pathlib import Path

from edansa import audio

# Set up logging
logger = logging.getLogger(__name__)

# Define a list of test audio files from clippingutils_test
# Assuming this list is available or re-defined here
TEST_FILES = [
    Path(
        'src/edansa/tests/assets/audio/real/dempster/20/2023/S4A10444_20230818_012000.wav'
    ),
    Path(
        'src/edansa/tests/assets/audio/real/dalton/10/2023/S4A10407_20230906_033155.flac'
    ),
    Path(
        'src/edansa/tests/assets/audio/real/anwr/47/2022/S4A10341_20220802_235902.flac'
    )
]


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption("--debug",
                     action="store_true",
                     default=False,
                     help="Enable debug output")


@pytest.fixture
def debug_mode(request):
    """Control whether debug output is displayed."""
    return request.config.getoption("--debug")


@pytest.fixture(autouse=True)
def setup_logging(debug_mode):
    """Set up logging based on debug mode."""
    level = logging.DEBUG if debug_mode else logging.WARNING

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Add the handler to the root logger
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    return level


@pytest.mark.parametrize("audio_file", TEST_FILES)
def test_torchaudio_backend_consistency(audio_file, debug_mode):
    """Compare audio loading between torch_soxio and torch_soundfile backends."""
    logger.debug("Testing torchaudio backend consistency for file: %s",
                 audio_file)

    # 1. Load with torch_soxio (normalized float32 tensor)
    try:
        data_soxio, sr_soxio = audio.load(audio_file,
                                          dtype=torch.float32,
                                          backend="torch_soxio",
                                          normalize=True)
        logger.debug("Soxio loaded shape: %s, sr: %s, dtype: %s",
                     data_soxio.shape, sr_soxio, data_soxio.dtype)
    except Exception as e:
        pytest.fail(f"Failed to load {audio_file} with torch_soxio: {e}")

    # 2. Load with torch_soundfile (normalized float32 tensor)
    try:
        data_soundfile, sr_soundfile = audio.load(audio_file,
                                                  dtype=torch.float32,
                                                  backend="torch_soundfile",
                                                  normalize=True)
        logger.debug("Soundfile loaded shape: %s, sr: %s, dtype: %s",
                     data_soundfile.shape, sr_soundfile, data_soundfile.dtype)
    except Exception as e:
        # Soundfile might fail on certain formats like FLAC without library
        logger.warning("Skipping soundfile test for %s due to load error: %s",
                       audio_file, e)
        pytest.skip(f"Soundfile failed to load {audio_file}: {e}")

    # 3. Compare Sample Rates
    assert sr_soxio == sr_soundfile, \
        f"Sample rates differ: {sr_soxio} (soxio) vs {sr_soundfile} (soundfile)"

    # 4. Compare Shapes
    assert data_soxio.shape == data_soundfile.shape, \
        f"Shapes differ: {data_soxio.shape} (soxio) vs {data_soundfile.shape} (soundfile)"

    # 5. Compare Data (Tensor comparison)
    # Use torch.allclose for tensor comparison
    atol_value = 1e-5  # Tolerance for float comparison
    are_close = torch.allclose(data_soxio,
                               data_soundfile,
                               rtol=0,
                               atol=atol_value)

    if not are_close:
        # Calculate difference statistics if not close for debugging
        diff = torch.abs(data_soxio - data_soundfile)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        mismatched_count = torch.sum(diff > atol_value).item()
        mismatched_percent = (mismatched_count / diff.numel()) * 100
        logger.error(
            "Data mismatch detected between torchaudio backends for %s:",
            audio_file)
        logger.error("  Max difference: %s", max_diff)
        logger.error("  Mean difference: %.6f", mean_diff)
        logger.error("  Samples differing by more than %s: %s (%.4f%%)",
                     atol_value, mismatched_count, mismatched_percent)

        # Optionally show some differing values
        diff_indices = torch.where(diff > atol_value)
        limit = 5
        logger.error("  First %s differing samples (soxio, soundfile, diff):",
                     limit)
        for i in range(min(limit, len(diff_indices[0]))):
            # Construct tuple index for multi-dimensional tensors if necessary
            idx = tuple(d[i].item() for d in diff_indices)
            soxio_val = data_soxio[idx].item()
            soundfile_val = data_soundfile[idx].item()
            diff_val = diff[idx].item()
            logger.error("    idx%s: (%.6f, %.6f, %.6f)", idx, soxio_val,
                         soundfile_val, diff_val)

    assert are_close, \
        f"Audio data differs significantly between torchaudio backends (atol={atol_value})"

    logger.debug("Torchaudio backend consistency test passed for %s",
                 audio_file)
