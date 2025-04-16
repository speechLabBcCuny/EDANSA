"""Tests for audio loading functionalities."""

import pytest
import torch
import logging
from pathlib import Path
import torchaudio

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
    """Compare audio loading between sox and soundfile backends."""
    logger.debug("Testing torchaudio backend consistency for file: %s",
                 audio_file)

    # 1. Load with default backend (normalized float32 tensor)
    try:
        data_default, sr_default = audio.load(audio_file,
                                              dtype=torch.float32,
                                              normalize=True)
        assert isinstance(data_default, torch.Tensor)
        logger.debug("Default backend loaded shape: %s, sr: %s, dtype: %s",
                     data_default.shape, sr_default, data_default.dtype)
    except Exception as e:
        pytest.fail(f"Failed to load {audio_file} with default backend: {e}")

    # Check if soundfile backend is available
    available_backends = torchaudio.list_audio_backends()
    if "soundfile" not in available_backends:
        pytest.skip(
            f"Soundfile backend not available (available: {available_backends}). Skipping comparison."
        )

    # 2. Load with soundfile backend (normalized float32 tensor)
    try:
        data_sf, sr_sf = audio.load(audio_file,
                                    dtype=torch.float32,
                                    backend="soundfile",
                                    normalize=True)
        assert isinstance(data_sf, torch.Tensor)
        logger.debug("Soundfile loaded shape: %s, sr: %s, dtype: %s",
                     data_sf.shape, sr_sf, data_sf.dtype)
    except Exception as e:
        pytest.fail(f"Failed to load {audio_file} with soundfile: {e}")

    # 3. Compare sampling rates
    assert sr_default == sr_sf, f"Sampling rates differ: Default ({sr_default}) vs Soundfile ({sr_sf}) for {audio_file}"

    # 4. Compare shapes (allow for minor differences due to backend handling?)
    assert data_default.shape == data_sf.shape, f"Shapes differ: Default {data_default.shape} vs Soundfile {data_sf.shape} for {audio_file}"

    # 5. Compare content (allow for small numerical differences)
    # Increased tolerance for floating point comparisons
    is_close = torch.allclose(data_default, data_sf, rtol=1e-4, atol=1e-5)
    if not is_close:
        diff = torch.abs(data_default - data_sf)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        logger.warning(
            f"Content mismatch for {audio_file}. Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
        )
        # Optional: Save differing tensors for debugging if debug_mode is set
        if debug_mode:
            # Create a unique filename based on the audio file name
            base_name = Path(audio_file).stem
            diff_dir = Path(debug_mode) / "diffs"
            diff_dir.mkdir(parents=True,
                           exist_ok=True)  # Ensure directory exists
            torch.save(data_default, diff_dir / f"{base_name}_default.pt")
            torch.save(data_sf, diff_dir / f"{base_name}_soundfile.pt")
            logger.info(f"Saved differing tensors to {diff_dir}")

    assert is_close, f"Content differs significantly between default and soundfile backends for {audio_file}. Max diff: {max_diff:.6f}"

    logger.debug("Torchaudio backend consistency test passed for %s",
                 audio_file)
