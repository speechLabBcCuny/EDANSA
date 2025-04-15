"""Tests for clippingutils module with various audio loading backends."""

import pytest
import numpy as np
import torch
import tempfile
import shutil
import os
import logging
from pathlib import Path

from edansa import clippingutils
from edansa import audio

# Set up logging
logger = logging.getLogger(__name__)

# Define a list of test audio files
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

# Filter for WAV files specifically for the dtype comparison test
WAV_TEST_FILES = [f for f in TEST_FILES if f.suffix.lower() == '.wav']

# For backwards compatibility
TEST_AUDIO_FILE = TEST_FILES[0]

# Determine devices to test on
devices_to_test = ['cpu']
if torch.cuda.is_available():
    devices_to_test.append('cuda')
else:
    # If CUDA is not available, add a marker so the test isn't completely skipped if parametrized
    # This uses pytest's conditional skipping mechanism implicitly via parametrization
    pass


@pytest.mark.parametrize("audio_file", TEST_FILES)
def test_get_clipping_percent_with_audio_file(audio_file):
    """Test get_clipping_percent with audio loaded via torchaudio."""
    # Load audio with torchaudio backend
    sound_array, _ = audio.load(audio_file,
                                dtype=torch.float32,
                                backend=None,
                                normalize=True)

    # Get clipping percentage directly with torch tensor
    results_tensor = clippingutils.get_clipping_percent(sound_array,
                                                        threshold=0.9)

    # Validate results
    assert isinstance(results_tensor, torch.Tensor)
    # Assuming input shape (C, S) or (S,), output should be (C,) or ()
    # Check ndim corresponds to input channels
    expected_ndim = max(0, sound_array.ndim - 1)
    assert results_tensor.ndim == expected_ndim

    # Convert tensor to numpy for numerical checks
    results_np = results_tensor.cpu().numpy()

    # Check range
    assert np.all((results_np >= 0) & (results_np <= 1))

    # Manual calculation for validation
    sound_array_np = sound_array.cpu().numpy() if isinstance(
        sound_array, torch.Tensor) else sound_array
    threshold_val = 0.9
    minval = -threshold_val
    maxval = threshold_val * 0.9999  # Match internal logic

    if sound_array_np.ndim == 1:
        clipped_count = np.sum(sound_array_np <= minval) + np.sum(
            sound_array_np >= maxval)
        manual_result = clipped_count / sound_array_np.size if sound_array_np.size > 0 else 0.0
        assert np.allclose(results_np, manual_result)  # Compare scalar/0D
    elif sound_array_np.ndim == 2:
        clipped_count = np.sum(sound_array_np <= minval, axis=1) + np.sum(
            sound_array_np >= maxval, axis=1)
        manual_results = clipped_count / sound_array_np.shape[
            -1] if sound_array_np.shape[-1] > 0 else np.zeros(
                sound_array_np.shape[0])
        assert np.allclose(results_np, manual_results)  # Compare 1D array


@pytest.mark.parametrize("audio_file", TEST_FILES)
def test_get_clipping_percent_file_with_audio_file(audio_file, debug_mode):
    """Test get_clipping_percent_file with audio loaded via torchaudio."""
    # Load audio with torchaudio backend
    sound_array, sr = audio.load(audio_file,
                                 dtype=torch.float32,
                                 backend=None,
                                 normalize=True)

    # Set segment length (in seconds)
    segment_len = 5

    # Get clipping percentage per segment directly with torch tensor
    # Result is now a torch.Tensor
    results_tensor = clippingutils.get_clipping_percent_file(
        sound_array, sr, segment_len, clipping_threshold=0.9)

    # Validate results
    assert isinstance(results_tensor, torch.Tensor)
    assert results_tensor.dtype == torch.float32

    # Convert original sound_array to numpy ONLY for calculating expected segments
    # (as the segmentation logic in the function currently uses numpy)
    sound_array_np = sound_array.cpu().numpy() if isinstance(
        sound_array, torch.Tensor) else sound_array

    # Log debug info
    logger.debug("File: %s", audio_file)
    logger.debug("Audio shape (numpy for calculation): %s",
                 sound_array_np.shape)
    logger.debug("Sample rate: %s", sr)
    logger.debug("Result tensor shape: %s", results_tensor.shape)
    logger.debug("Number of results segments from tensor: %s",
                 results_tensor.shape[0])

    # Calculate expected number of segments based on the numpy array length
    total_samples = sound_array_np.shape[-1]
    segment_samples = segment_len * sr

    # --- NEW CALCULATION: Use ceiling division to account for padding --- #
    if segment_samples > 0:
        expected_segments = int(np.ceil(total_samples / segment_samples))
    else:
        expected_segments = 0  # Avoid division by zero

    logger.debug("Expected segments calculated (padding logic): %s",
                 expected_segments)

    # Check the number of segments in the result tensor
    assert results_tensor.shape[0] == expected_segments

    # Check result tensor dimensionality (should be >= 1: [segments, ...])
    assert results_tensor.ndim >= 1
    # If original audio was stereo (or more channels), result should reflect that
    # Example: input (C, S) -> result (num_segments, C)
    # Example: input (S,) -> result (num_segments,) - view as (num_segments, 1) implicitly?
    # Let's check the number of dimensions AFTER the segment dimension
    expected_leading_dims = len(
        sound_array_np.shape[:-1])  # Number of dims before samples
    # If input was 1D (samples,), expected_leading_dims is 0. Result shape (N,) -> ndim=1.
    # If input was 2D (C, S), expected_leading_dims is 1. Result shape (N, C) -> ndim=2.
    # The results_tensor shape should be (num_segments,) + shape_of_leading_dims
    assert results_tensor.ndim == 1 + expected_leading_dims
    if expected_leading_dims > 0:
        assert results_tensor.shape[1:] == sound_array_np.shape[:-1]


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.mark.parametrize("audio_file", TEST_FILES)
def test_run_task_save_with_backends(audio_file, temp_output_dir):
    """Test run_task_save uses the default ffmpeg backend."""
    # Generate a unique ID for each file to avoid overwriting results
    file_id = audio_file.stem

    # Test with default (ffmpeg) backend
    results_dict_torch, errors_torch = clippingutils.run_task_save(
        [str(audio_file)],
        f"test_{file_id}_torch",
        temp_output_dir,
        clipping_threshold=0.9,
        segment_len=5,
        # audio_load_backend removed, uses default
        save=True)

    # Validate results
    assert isinstance(results_dict_torch, dict)
    assert str(audio_file) in results_dict_torch
    # run_task_save calls get_clipping_percent_file, which returns a tensor
    assert isinstance(results_dict_torch[str(audio_file)], np.ndarray)
    assert len(errors_torch) == 0  # No errors should occur

    # Verify the file was created
    output_file_path = os.path.join(temp_output_dir,
                                    f"test_{file_id}_torch_0,9.pkl")
    assert os.path.exists(output_file_path)


@pytest.mark.parametrize("audio_file", WAV_TEST_FILES)
def test_compare_dtypes_with_audio_file(audio_file, debug_mode):
    """Test that clipping detection gives consistent results for different dtypes using ffmpeg.

    This test only runs on WAV files because loading unnormalized data might fail
    for other formats like FLAC with certain backends.
    """
    logger.debug("Testing file: %s", audio_file)

    # Load the same audio file with different dtypes using ffmpeg
    sound_array_float, sr_float = audio.load(audio_file,
                                             dtype=torch.float32,
                                             backend=None,
                                             normalize=True)

    sound_array_int, sr_int = audio.load(
        audio_file,
        dtype=np.int16,  # Should use torch.int16 internally but return np
        backend=None,
        normalize=False)  # Load unnormalized int16

    assert sr_float == sr_int

    # Ensure float is torch.Tensor and int is np.ndarray (as per audio.load logic)
    assert isinstance(sound_array_float, torch.Tensor)
    assert isinstance(sound_array_int, np.ndarray)

    # Check data stats
    float_max_abs = torch.max(torch.abs(sound_array_float)).item()
    logger.debug("Float audio max absolute value: %.6f", float_max_abs)
    logger.debug("Float audio dtype: %s", sound_array_float.dtype)
    assert float_max_abs <= 1.0 + 1e-4  # Should be normalized

    int_max_abs = np.max(np.abs(sound_array_int))
    int_info = np.iinfo(np.int16)
    logger.debug("Int audio max absolute value: %s (int16 max: %s)",
                 int_max_abs, int_info.max)
    logger.debug("Int audio dtype: %s", sound_array_int.dtype)
    assert int_max_abs <= int_info.max  # Should be within int16 range

    # Test with multiple thresholds using get_clipping_percent directly
    thresholds = [0.9, 0.95, 0.99, 1.0]
    for threshold in thresholds:
        # get_clipping_percent now returns tensors
        float_clipping_tensor = clippingutils.get_clipping_percent(
            sound_array_float, threshold=threshold)
        int_clipping_tensor = clippingutils.get_clipping_percent(
            sound_array_int, threshold=threshold)

        # Assuming mono/stereo, results should be 1D tensor
        assert float_clipping_tensor.ndim == 1
        assert int_clipping_tensor.ndim == 1
        assert float_clipping_tensor.shape == int_clipping_tensor.shape

        # Convert to numpy for logging and comparison with allclose
        float_clipping = float_clipping_tensor.cpu().numpy()
        int_clipping = int_clipping_tensor.cpu().numpy()

        logger.debug(
            "Threshold %.2f - Float clipping: %.6f, Int clipping: %.6f, Difference: %.6f",
            threshold,
            float_clipping[0],
            int_clipping[0],  # Access first element for logging
            abs(float_clipping[0] - int_clipping[0]))

        # Compare clipping values with tolerance
        if np.max([float_clipping, int_clipping]) > 0.00001:
            assert np.all(float_clipping > 0) and np.all(int_clipping > 0)

        assert np.allclose(float_clipping, int_clipping, rtol=0, atol=0.001)

    # Test segment-based function
    segment_len = 5
    # get_clipping_percent_file now returns tensors
    float_segments_tensor = clippingutils.get_clipping_percent_file(
        sound_array_float, sr_float, segment_len, 0.99)
    int_segments_tensor = clippingutils.get_clipping_percent_file(
        sound_array_int, sr_int, segment_len, 0.99)

    # Convert tensors to numpy arrays for comparison
    float_segments = float_segments_tensor.cpu().numpy()
    int_segments = int_segments_tensor.cpu().numpy()

    # Compare shapes (number of segments, number of channels)
    assert float_segments.shape == int_segments.shape

    # Compare segment results
    # Use np.allclose for element-wise comparison of the numpy arrays
    assert np.allclose(float_segments, int_segments, rtol=0, atol=0.001)


@pytest.mark.parametrize("device", devices_to_test)
def test_get_clipping_percent_with_synthetic_data(device):
    """Test get_clipping_percent with synthetic audio data with known clipping on specified device."""
    # Skip message if CUDA is specified but not available (handled by parametrize)
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA device not available, skipping GPU test.")

    logger.debug("Running synthetic test on device: %s", device)

    # --- Basic Float Test (Torch Tensor) ---
    sample_count = 10000
    # Create tensor directly on the target device if possible, otherwise move it
    sound_tensor_float = (torch.rand(sample_count, dtype=torch.float32) * 0.8 -
                          0.4).to(device)
    clip_indices_max = torch.randperm(sample_count).to(
        device)[:int(sample_count * 0.1)]
    sound_tensor_float[clip_indices_max] = 1.0
    clip_indices_min = torch.randperm(sample_count).to(
        device)[:int(sample_count * 0.05)]
    sound_tensor_float[clip_indices_min] = -1.0
    expected_clipping_float = 0.15

    # Returns 0D tensor for 1D input
    results_float_tensor = clippingutils.get_clipping_percent(
        sound_tensor_float, threshold=1.0)
    assert isinstance(results_float_tensor, torch.Tensor)
    assert results_float_tensor.ndim == 0
    assert results_float_tensor.device.type == device  # Check device
    # Use item() to compare scalar value
    assert np.allclose(results_float_tensor.item(),
                       expected_clipping_float,
                       rtol=0,
                       atol=0.01)

    # --- Basic Int Test (Torch Tensor) ---
    sound_tensor_int = (sound_tensor_float * 32767).clamp(
        -32768, 32767).short()  # Already on device
    results_int_tensor = clippingutils.get_clipping_percent(sound_tensor_int,
                                                            threshold=1.0)
    expected_clipping_int = (torch.sum(sound_tensor_int <= -32768) + torch.sum(
        sound_tensor_int >= 32767)).float().item() / sample_count
    assert isinstance(results_int_tensor, torch.Tensor)
    assert results_int_tensor.ndim == 0
    assert results_int_tensor.device.type == device
    assert np.allclose(results_int_tensor.item(),
                       expected_clipping_int,
                       rtol=0,
                       atol=0.01)

    # --- NumPy Input Test (NumPy arrays are always CPU) ---
    # get_clipping_percent returns tensor even for numpy input
    if device == 'cpu':
        sound_array_np_float = sound_tensor_float.cpu().numpy()
        results_np_tensor = clippingutils.get_clipping_percent(
            sound_array_np_float, threshold=1.0)
        assert isinstance(results_np_tensor, torch.Tensor)
        assert results_np_tensor.ndim == 0
        assert results_np_tensor.device.type == 'cpu'  # Result should be CPU for numpy input
        assert np.allclose(results_np_tensor.item(),
                           expected_clipping_float,
                           rtol=0,
                           atol=0.01)

        sound_array_np_int = sound_tensor_int.cpu().numpy()
        results_np_int_tensor = clippingutils.get_clipping_percent(
            sound_array_np_int, threshold=1.0)
        assert isinstance(results_np_int_tensor, torch.Tensor)
        assert results_np_int_tensor.ndim == 0
        assert results_np_int_tensor.device.type == 'cpu'
        assert np.allclose(results_np_int_tensor.item(),
                           expected_clipping_int,
                           rtol=0,
                           atol=0.01)

    # --- Stereo Test (2D Tensor) ---
    second_channel = (sound_tensor_float * 0.9)  # Already on device
    stereo_tensor = torch.stack([sound_tensor_float, second_channel
                                ])  # Shape (2, samples), on device
    results_stereo_tensor = clippingutils.get_clipping_percent(stereo_tensor,
                                                               threshold=1.0)
    manual_second_channel_clip = (torch.sum(second_channel <= -1.0) + torch.sum(
        second_channel >= 1.0)).float().item() / sample_count
    expected_stereo = torch.tensor(
        [expected_clipping_float, manual_second_channel_clip], device=device)
    assert isinstance(results_stereo_tensor, torch.Tensor)
    assert results_stereo_tensor.ndim == 1
    assert results_stereo_tensor.shape[0] == 2
    assert results_stereo_tensor.device.type == device
    assert torch.allclose(results_stereo_tensor,
                          expected_stereo,
                          rtol=0,
                          atol=0.01)

    # --- Multi-channel Test (4 channels, 2D Tensor) ---
    ch3 = (sound_tensor_float * 0.5)  # Already on device
    ch4 = torch.zeros_like(sound_tensor_float)  # Already on device
    multi_channel_tensor = torch.stack(
        [sound_tensor_float, second_channel, ch3,
         ch4])  # Shape (4, samples), on device
    results_multi_tensor = clippingutils.get_clipping_percent(
        multi_channel_tensor, threshold=1.0)
    manual_ch3_clip = (torch.sum(ch3 <= -1.0) +
                       torch.sum(ch3 >= 1.0)).float().item() / sample_count
    manual_ch4_clip = 0.0
    expected_multi = torch.tensor([
        expected_clipping_float, manual_second_channel_clip, manual_ch3_clip,
        manual_ch4_clip
    ],
                                  device=device)
    assert isinstance(results_multi_tensor, torch.Tensor)
    assert results_multi_tensor.ndim == 1
    assert results_multi_tensor.shape[0] == 4
    assert results_multi_tensor.device.type == device
    assert torch.allclose(results_multi_tensor,
                          expected_multi,
                          rtol=0,
                          atol=0.01)

    # --- Multi-dimensional Test (3D Tensor) ---
    batch_tensor_3d = torch.stack([stereo_tensor, stereo_tensor * 0.7
                                  ])  # shape (2, 2, S), on device
    results_3d_tensor = clippingutils.get_clipping_percent(
        batch_tensor_3d, threshold=1.0)  # Result shape (2, 2)
    b2_ch1 = batch_tensor_3d[1, 0]
    b2_ch2 = batch_tensor_3d[1, 1]
    manual_b2_ch1_clip = (torch.sum(b2_ch1 <= -1.0) + torch.sum(b2_ch1 >= 1.0)
                         ).float().item() / sample_count
    manual_b2_ch2_clip = (torch.sum(b2_ch2 <= -1.0) + torch.sum(b2_ch2 >= 1.0)
                         ).float().item() / sample_count
    expected_results_3d_flat = [
        expected_clipping_float, manual_second_channel_clip, manual_b2_ch1_clip,
        manual_b2_ch2_clip
    ]
    expected_results_3d = torch.tensor(expected_results_3d_flat,
                                       device=device).view(
                                           2, 2)  # Reshape expected to (2, 2)
    assert isinstance(results_3d_tensor, torch.Tensor)
    assert results_3d_tensor.shape == (2, 2)
    assert results_3d_tensor.device.type == device
    assert torch.allclose(results_3d_tensor,
                          expected_results_3d,
                          rtol=0,
                          atol=0.01)

    # --- Edge Case: Empty Tensors (Create directly on device) ---
    # Expect 0D tensor for shape (0,)
    assert isinstance(
        clippingutils.get_clipping_percent(torch.empty(0, device=device)),
        torch.Tensor)
    assert clippingutils.get_clipping_percent(torch.empty(
        0, device=device)).shape == torch.Size([])  # 0D
    # Expect 1D tensor for shape (2, 0)
    assert isinstance(
        clippingutils.get_clipping_percent(torch.empty(2, 0, device=device)),
        torch.Tensor)
    assert clippingutils.get_clipping_percent(torch.empty(
        2, 0, device=device)).shape == torch.Size([2])
    # Expect 2D tensor for shape (2, 3, 0)
    assert isinstance(
        clippingutils.get_clipping_percent(torch.empty(2, 3, 0, device=device)),
        torch.Tensor)
    assert clippingutils.get_clipping_percent(
        torch.empty(2, 3, 0, device=device)).shape == torch.Size([2, 3])

    # NumPy empty tests only need to run once (on cpu)
    if device == 'cpu':
        assert clippingutils.get_clipping_percent(
            np.empty(0)).shape == torch.Size([])  # 0D tensor
        assert clippingutils.get_clipping_percent(np.empty(
            (2, 0))).shape == torch.Size([2])  # 1D tensor


def test_analyze_high_clipping_segments(debug_mode):
    """Test that analyzes segments with high clipping in more detail."""
    # We'll use the FLAC file since we know it has significant clipping
    audio_file = TEST_FILES[1]

    # Load with torchaudio
    sound_array, sr = audio.load(audio_file,
                                 dtype=torch.float32,
                                 backend=None,
                                 normalize=True)

    # Convert to numpy for easier analysis
    if isinstance(sound_array, torch.Tensor):
        sound_array_np = sound_array.numpy()
    else:
        sound_array_np = sound_array

    # Get clipping by segment
    segment_len = 5
    results = clippingutils.get_clipping_percent_file(sound_array,
                                                      sr,
                                                      segment_len,
                                                      clipping_threshold=0.99)

    # Find the segments with the highest clipping
    clip_percentages = [seg[0] for seg in results]
    top_segments = sorted(enumerate(clip_percentages),
                          key=lambda x: x[1],
                          reverse=True)[:5]

    logger.debug("\nTop 5 segments with highest clipping:")
    for idx, clip_pct in top_segments:
        logger.debug("Segment %s: %.6f clipping", idx, clip_pct)

        # Extract the segment data
        start_sample = idx * segment_len * sr
        end_sample = min((idx + 1) * segment_len * sr, sound_array_np.shape[-1])

        if len(sound_array_np.shape) == 1:
            segment_data = sound_array_np[start_sample:end_sample]
        else:
            segment_data = sound_array_np[:, start_sample:end_sample]

        # Calculate some statistics about the segment
        seg_max = np.max(segment_data)
        seg_min = np.min(segment_data)
        seg_above_thresh = np.sum(segment_data >= 0.99) / segment_data.size
        seg_below_thresh = np.sum(segment_data <= -0.99) / segment_data.size

        logger.debug("  Max value: %.6f", seg_max)
        logger.debug("  Min value: %.6f", seg_min)
        logger.debug("  Samples above 0.99: %.6f", seg_above_thresh)
        logger.debug("  Samples below -0.99: %.6f", seg_below_thresh)

    # Verify our clipping calculation matches the one from get_clipping_percent_file
    # for the highest clipping segment
    highest_idx = top_segments[0][0]
    start_sample = highest_idx * segment_len * sr
    end_sample = min((highest_idx + 1) * segment_len * sr,
                     sound_array_np.shape[-1])

    if len(sound_array_np.shape) == 1:
        segment_data = sound_array_np[start_sample:end_sample]
    else:
        segment_data = sound_array_np[:, start_sample:end_sample]

    manual_clip = clippingutils.get_clipping_percent(segment_data,
                                                     threshold=0.99)
    logger.debug("\nHighest segment %s:", highest_idx)
    logger.debug("  Calculated by get_clipping_percent_file: %.6f",
                 top_segments[0][1])
    logger.debug("  Calculated manually: %.6f", manual_clip[0])

    # The values should match closely
    assert np.allclose(top_segments[0][1], manual_clip[0], rtol=0, atol=0.0001)
