"""Tests for the inference._select_inference_channel function."""

import pytest
import numpy as np
import torch
import logging
from edansa import inference

# Sample data from inference_test.py
mono_np = np.arange(10, dtype=np.float32)
stereo_np = np.stack([mono_np * 2, mono_np * 0.5])  # shape (2, 10)
mono_torch = torch.from_numpy(mono_np)
stereo_torch = torch.from_numpy(stereo_np)

# Sample clipping data (Num segments, Num channels)
clipping_np_stereo_ch0_less = np.array([[0.1, 0.5], [0.2, 0.6]],
                                       dtype=np.float32)
clipping_np_stereo_ch1_less = np.array([[0.5, 0.1], [0.6, 0.2]],
                                       dtype=np.float32)
clipping_np_mono = np.array([[0.1], [0.2]], dtype=np.float32)  # Shape (2, 1)

# Define constants for dummy values if not already defined globally in the test file
DUMMY_SR_TEST = 16000
DUMMY_EXCERPT_LEN_TEST = 10


class TestSelectInferenceChannel:

    @pytest.mark.parametrize("data", [mono_np, mono_torch])
    def test_mono_input(self, data):
        """Test that mono input (1D numpy or 1D/2D tensor) is returned as is."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        result = inference._select_inference_channel(
            data,
            'average',
            None,
            'test_file',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        # assert torch.equal(result, data.squeeze()) # Check if squeezing is intended
        if data.ndim == 2:
            assert torch.equal(result, data.squeeze(0))
        else:
            assert torch.equal(result, data)

    @pytest.mark.parametrize("data", [stereo_np, stereo_torch])
    def test_average_method(self, data):
        """Test 'average' method correctly averages channels."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        expected = torch.mean(data.float(), dim=0)
        result = inference._select_inference_channel(
            data,
            'average',
            None,
            'test_file',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("data, expected_channel_data",
                             [(stereo_np, stereo_np[0, :]),
                              (stereo_torch, stereo_torch[0, :])])
    def test_channel_0_method(self, data, expected_channel_data):
        """Test 'channel_0' method selects the first channel."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        if isinstance(expected_channel_data, np.ndarray):
            expected_channel_data = torch.from_numpy(
                expected_channel_data)  # Convert numpy to tensor
        result = inference._select_inference_channel(
            data,
            'channel_0',
            None,
            'test_file',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        assert torch.equal(result, expected_channel_data)

    @pytest.mark.parametrize("data, expected_channel_data",
                             [(stereo_np, stereo_np[1, :]),
                              (stereo_torch, stereo_torch[1, :])])
    def test_channel_1_method(self, data, expected_channel_data):
        """Test 'channel_1' method selects the second channel."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        if isinstance(expected_channel_data, np.ndarray):
            expected_channel_data = torch.from_numpy(
                expected_channel_data)  # Convert numpy to tensor
        result = inference._select_inference_channel(
            data,
            'channel_1',
            None,
            'test_file',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        assert torch.equal(result, expected_channel_data)

    @pytest.mark.parametrize("data", [stereo_np, stereo_torch])
    def test_invalid_channel_index(self, data):
        """Test error for invalid channel index."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        with pytest.raises(ValueError, match="Invalid channel index 2"):
            inference._select_inference_channel(
                data,
                'channel_2',
                None,
                'test_file',
                sr=DUMMY_SR_TEST,
                excerpt_len=DUMMY_EXCERPT_LEN_TEST)

    @pytest.mark.parametrize("data", [stereo_np, stereo_torch])
    def test_invalid_channel_format(self, data):
        """Test error for invalid channel format."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        # Use a format that starts with 'channel_' but has non-integer suffix
        # The function should raise a ValueError during int() conversion
        with pytest.raises(ValueError, match="invalid literal for int()"
                          ):  # Match the int() conversion error
            inference._select_inference_channel(
                data,
                'channel_abc',
                None,
                'test_file',
                sr=DUMMY_SR_TEST,
                excerpt_len=DUMMY_EXCERPT_LEN_TEST)

    @pytest.mark.parametrize(
        "data, clipping_data, expected_channel_data",
        [
            (stereo_np, clipping_np_stereo_ch0_less, stereo_np[0, :]),
            (stereo_torch, clipping_np_stereo_ch0_less,
             stereo_torch[0, :]),  # Use torch clipping data
            (stereo_np, clipping_np_stereo_ch1_less, stereo_np[1, :]),
            (stereo_torch, clipping_np_stereo_ch1_less,
             stereo_torch[1, :]),  # Use torch clipping data
        ])
    def test_clipping_method_valid(self, data, clipping_data,
                                   expected_channel_data):
        """Test 'clipping' method selects the channel with minimum clipping."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        # Clipping data should already be tensor from parametrize if needed
        if isinstance(expected_channel_data, np.ndarray):
            expected_channel_data = torch.from_numpy(
                expected_channel_data)  # Convert numpy to tensor
        if isinstance(clipping_data, np.ndarray):
            clipping_data = torch.from_numpy(
                clipping_data)  # Convert numpy to tensor

        result = inference._select_inference_channel(
            data,
            'clipping',
            clipping_data,
            'test_file',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        assert torch.equal(result, expected_channel_data)

    @pytest.mark.parametrize("data", [stereo_np, stereo_torch])
    def test_clipping_method_missing_data(self, data, caplog):
        """Test warning and fallback when clipping data is missing."""
        caplog.set_level(logging.WARNING)  # Explicitly set level for this test
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor

        # Should fallback to average
        expected = torch.mean(data.float(), dim=0)
        result = inference._select_inference_channel(
            data,
            'clipping',
            None,
            'test_file_missing',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        assert torch.allclose(result, expected)
        # Assert based on the ACTUAL warning message from the code
        assert f"Clipping method requested for file 'test_file_missing' but no clipping data provided. Falling back to 'average' method." in caplog.text

    @pytest.mark.parametrize("data", [stereo_np, stereo_torch])
    def test_clipping_method_mismatched_channels(self, data, caplog):
        """Test warning and fallback when clipping channels mismatch audio."""
        caplog.set_level(logging.WARNING)  # Explicitly set level for this test
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        # Use mono clipping data for mismatch
        clipping_mono_tensor = torch.from_numpy(clipping_np_mono)

        # Should fallback to average
        expected = torch.mean(data.float(), dim=0)
        result = inference._select_inference_channel(
            data,
            'clipping',
            clipping_mono_tensor,
            'test_file_mismatch',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        assert torch.allclose(result, expected)
        # Assert based on the ACTUAL warning message from the code
        assert f"Clipping method requested for file 'test_file_mismatch' but clipping data dimensions mismatch audio channels" in caplog.text
        assert f"Falling back to 'average' method." in caplog.text

    @pytest.mark.parametrize("data", [stereo_np, stereo_torch])
    def test_unknown_method(self, data):
        """Test error for unknown channel selection method."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        with pytest.raises(ValueError,
                           match="Unknown channel selection method: 'weird'"):
            inference._select_inference_channel(
                data,
                'weird',
                None,
                'test_file',
                sr=DUMMY_SR_TEST,
                excerpt_len=DUMMY_EXCERPT_LEN_TEST)

    def test_unsupported_type(self):
        """Test TypeError for unsupported input data types (non-tensor/numpy)."""
        with pytest.raises(TypeError, match="Unsupported data type"):
            # Pass a list instead of tensor/array
            inference._select_inference_channel(
                [1, 2, 3],
                'average',
                None,
                'test_file',
                sr=DUMMY_SR_TEST,
                excerpt_len=DUMMY_EXCERPT_LEN_TEST)

    def test_invalid_dimensions_np(self):
        """Test ValueError on numpy array with incorrect dimensions."""
        # Create a 3D numpy array which is invalid for the function
        invalid_data_np = np.random.rand(2, 2, 10).astype(np.float32)
        invalid_data_torch = torch.from_numpy(
            invalid_data_np)  # Convert to tensor
        with pytest.raises(ValueError, match="Unsupported tensor dimension: 3"):
            inference._select_inference_channel(
                invalid_data_torch,  # Pass tensor
                'average',
                None,
                'test_invalid_dim_np',
                DUMMY_SR_TEST,
                DUMMY_EXCERPT_LEN_TEST)

    def test_invalid_dimensions_torch(self):
        """Test ValueError for tensor with invalid dimensions."""
        data_3d_tensor = torch.rand(2, 2, 10)
        with pytest.raises(ValueError, match="Unsupported tensor dimension: 3"):
            inference._select_inference_channel(
                data_3d_tensor,
                'average',
                None,
                'test_file',
                sr=DUMMY_SR_TEST,
                excerpt_len=DUMMY_EXCERPT_LEN_TEST)

    def test_single_channel_stereo_input(self):
        """Test case where a 2D tensor has only one channel."""
        single_channel_tensor = torch.rand(1, 10)
        result = inference._select_inference_channel(
            single_channel_tensor,
            'average',
            None,
            'test_file',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        # Should return the single channel squeezed
        assert torch.equal(result, single_channel_tensor.squeeze(0))
        assert result.ndim == 1

    @pytest.mark.parametrize("data", [stereo_np, stereo_torch])
    def test_clipping_method_fallback_on_missing_data(self, data, caplog):
        """Test 'clipping' method falls back to average when clipping data is None."""
        caplog.set_level(logging.WARNING)  # Explicitly set level for this test
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        # The function should now succeed and return the average, not raise ValueError
        result = inference._select_inference_channel(
            data,
            'clipping',
            None,
            'test_file_fallback',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        assert torch.allclose(result, torch.mean(data.float(), dim=0))
        # Assert based on the ACTUAL warning message from the code
        assert f"Clipping method requested for file 'test_file_fallback' but no clipping data provided. Falling back to 'average' method." in caplog.text

    @pytest.mark.parametrize("data", [stereo_np, stereo_torch])
    def test_clipping_method_fallback_on_mismatched_channels(
            self, data, caplog):
        """Test 'clipping' method falls back to average when channels mismatch."""
        caplog.set_level(logging.WARNING)  # Explicitly set level for this test
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)  # Convert numpy to tensor
        # Use mono clipping data for mismatch
        clipping_mono_tensor = torch.from_numpy(clipping_np_mono)
        result = inference._select_inference_channel(
            data,
            'clipping',
            clipping_mono_tensor,  # Use tensor version
            'test_file_fallback2',
            sr=DUMMY_SR_TEST,
            excerpt_len=DUMMY_EXCERPT_LEN_TEST)
        assert torch.allclose(result, torch.mean(data.float(), dim=0))
        # Assert based on the ACTUAL warning message from the code
        assert f"Clipping method requested for file 'test_file_fallback2' but clipping data dimensions mismatch audio channels" in caplog.text
        assert f"Falling back to 'average' method." in caplog.text
