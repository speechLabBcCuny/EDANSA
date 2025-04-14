"""Tests for the inference.pad_audio function."""

import pytest
import numpy as np
import torch
from edansa import inference

# Test cases derived from inference_test.py
test_pad_audio_samples = [
    (np.array([1, 2, 3]), np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0]), 5, 2),
    (np.array([[1, 2, 3]]), np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0]]), 5, 2),
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0], [4, 5, 6, 0, 0, 0, 0, 0, 0, 0],
               [7, 8, 9, 0, 0, 0, 0, 0, 0, 0]]), 5, 2),
    (np.array([[1, 2, 3], [4, 5, 6]]),
     np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0], [4, 5, 6, 0, 0, 0, 0, 0, 0,
                                                0]]), 5, 2),
    (np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [9, 8, 3, 4, 5, 6, 7, 8, 9,
                                                 10]]),
     np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [9, 8, 3, 4, 5, 6, 7, 8, 9,
                                                 10]]), 5, 2),
]


@pytest.mark.parametrize('data_np, expected_result_np, expected_len, sr', [
    (np.zeros(1000), np.zeros(1000), 1, 1000),
    (np.zeros(1500), np.zeros(2000), 1, 1000),
    (np.ones((2, 1500)), np.pad(np.ones((2, 1500)),
                                ((0, 0), (0, 500))), 1, 1000),
])
def test_pad_audio(data_np, expected_result_np, expected_len, sr):
    """Test pad_audio with default padding value."""
    # Convert input to tensor
    data_tensor = torch.from_numpy(data_np).float()
    # Call the function
    # Use default constant_value=0 by not passing it
    computed_tensor = inference.pad_audio(data_tensor, expected_len, sr)
    # Convert result back to numpy for assertion
    computed_data_np = computed_tensor.numpy()
    assert np.array_equal(computed_data_np, expected_result_np)


def test_pad_audio_constant_value():
    """Test pad_audio with a specified constant value."""
    data_np = np.ones(1500) * 0.1
    # Convert input to tensor
    data_tensor = torch.from_numpy(data_np).float()
    # Call the function with correct argument name
    computed_tensor = inference.pad_audio(
        data_tensor, expected_len=1, sr=1000,
        constant_value=0.5)  # Corrected arg name
    # Expected result (using numpy for comparison reference)
    expected_result_np = np.pad(data_np, (0, 500), constant_values=0.5)
    # Convert result back to numpy for assertion
    computed_data_np = computed_tensor.numpy()
    # Use assert_allclose for floating-point comparison
    np.testing.assert_allclose(computed_data_np, expected_result_np)
