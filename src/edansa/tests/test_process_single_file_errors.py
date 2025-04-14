from unittest.mock import patch
import logging
import numpy as np


@patch('edansa.inference.single_file_inference')
@patch('edansa.io.IO.save_results_per_file')
def test_run_inference_error_save(
    mock_save_results,
    mock_single_inference,
    caplog,
    tmp_path,
):
    """Test run_inference when save_results_per_file fails."""
    caplog.set_level(logging.ERROR)
    # mock_path.exists.return_value = True # Remove this line
    dummy_original_path = tmp_path / "error_save.wav"
    dummy_original_path.touch()
    # mock_path.is_file.return_value = True # Also remove this line
    # Mock a successful inference result (e.g., 5 excerpts, 1 class)
    num_excerpts = 5
    mock_single_inference.return_value = (
        np.random.rand(num_excerpts, 1),  # Make sure it's 2D
        np.random.rand(num_excerpts),
        np.array([f'ts_{i:02d}' for i in range(num_excerpts)]),
    )
    # Mock save_results_per_file to raise an exception
    mock_save_results.side_effect = IOError("Disk full")
