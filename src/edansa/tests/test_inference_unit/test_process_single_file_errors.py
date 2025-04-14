"""Integration tests for error handling within the inference processing flow."""

import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
import logging

from edansa import inference, io as eio, dataimport, clippingutils

# --- Dummy Data (copied from test_process_single_file_logic) --- #
DUMMY_SR = 1000
DUMMY_EXCERPT_LEN = 2
DUMMY_TIMESTAMP_START = pd.Timestamp('2023-01-01 12:00:00')
DUMMY_PREDS_NP = np.random.randn(3, 2).astype(
    np.float32)  # For mocks that need preds

# --- Tests for Error Handling in run_inference_on_dataframe --- #


@patch('edansa.inference.dataimport.Audio')
@patch('edansa.io.IO.save_results_per_file')  # Patch save function
def test_run_inference_error_audio_load(mock_save_results, mock_audio_class,
                                        caplog, tmp_path):
    """Test error handling when audio loading fails."""
    # --- Setup Mocks --- #
    mock_audio_instance = MagicMock()
    mock_load_exception = RuntimeError("Mock load error")
    mock_audio_instance.load_data.side_effect = mock_load_exception
    # Ensure unload_data exists even if load fails
    mock_audio_instance.unload_data = MagicMock()
    mock_audio_class.return_value = mock_audio_instance

    # --- Setup Inputs --- #
    input_root = tmp_path / "input_err_load"
    input_root.mkdir()
    dummy_original_path = input_root / 'error_load.wav'
    dummy_original_path.touch()

    output_folder = tmp_path / "output_err_load"
    # Ensure output folder exists for IO handler init
    output_folder.mkdir()
    mock_file_io = eio.IO(excerpt_len=DUMMY_EXCERPT_LEN,
                          output_folder=str(output_folder))

    config = {
        'sampling_rate': DUMMY_SR,
        'excerpt_length': DUMMY_EXCERPT_LEN,
        'channel_selection_method': 'average',
        'target_taxo': [],
        'code2excell_names': {},
        'device': torch.device('cpu')
    }
    mock_args = MagicMock()
    mock_args.skip_clipping_info = True
    mock_args.embeddings = False

    # --- Execute --- #
    with caplog.at_level(logging.ERROR):
        inference.run_inference_on_dataframe(
            file_paths=[str(dummy_original_path)],
            file_io=mock_file_io,
            config=config,
            model_saved=MagicMock(),  # Not used before load error
            get_data_loader=MagicMock(),  # Not used before load error
            args=mock_args,
            input_data_root=input_root)

    # --- Assertions --- #
    mock_audio_class.assert_called_once_with(str(dummy_original_path))
    mock_audio_instance.load_data.assert_called_once()
    # Assert error logged
    assert f"Failed to load audio for {dummy_original_path.name}" in caplog.text
    assert "Mock load error" in caplog.text
    # Check summary log (might be INFO level, adjust if needed)
    # assert "Finished processing. Success: 0, Errors: 1, Skipped: 0." in caplog.text
    mock_save_results.assert_not_called()
    # Assert unload_data was still called in finally block
    mock_audio_instance.unload_data.assert_called_once()


@patch('edansa.inference.dataimport.Audio')
@patch('edansa.inference._select_inference_channel')
@patch('edansa.io.IO.save_results_per_file')
def test_run_inference_error_dataloader(mock_save_results, mock_select_channel,
                                        mock_audio_class, caplog, tmp_path):
    """Test error handling when dataloader creation fails."""
    # --- Setup Mocks --- #
    mock_audio_instance = MagicMock()
    mock_stereo_data_np = np.random.rand(2, DUMMY_SR * 5).astype(np.float32)
    mock_stereo_data_tensor = torch.from_numpy(mock_stereo_data_np).float()
    mock_audio_instance.load_data.return_value = (mock_stereo_data_tensor,
                                                  DUMMY_SR)
    mock_audio_instance.unload_data = MagicMock()
    mock_audio_class.return_value = mock_audio_instance
    # Assume channel selection works, returns mono tensor
    mock_select_channel.return_value = torch.from_numpy(
        np.mean(mock_stereo_data_np, axis=0))
    # Mock get_data_loader to raise error
    mock_get_data_loader_func = MagicMock(
        side_effect=ValueError("Mock dataloader error"))

    # --- Setup Inputs --- #
    input_root = tmp_path / "input_err_dl"
    input_root.mkdir()
    dummy_original_path = input_root / 'error_dl.wav'
    dummy_original_path.touch()

    output_folder = tmp_path / "output_err_dl"
    output_folder.mkdir()
    mock_file_io = eio.IO(excerpt_len=DUMMY_EXCERPT_LEN,
                          output_folder=str(output_folder))

    config = {
        'sampling_rate': DUMMY_SR,
        'excerpt_length': DUMMY_EXCERPT_LEN,
        'channel_selection_method': 'average',
        'target_taxo': [],
        'code2excell_names': {},
        'device': torch.device('cpu')
    }
    mock_args = MagicMock()
    mock_args.skip_clipping_info = True
    mock_args.embeddings = False

    # --- Execute --- #
    with caplog.at_level(logging.ERROR):
        inference.run_inference_on_dataframe(
            file_paths=[str(dummy_original_path)],
            file_io=mock_file_io,
            config=config,
            model_saved=MagicMock(),
            get_data_loader=mock_get_data_loader_func,  # Pass the failing mock
            args=mock_args,
            input_data_root=input_root)

    # --- Assertions --- #
    mock_audio_class.assert_called_once()
    mock_audio_instance.load_data.assert_called_once()
    mock_select_channel.assert_called_once()
    mock_get_data_loader_func.assert_called_once()
    assert f"Failed to create DataLoader for {dummy_original_path.name}" in caplog.text
    assert "Mock dataloader error" in caplog.text
    # assert "Finished processing. Success: 0, Errors: 1, Skipped: 0." in caplog.text
    mock_save_results.assert_not_called()
    mock_audio_instance.unload_data.assert_called_once()


@patch('edansa.inference.dataimport.Audio')
@patch('edansa.inference._select_inference_channel')
@patch('edansa.inference.single_file_inference')
@patch('edansa.io.IO.save_results_per_file')
def test_run_inference_error_inference(mock_save_results, mock_single_inference,
                                       mock_select_channel, mock_audio_class,
                                       caplog, tmp_path):
    """Test error handling when single_file_inference fails."""
    # --- Setup Mocks --- #
    mock_audio_instance = MagicMock()
    mock_stereo_data_np = np.random.rand(2, DUMMY_SR * 5).astype(np.float32)
    mock_stereo_data_tensor = torch.from_numpy(mock_stereo_data_np).float()
    mock_audio_instance.load_data.return_value = (mock_stereo_data_tensor,
                                                  DUMMY_SR)
    mock_audio_instance.unload_data = MagicMock()
    mock_audio_class.return_value = mock_audio_instance
    mock_select_channel.return_value = torch.from_numpy(
        np.mean(mock_stereo_data_np, axis=0))
    # Mock single_file_inference to raise error
    mock_single_inference.side_effect = RuntimeError("Mock inference error")
    mock_get_data_loader_func = MagicMock(return_value={"predict": [
    ]})  # Simulate valid loader

    # --- Setup Inputs --- #
    input_root = tmp_path / "input_err_inf"
    input_root.mkdir()
    dummy_original_path = input_root / 'error_inf.wav'
    dummy_original_path.touch()

    output_folder = tmp_path / "output_err_inf"
    output_folder.mkdir()
    mock_file_io = eio.IO(excerpt_len=DUMMY_EXCERPT_LEN,
                          output_folder=str(output_folder))

    config = {
        'sampling_rate': DUMMY_SR,
        'excerpt_length': DUMMY_EXCERPT_LEN,
        'channel_selection_method': 'average',
        'target_taxo': [],
        'code2excell_names': {},
        'device': torch.device('cpu')
    }
    mock_args = MagicMock()
    mock_args.skip_clipping_info = True
    mock_args.embeddings = False

    # --- Execute --- #
    with caplog.at_level(logging.ERROR):
        inference.run_inference_on_dataframe(
            file_paths=[str(dummy_original_path)],
            file_io=mock_file_io,
            config=config,
            model_saved=MagicMock(),
            get_data_loader=mock_get_data_loader_func,
            args=mock_args,
            input_data_root=input_root)

    # --- Assertions --- #
    mock_audio_class.assert_called_once()
    mock_audio_instance.load_data.assert_called_once()
    mock_select_channel.assert_called_once()
    mock_get_data_loader_func.assert_called_once()
    mock_single_inference.assert_called_once()  # Inference was called
    assert f"Inference failed for {dummy_original_path.name}" in caplog.text
    assert "Mock inference error" in caplog.text
    # assert "Finished processing. Success: 0, Errors: 1, Skipped: 0." in caplog.text
    mock_save_results.assert_not_called()
    mock_audio_instance.unload_data.assert_called_once()


@patch('edansa.inference.dataimport.Audio')
@patch('edansa.inference.clippingutils.get_clipping_percent_file')
@patch('edansa.inference._select_inference_channel')
@patch('edansa.inference.single_file_inference')
@patch('edansa.io.IO.save_results_per_file')
def test_run_inference_warning_clipping_calc(
        mock_save_results, mock_single_inference, mock_select_channel,
        mock_get_clipping, mock_audio_class, caplog, tmp_path):
    """Test warning log when clipping calculation fails but processing continues."""
    # --- Setup Mocks --- #
    mock_audio_instance = MagicMock()
    mock_stereo_data_np = np.random.rand(2, DUMMY_SR * 5).astype(np.float32)
    mock_stereo_data_tensor = torch.from_numpy(mock_stereo_data_np).float()
    mock_mono_data_tensor = torch.mean(mock_stereo_data_tensor, dim=0)
    mock_audio_instance.load_data.return_value = (mock_stereo_data_tensor,
                                                  DUMMY_SR)
    mock_audio_instance.unload_data = MagicMock()
    mock_audio_class.return_value = mock_audio_instance
    # Mock clipping calculation to raise error
    mock_get_clipping.side_effect = ValueError("Mock clipping calc error")
    # Mock downstream functions to succeed
    # Channel selection will be called with None for clipping, assume it falls back
    mock_select_channel.return_value = mock_mono_data_tensor
    num_excerpts = 2  # Example
    mock_single_inference.return_value = torch.from_numpy(
        DUMMY_PREDS_NP[:num_excerpts, :1]).reshape(num_excerpts, 1)
    mock_get_data_loader_func = MagicMock(return_value={"predict": []})
    mock_save_results.return_value = None  # Mock successful save

    # --- Setup Inputs --- #
    input_root = tmp_path / "input_warn_clip"
    input_root.mkdir()
    dummy_original_path = input_root / 'warn_clip.wav'
    dummy_original_path.touch()

    output_folder = tmp_path / "output_warn_clip"
    output_folder.mkdir()
    mock_file_io = eio.IO(excerpt_len=DUMMY_EXCERPT_LEN,
                          output_folder=str(output_folder))

    config = {
        'sampling_rate': DUMMY_SR,
        'excerpt_length': DUMMY_EXCERPT_LEN,
        'channel_selection_method': 'clipping',  # Method that uses clipping
        'target_taxo': ['A'],
        'code2excell_names': {
            'A': 'A'
        },
        'device': torch.device('cpu')
    }
    mock_args = MagicMock()
    mock_args.skip_clipping_info = False  # Clipping IS attempted
    mock_args.embeddings = False

    # --- Execute --- #
    with caplog.at_level(logging.WARNING):  # Capture warnings
        inference.run_inference_on_dataframe(
            file_paths=[str(dummy_original_path)],
            file_io=mock_file_io,
            config=config,
            model_saved=MagicMock(),
            get_data_loader=mock_get_data_loader_func,
            args=mock_args,
            input_data_root=input_root)

    # --- Assertions --- #
    mock_audio_class.assert_called_once()
    mock_audio_instance.load_data.assert_called_once()
    mock_get_clipping.assert_called_once()
    # Assert warning logged for clipping failure
    assert f"Could not calculate clipping for {dummy_original_path.name}" in caplog.text
    assert "Mock clipping calc error" in caplog.text

    # Check channel selection was called (with None for clipping data)
    mock_select_channel.assert_called_once()
    assert mock_select_channel.call_args[0][2] is None  # Clipping tensor is None

    # Check downstream functions were still called
    mock_get_data_loader_func.assert_called_once()
    mock_single_inference.assert_called_once()
    mock_save_results.assert_called_once()  # Results should still be saved

    # Check saved data structure minimally
    saved_results = mock_save_results.call_args[0][0]
    assert isinstance(saved_results, pd.DataFrame)
    assert 'pred_A' in saved_results.columns
    assert 'clipping' in saved_results.columns
    assert saved_results['clipping'].isna().all()  # Clipping is NaN
    mock_audio_instance.unload_data.assert_called_once()


@patch('edansa.inference.dataimport.Audio')
@patch('edansa.inference.clippingutils.get_clipping_percent_file')
@patch('edansa.inference.single_file_inference')
@patch('edansa.inference._select_inference_channel')
@patch('edansa.io.IO.save_results_per_file')
def test_run_inference_error_save(mock_save_results, mock_single_inference,
                                  mock_select_channel, mock_get_clipping,
                                  mock_audio_class, caplog, tmp_path):
    """Test error handling when saving the results fails."""
    # --- Setup Mocks --- #
    mock_audio_instance = MagicMock()
    # Mock successful processing up to saving
    mock_stereo_data_np = np.random.rand(2, DUMMY_SR * 5).astype(
        np.float32)  # 5 seconds duration
    mock_stereo_data_tensor = torch.from_numpy(mock_stereo_data_np).float()
    mock_mono_data_tensor = torch.mean(mock_stereo_data_tensor, dim=0)
    mock_audio_instance.load_data.return_value = (mock_stereo_data_tensor,
                                                  DUMMY_SR)
    mock_audio_instance.unload_data = MagicMock()
    mock_audio_class.return_value = mock_audio_instance
    num_excerpts = int(
        5 / DUMMY_EXCERPT_LEN)  # Calculate expected excerpts (should be 2)
    mock_get_clipping.return_value = torch.rand(num_excerpts,
                                                2)  # Clipping for 2 excerpts
    mock_select_channel.return_value = mock_mono_data_tensor
    # Mock inference result needs exactly num_excerpts rows and 1 column (for target_taxo=['cA'])
    mock_single_inference.return_value = torch.from_numpy(
        DUMMY_PREDS_NP[:num_excerpts, :1]).reshape(num_excerpts, 1)
    mock_get_data_loader_func = MagicMock(return_value={"predict": []})
    # Mock save to raise error
    mock_save_results.side_effect = IOError("Disk full!")

    # --- Setup Inputs --- #
    input_root = tmp_path / "input_err_save"
    input_root.mkdir()
    dummy_original_path = input_root / 'error_save.wav'
    dummy_original_path.touch()

    output_folder = tmp_path / "output_err_save"
    output_folder.mkdir()
    mock_file_io = eio.IO(excerpt_len=DUMMY_EXCERPT_LEN,
                          output_folder=str(output_folder))

    config = {
        'sampling_rate': DUMMY_SR,
        'excerpt_length': DUMMY_EXCERPT_LEN,
        'channel_selection_method': 'clipping',
        # Simplify config to avoid formatting issues unrelated to save error
        'target_taxo': ['cA'],
        'code2excell_names': {
            'cA': 'A'
        },
        'device': torch.device('cpu')
    }
    mock_args = MagicMock()
    mock_args.skip_clipping_info = False
    mock_args.embeddings = False

    # --- Execute --- #
    with caplog.at_level(logging.ERROR):
        inference.run_inference_on_dataframe(
            file_paths=[str(dummy_original_path)],
            file_io=mock_file_io,
            config=config,
            model_saved=MagicMock(),
            get_data_loader=mock_get_data_loader_func,
            args=mock_args,
            input_data_root=input_root)

    # --- Assertions --- #
    # Check all processing steps were called
    mock_audio_class.assert_called_once()
    mock_audio_instance.load_data.assert_called_once()
    mock_get_clipping.assert_called_once()
    mock_select_channel.assert_called_once()
    mock_get_data_loader_func.assert_called_once()
    mock_single_inference.assert_called_once()
    # Saving was attempted
    mock_save_results.assert_called_once()
    # Assert error logged
    assert f"ERROR saving results for {dummy_original_path.name}" in caplog.text
    assert "Disk full!" in caplog.text
    # assert "Finished processing. Success: 0, Errors: 1, Skipped: 0." in caplog.text
    mock_audio_instance.unload_data.assert_called_once()
