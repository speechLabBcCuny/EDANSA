"""Integration tests for the inference processing logic (using mocks)."""

import pytest
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

from edansa import inference, io as eio, dataimport, clippingutils

# --- Dummy Data (from inference_test.py) --- #
DUMMY_SR = 1000
DUMMY_EXCERPT_LEN = 2  # seconds
DUMMY_EXCERPT_SAMPLES = DUMMY_SR * DUMMY_EXCERPT_LEN

# Example: 3 excerpts long
DUMMY_STEREO_DATA_NP = np.random.randn(2, DUMMY_EXCERPT_SAMPLES * 3).astype(
    np.float32)
DUMMY_MONO_DATA_NP = np.mean(DUMMY_STEREO_DATA_NP, axis=0)
# Example: 3 excerpts, 2 classes (logits)
DUMMY_PREDS_NP = np.random.randn(3, 2).astype(np.float32)
# Example: 3 excerpts, 2 channels
DUMMY_CLIPPING_TENSOR = torch.rand(3, 2, dtype=torch.float32)
DUMMY_CLIPPING_NP = DUMMY_CLIPPING_TENSOR.numpy()
DUMMY_MEAN_CLIPPING_NP = np.mean(DUMMY_CLIPPING_NP, axis=1)

DUMMY_TIMESTAMP_START = pd.Timestamp('2023-01-01 12:00:00')
DUMMY_TIMESTAMPS = pd.to_datetime([
    DUMMY_TIMESTAMP_START + pd.Timedelta(seconds=i * DUMMY_EXCERPT_LEN)
    for i in range(3)
])


# --- Mock Functions (from inference_test.py) --- #
def mock_get_data_loader(mono_data, config):
    """Minimal mock: returns a dummy dataloader suitable for single_file_inference loop."""
    if isinstance(mono_data, np.ndarray):
        mono_data_tensor = torch.from_numpy(mono_data)
    else:
        mono_data_tensor = mono_data

    # Simulate processing into excerpts for the loader (example)
    # This part might need adjustment based on how the actual get_data_loader
    # interacts with single_file_inference
    excerpt_samples = config['excerpt_length'] * config['sampling_rate']
    num_excerpts = mono_data_tensor.shape[0] // excerpt_samples
    reshaped_data = mono_data_tensor[:num_excerpts * excerpt_samples].reshape(
        num_excerpts, excerpt_samples)

    # Simple list of batched tensors (batch size 1)
    # Each item in the list represents a batch yielded by the dataloader
    dataset = [(reshaped_data[i:i + 1].float().to(config.get('device',
                                                             'cpu')), None)
               for i in range(num_excerpts)]  # Use None for label
    dataloader = {'predict': dataset}
    return dataloader


# --- Tests --- #


@patch('edansa.inference.dataimport.Audio')
@patch('edansa.inference.clippingutils.get_clipping_percent_file')
@patch('edansa.inference.single_file_inference')
@patch('edansa.inference._select_inference_channel')
@patch('edansa.io.IO.save_results_per_file')
def test_run_inference_export_flow(mock_save_results_per_file,
                                   mock_select_channel, mock_single_inference,
                                   mock_get_clipping, mock_audio_class,
                                   tmp_path):
    """Tests the overall flow of run_inference_on_dataframe (predictions, clipping)."""
    # --- Setup Mocks --- #
    mock_audio_instance = MagicMock()
    mock_stereo_data_tensor = torch.from_numpy(DUMMY_STEREO_DATA_NP).float()
    mock_audio_instance.load_data.return_value = (mock_stereo_data_tensor,
                                                  DUMMY_SR)
    mock_audio_class.return_value = mock_audio_instance
    mock_get_clipping.return_value = DUMMY_CLIPPING_TENSOR  # Clipping calculated
    mock_select_channel.return_value = torch.from_numpy(
        DUMMY_MONO_DATA_NP).float()  # Assume clipping method returns this
    mock_single_inference.return_value = torch.from_numpy(
        DUMMY_PREDS_NP)  # Raw predictions

    # --- Setup Inputs --- #
    input_root = tmp_path / "input_audio"
    input_root.mkdir()
    dummy_filename = "DUMMYREC_20230101_120000.wav"  # Parsable filename
    dummy_original_path = input_root / dummy_filename
    dummy_original_path.touch()

    output_folder = tmp_path / "output"
    mock_file_io = eio.IO(excerpt_len=DUMMY_EXCERPT_LEN,
                          output_folder=str(output_folder))

    config = {
        'target_taxo': ['classA', 'classB'],
        'code2excell_names': {
            'classA': 'Class A Name',
            'classB': 'Class B Name'
        },
        'excerpt_length': DUMMY_EXCERPT_LEN,
        'sampling_rate': DUMMY_SR,
        'channel_selection_method': 'clipping',
        'device': torch.device('cpu')
    }
    mock_args = MagicMock()
    mock_args.skip_clipping_info = False
    mock_args.embeddings = False

    # --- Execute Function --- #
    inference.run_inference_on_dataframe(
        file_paths=[str(dummy_original_path)],  # Pass as list
        file_io=mock_file_io,
        config=config,
        model_saved=MagicMock(),  # Model itself is mocked by single_inference
        get_data_loader=mock_get_data_loader,
        args=mock_args,
        input_data_root=input_root)

    # --- Assertions --- #
    mock_audio_class.assert_called_once_with(str(dummy_original_path))
    mock_audio_instance.load_data.assert_called_once_with(
        mono=False, resample_rate=DUMMY_SR, dtype=torch.float32, store=False)
    # Verify clipping calculation was called with the stereo tensor
    mock_get_clipping.assert_called_once()
    assert torch.equal(mock_get_clipping.call_args[0][0],
                       mock_stereo_data_tensor.cpu())  # Arg should be tensor
    assert mock_get_clipping.call_args[0][1] == DUMMY_SR
    assert mock_get_clipping.call_args[0][2] == DUMMY_EXCERPT_LEN

    # Verify channel selection was called with stereo tensor and clipping tensor
    mock_select_channel.assert_called_once()
    assert torch.equal(mock_select_channel.call_args[0][0],
                       mock_stereo_data_tensor)  # Arg 1: stereo data
    assert mock_select_channel.call_args[0][1] == 'clipping'  # Arg 2: method
    assert torch.equal(mock_select_channel.call_args[0][2],
                       DUMMY_CLIPPING_TENSOR)  # Arg 3: clipping tensor
    assert mock_select_channel.call_args[0][3] == str(
        dummy_original_path)  # Arg 4: path
    assert mock_select_channel.call_args[0][4] == DUMMY_SR  # Arg 5: sr
    assert mock_select_channel.call_args[0][
        5] == DUMMY_EXCERPT_LEN  # Arg 6: excerpt_len

    mock_single_inference.assert_called_once()
    mock_audio_instance.unload_data.assert_called_once()

    # Verify save call
    mock_save_results_per_file.assert_called_once()
    call_args, _ = mock_save_results_per_file.call_args
    saved_results = call_args[0]
    result_type_arg = call_args[1]
    audio_file_path_arg = call_args[2]
    input_data_root_arg = call_args[3]

    assert result_type_arg == 'predictions'
    assert audio_file_path_arg == dummy_original_path
    assert input_data_root_arg == input_root
    assert isinstance(saved_results, pd.DataFrame)

    # Check DataFrame content
    assert saved_results.shape[0] == 3  # 3 excerpts
    assert 'pred_Class A Name' in saved_results.columns
    assert 'pred_Class B Name' in saved_results.columns
    assert 'clipping' in saved_results.columns
    assert saved_results.index.equals(pd.DatetimeIndex(DUMMY_TIMESTAMPS))

    # Check prediction values (sigmoid applied)
    expected_preds_sig = inference.sigmoid(torch.from_numpy(DUMMY_PREDS_NP))
    np.testing.assert_allclose(saved_results['pred_Class A Name'].values,
                               expected_preds_sig[:, 0].cpu().numpy(),
                               rtol=1e-6)
    np.testing.assert_allclose(saved_results['pred_Class B Name'].values,
                               expected_preds_sig[:, 1].cpu().numpy(),
                               rtol=1e-6)
    # Check clipping values (mean)
    np.testing.assert_allclose(saved_results['clipping'].values,
                               DUMMY_MEAN_CLIPPING_NP,
                               rtol=1e-6)


@patch('edansa.inference.dataimport.Audio')
@patch('edansa.inference.clippingutils.get_clipping_percent_file')
@patch('edansa.inference.single_file_inference')
@patch('edansa.inference._select_inference_channel')
@patch('edansa.io.IO.save_results_per_file')
def test_run_inference_skip_clipping(mock_save_results_per_file,
                                     mock_select_channel, mock_single_inference,
                                     mock_get_clipping, mock_audio_class,
                                     tmp_path):
    """Tests the flow when skip_clipping_info is True."""
    # --- Setup Mocks --- #
    mock_audio_instance = MagicMock()
    mock_stereo_data_tensor = torch.from_numpy(DUMMY_STEREO_DATA_NP).float()
    mock_mono_data_tensor = torch.from_numpy(DUMMY_MONO_DATA_NP).float()
    mock_audio_instance.load_data.return_value = (mock_stereo_data_tensor,
                                                  DUMMY_SR)
    mock_audio_class.return_value = mock_audio_instance
    # Channel selection should be called with None for clipping data and return mono
    mock_select_channel.return_value = mock_mono_data_tensor
    mock_single_inference.return_value = torch.from_numpy(DUMMY_PREDS_NP)

    # --- Setup Inputs --- #
    input_root = tmp_path / "input_audio_skip"
    input_root.mkdir()
    dummy_filename = "SKIPREC_20230101_130000.flac"
    dummy_original_path = input_root / dummy_filename
    dummy_original_path.touch()

    output_folder = tmp_path / "output_skip"
    mock_file_io = eio.IO(excerpt_len=DUMMY_EXCERPT_LEN,
                          output_folder=str(output_folder))

    config = {
        'target_taxo': ['classA'],
        'code2excell_names': {
            'classA': 'Class A'
        },
        'excerpt_length': DUMMY_EXCERPT_LEN,
        'sampling_rate': DUMMY_SR,
        'channel_selection_method':
            'average',  # Method doesn't rely on clipping
        'device': torch.device('cpu')
    }
    mock_args = MagicMock()
    mock_args.skip_clipping_info = True  # Key setting for this test
    mock_args.embeddings = False

    # --- Execute Function --- #
    inference.run_inference_on_dataframe(file_paths=[str(dummy_original_path)],
                                         file_io=mock_file_io,
                                         config=config,
                                         model_saved=MagicMock(),
                                         get_data_loader=mock_get_data_loader,
                                         args=mock_args,
                                         input_data_root=input_root)

    # --- Assertions --- #
    mock_audio_class.assert_called_once_with(str(dummy_original_path))
    mock_audio_instance.load_data.assert_called_once()
    # CRITICAL: Clipping calculation should NOT be called
    mock_get_clipping.assert_not_called()
    # Channel selection called with None for clipping data
    mock_select_channel.assert_called_once()
    assert torch.equal(mock_select_channel.call_args[0][0],
                       mock_stereo_data_tensor)
    assert mock_select_channel.call_args[0][1] == 'average'
    assert mock_select_channel.call_args[0][2] is None  # Clipping data is None

    mock_single_inference.assert_called_once()
    mock_audio_instance.unload_data.assert_called_once()

    # Verify save call
    mock_save_results_per_file.assert_called_once()
    call_args, _ = mock_save_results_per_file.call_args
    saved_results = call_args[0]
    result_type_arg = call_args[1]
    audio_file_path_arg = call_args[2]

    assert result_type_arg == 'predictions'
    assert audio_file_path_arg == dummy_original_path
    assert isinstance(saved_results, pd.DataFrame)

    # Check DataFrame content
    assert saved_results.shape[0] == 3
    assert 'pred_Class A' in saved_results.columns
    assert 'clipping' in saved_results.columns
    # CRITICAL: Clipping column should be all NaN
    assert saved_results['clipping'].isna().all()


@patch('edansa.inference.dataimport.Audio')
@patch('edansa.inference.clippingutils.get_clipping_percent_file')
@patch('edansa.inference.single_file_inference')
@patch('edansa.inference._select_inference_channel')
@patch('edansa.io.IO.save_results_per_file')
def test_run_inference_average_channel(mock_save_results_per_file,
                                       mock_select_channel,
                                       mock_single_inference, mock_get_clipping,
                                       mock_audio_class, tmp_path):
    """Tests the flow using 'average' channel selection."""
    # --- Setup Mocks --- #
    mock_audio_instance = MagicMock()
    mock_stereo_data_tensor = torch.from_numpy(DUMMY_STEREO_DATA_NP).float()
    mock_mono_data_tensor = torch.from_numpy(DUMMY_MONO_DATA_NP).float()
    mock_audio_instance.load_data.return_value = (mock_stereo_data_tensor,
                                                  DUMMY_SR)
    mock_audio_class.return_value = mock_audio_instance
    mock_get_clipping.return_value = DUMMY_CLIPPING_TENSOR  # Clipping still calculated by default
    # Channel selection should be called with 'average' and return the averaged mono data
    mock_select_channel.return_value = mock_mono_data_tensor
    mock_single_inference.return_value = torch.from_numpy(DUMMY_PREDS_NP)

    # --- Setup Inputs --- #
    input_root = tmp_path / "input_audio_avg"
    input_root.mkdir()
    dummy_filename = "AVG_REC_20230101_140000.wav"
    dummy_original_path = input_root / dummy_filename
    dummy_original_path.touch()

    output_folder = tmp_path / "output_avg"
    mock_file_io = eio.IO(excerpt_len=DUMMY_EXCERPT_LEN,
                          output_folder=str(output_folder))

    config = {
        'target_taxo': ['classA'],
        'code2excell_names': {
            'classA': 'Class A'
        },
        'excerpt_length': DUMMY_EXCERPT_LEN,
        'sampling_rate': DUMMY_SR,
        'channel_selection_method': 'average',  # Key setting
        'device': torch.device('cpu')
    }
    mock_args = MagicMock()
    mock_args.skip_clipping_info = False  # Clipping IS calculated
    mock_args.embeddings = False

    # --- Execute --- #
    inference.run_inference_on_dataframe(file_paths=[str(dummy_original_path)],
                                         file_io=mock_file_io,
                                         config=config,
                                         model_saved=MagicMock(),
                                         get_data_loader=mock_get_data_loader,
                                         args=mock_args,
                                         input_data_root=input_root)

    # --- Assertions --- #
    mock_audio_class.assert_called_once()
    mock_audio_instance.load_data.assert_called_once()
    mock_get_clipping.assert_called_once()  # Clipping IS called
    # Channel selection called with 'average' method and the calculated clipping tensor
    mock_select_channel.assert_called_once()
    assert torch.equal(mock_select_channel.call_args[0][0],
                       mock_stereo_data_tensor)
    assert mock_select_channel.call_args[0][1] == 'average'
    assert torch.equal(mock_select_channel.call_args[0][2],
                       DUMMY_CLIPPING_TENSOR)

    mock_single_inference.assert_called_once()
    mock_audio_instance.unload_data.assert_called_once()

    # Verify save call
    mock_save_results_per_file.assert_called_once()
    call_args, _ = mock_save_results_per_file.call_args
    saved_results = call_args[0]
    assert isinstance(saved_results, pd.DataFrame)
    assert saved_results.shape[0] == 3
    assert 'pred_Class A' in saved_results.columns
    assert 'clipping' in saved_results.columns
    # Clipping values should be the mean across original channels
    np.testing.assert_allclose(saved_results['clipping'].values,
                               DUMMY_MEAN_CLIPPING_NP,
                               rtol=1e-6)


@patch('edansa.inference.dataimport.Audio')
@patch('edansa.inference.clippingutils.get_clipping_percent_file')
@patch('edansa.inference.single_file_inference')
@patch('edansa.inference._select_inference_channel')
@patch('edansa.io.IO.save_results_per_file')
def test_run_inference_manual_channel(mock_save_results_per_file,
                                      mock_select_channel,
                                      mock_single_inference, mock_get_clipping,
                                      mock_audio_class, tmp_path):
    """Tests the flow using manual 'channel_0' selection (and skipping clipping)."""
    # --- Setup Mocks --- #
    mock_audio_instance = MagicMock()
    mock_stereo_data_tensor = torch.from_numpy(DUMMY_STEREO_DATA_NP).float()
    # Simulate selecting channel 0
    mock_channel0_data_tensor = mock_stereo_data_tensor[0, :].clone()
    mock_audio_instance.load_data.return_value = (mock_stereo_data_tensor,
                                                  DUMMY_SR)
    mock_audio_class.return_value = mock_audio_instance
    # Channel selection should be called with 'channel_0', None for clipping, and return channel 0
    mock_select_channel.return_value = mock_channel0_data_tensor
    mock_single_inference.return_value = torch.from_numpy(DUMMY_PREDS_NP)

    # --- Setup Inputs --- #
    input_root = tmp_path / "input_audio_ch0"
    input_root.mkdir()
    dummy_filename = "CH0_REC_20230101_150000.ogg"
    dummy_original_path = input_root / dummy_filename
    dummy_original_path.touch()

    output_folder = tmp_path / "output_ch0"
    mock_file_io = eio.IO(excerpt_len=DUMMY_EXCERPT_LEN,
                          output_folder=str(output_folder))

    config = {
        'target_taxo': ['classB'],
        'code2excell_names': {
            'classB': 'Class B Name'
        },
        'excerpt_length': DUMMY_EXCERPT_LEN,
        'sampling_rate': DUMMY_SR,
        'channel_selection_method': 'channel_0',  # Key setting
        'device': torch.device('cpu')
    }
    mock_args = MagicMock()
    mock_args.skip_clipping_info = True  # Skip clipping calculation
    mock_args.embeddings = False

    # --- Execute --- #
    inference.run_inference_on_dataframe(file_paths=[str(dummy_original_path)],
                                         file_io=mock_file_io,
                                         config=config,
                                         model_saved=MagicMock(),
                                         get_data_loader=mock_get_data_loader,
                                         args=mock_args,
                                         input_data_root=input_root)

    # --- Assertions --- #
    mock_audio_class.assert_called_once()
    mock_audio_instance.load_data.assert_called_once()
    mock_get_clipping.assert_not_called()  # Clipping skipped
    # Channel selection called with 'channel_0' method and None for clipping
    mock_select_channel.assert_called_once()
    assert torch.equal(mock_select_channel.call_args[0][0],
                       mock_stereo_data_tensor)
    assert mock_select_channel.call_args[0][1] == 'channel_0'
    assert mock_select_channel.call_args[0][2] is None

    mock_single_inference.assert_called_once()
    mock_audio_instance.unload_data.assert_called_once()

    # Verify save call
    mock_save_results_per_file.assert_called_once()
    call_args, _ = mock_save_results_per_file.call_args
    saved_results = call_args[0]
    assert isinstance(saved_results, pd.DataFrame)
    assert saved_results.shape[0] == 3
    assert 'pred_Class B Name' in saved_results.columns
    assert 'clipping' in saved_results.columns
    assert saved_results['clipping'].isna().all(
    )  # Clipping is NaN because it was skipped
