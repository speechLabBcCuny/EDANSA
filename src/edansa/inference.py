"""Inference functions for the edansa package."""

import argparse
from pathlib import Path
import random
import time
import glob
import os
from typing import Union, Tuple, Dict, List, Callable, Optional
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import logging
from torch.nn.functional import pad  # type: ignore

from edansa import io as eio
from edansa import dataimport, runutils
from edansa import clippingutils


def _select_inference_channel(stereo_data: torch.Tensor,
                              channel_selection_method: str,
                              clipping_per_segment_tensor: Union[torch.Tensor,
                                                                 None],
                              file_path_for_log: str, sr: int,
                              excerpt_len: int) -> torch.Tensor:
    """Selects or creates mono audio data based on the specified method.

    Handles fallback to 'average' if 'clipping' is requested but data is missing/invalid.
    Clipping method reconstructs mono by picking the best channel per segment.

    Args:
        stereo_data: Input audio data, stereo (C, S). Must be a torch.Tensor.
        channel_selection_method: 'clipping', 'average', or 'channel_N'.
        clipping_per_segment_tensor: Torch tensor of clipping per segment per channel,
                                   shape (num_segments, num_channels). Should be on the same device as stereo_data.
        file_path_for_log: The file path string, used for logging warnings/errors.
        sr: Sampling rate of the audio data. Required for 'clipping' method.
        excerpt_len: Length of segments in seconds. Required for 'clipping' method.


    Returns:
        Mono audio data as a torch tensor (S,).

    Raises:
        ValueError: If channel selection fails (e.g., invalid method name, invalid channel index, missing sr/excerpt_len for clipping).
        TypeError: If input data type is unsupported.
    """
    default_method = 'average'  # Fallback method if clipping fails

    # --- Input Validation and Mono Check --- #
    if not isinstance(stereo_data, torch.Tensor):
        raise TypeError(
            f"Unsupported data type for channel selection: {type(stereo_data)} for {file_path_for_log}. Expected torch.Tensor."
        )

    if stereo_data.ndim == 1:
        return stereo_data  # Already mono
    if stereo_data.ndim != 2:
        raise ValueError(
            f"Unsupported tensor dimension: {stereo_data.ndim} for {file_path_for_log}"
        )
    num_channels, num_samples = stereo_data.shape
    if num_channels == 1:
        return stereo_data.squeeze(0)  # Already effectively mono

    # --- Channel Selection Logic --- #
    selected_method = channel_selection_method

    if selected_method == 'clipping':
        # Check if necessary args are present for clipping method
        if sr is None or excerpt_len is None or sr <= 0 or excerpt_len <= 0:
            raise ValueError(
                f"Sampling rate ({sr}) and excerpt length ({excerpt_len}) must be provided and positive for 'clipping' method in {file_path_for_log}."
            )

        clipping_available = (clipping_per_segment_tensor is not None and
                              clipping_per_segment_tensor.ndim == 2 and
                              clipping_per_segment_tensor.shape[1]
                              == num_channels)

        if clipping_available:
            try:
                excerpt_sample_size = int(excerpt_len * sr)
                num_segments = clipping_per_segment_tensor.shape[0]

                # Verify number of segments matches audio length (approximately)
                expected_num_segments = (num_samples + excerpt_sample_size -
                                         1) // excerpt_sample_size
                if num_segments != expected_num_segments:
                    # Log warning but proceed if shapes are reasonable
                    logging.warning(
                        f"Clipping segment count ({num_segments}) mismatch for audio length ({num_samples} samples, expected {expected_num_segments} segments) in {file_path_for_log}. "
                        f"Ensure clipping was calculated with excerpt_len={excerpt_len}, sr={sr}. Proceeding anyway."
                    )
                    # Adjust num_segments to match audio length if drastically different? Or just use the smaller one?
                    # Let's use the number of segments from the clipping tensor for now.
                    # num_segments = min(num_segments, expected_num_segments) # Option to reconcile

                # Find best channel index for each segment
                best_channel_indices = torch.argmin(
                    clipping_per_segment_tensor,
                    dim=1)  # Shape: (num_segments,)

                # --- Vectorized Mono Reconstruction ---
                # Create sample indices (0 to num_samples - 1)
                sample_indices = torch.arange(num_samples,
                                              device=stereo_data.device)
                # Determine the segment index for each sample
                segment_indices = sample_indices // excerpt_sample_size
                # Clamp segment indices to ensure they are within the bounds of best_channel_indices
                # This handles cases where audio length might imply more segments than available in clipping data
                segment_indices = torch.clamp(segment_indices,
                                              max=num_segments - 1)
                # Get the best channel index corresponding to each sample's segment
                best_channels_per_sample = best_channel_indices[
                    segment_indices]  # Shape: (num_samples,)
                # Select data from stereo tensor using the best channel index for each sample
                # and the sample's own index along the time dimension.
                mono_data = stereo_data[best_channels_per_sample,
                                        sample_indices]
                # --- End Vectorized Reconstruction ---

                logging.info(
                    f"  Reconstructed mono using best channel per segment (clipping) for {file_path_for_log} (vectorized)"
                )
                return mono_data

            except Exception as e:
                logging.warning(
                    f"Error during segment-wise clipping-based channel selection for {file_path_for_log}: {e}. Falling back to '{default_method}'."
                )
                selected_method = default_method  # Fallback on error
        else:
            # Log warning for missing/invalid clipping data
            if clipping_per_segment_tensor is None:
                logging.warning(
                    f"Clipping method requested for file '{file_path_for_log}' but no clipping data provided. "
                    f"Falling back to '{default_method}' method.")
            else:  # Clipping data provided, but shape is wrong
                logging.warning(
                    f"Clipping method requested for file '{file_path_for_log}' but clipping data dimensions mismatch audio channels "
                    f"(clipping shape: {clipping_per_segment_tensor.shape}, audio channels: {num_channels}). Falling back to '{default_method}' method."
                )
            selected_method = default_method  # Change method for subsequent checks

    if selected_method == 'average':
        print(f"  Averaging channels for {file_path_for_log}")
        # Use torch.mean
        return torch.mean(stereo_data.float(), dim=0)  # Ensure float for mean

    elif selected_method.startswith('channel_'):
        try:
            channel_idx = int(selected_method.split('_')[-1])
            if not 0 <= channel_idx < num_channels:
                raise ValueError(
                    f"Invalid channel index {channel_idx} for {num_channels} channels"
                )
            print(
                f"  Selected channel {channel_idx} manually for {file_path_for_log}"
            )
            return stereo_data[channel_idx, :]
        except (ValueError, IndexError) as e:
            # Reraise with file path context
            raise ValueError(
                f"Invalid channel format '{selected_method}' for {file_path_for_log}: {e}"
            )

    # If we reach here, the original or fallback method was unknown
    raise ValueError(
        f"Unknown channel selection method: '{channel_selection_method}' (or fallback '{selected_method}') for {file_path_for_log}"
    )


def pad_audio(data: torch.Tensor,
              expected_len: int,
              sr: int,
              constant_value: Union[float, int] = 0) -> torch.Tensor:
    """Pads audio tensor with a constant value if its length is not a multiple
    of the expected length in samples.

    Args:
        data (torch.Tensor): Input audio tensor, shape (S,) or (C, S).
        expected_len (int): Expected segment length in seconds.
        sr (int): Sampling rate.
        constant_value (Union[float, int]): Value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: Padded audio tensor, on the same device as input.

    Raises:
        TypeError: If input data is not a torch.Tensor.
        ValueError: If input tensor dimensions are not 1 or 2.
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Input data must be a torch.Tensor, got {type(data)}.")

    target_samples = expected_len * sr
    current_samples = data.shape[-1]
    left_over = current_samples % target_samples

    if left_over == 0:
        return data

    missing_element_count = target_samples - left_over

    # F.pad expects padding in the format (pad_left, pad_right, pad_top, pad_bottom, ...)
    # For audio (last dim is time):
    # 1D (S,): (0, missing_element_count)
    # 2D (C, S): (0, missing_element_count, 0, 0) -> pad last dim right, no pad for channel dim
    if data.ndim == 1:
        padding = (0, missing_element_count)
    elif data.ndim == 2:
        padding = (0, missing_element_count, 0, 0)
    else:
        raise ValueError(
            f"Input tensor must be 1D or 2D, got {data.ndim} dimensions.")

    # Pad using the specified constant value
    padded_data = pad(  # pylint: disable=E1102:not-callable
        data,
        padding,
        mode='constant',
        value=constant_value)

    return padded_data


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def single_file_inference(
    dataloader,
    config,
    model_saved,
) -> torch.Tensor:

    preds = []
    device = config['device']
    model_saved.eval()  # Ensure model is in eval mode
    with torch.no_grad():  # Disable gradient calculations for inference
        for inputs, labels in dataloader['predict']:
            del labels  # Still don't need labels
            inputs = inputs.float().to(device)  # Move input to device
            output = model_saved(inputs)
            preds.append(output)  # Append the tensor directly
    if not preds:
        # Handle case where dataloader was empty
        # Depending on expected downstream shape, return empty tensor or raise error
        logging.warning(
            "No predictions generated in single_file_inference, dataloader might be empty."
        )
        # Returning an empty tensor on the correct device, shape might need adjustment
        # Example: return torch.empty((0, model_saved.fc_audioset.out_features), device=device) # If shape is known
        return torch.empty((0, 0), device=device)  # Placeholder empty tensor

    preds_tensor = torch.cat(preds,
                             dim=0)  # Concatenate tensors along batch dimension
    return preds_tensor  # Return tensor on the original device


def replace_fc_layer(model_saved):
    model_saved.fc_audioset = torch.nn.Identity()
    for param in model_saved.parameters():
        param.requires_grad = False
    return model_saved


def create_arg_parser():
    """Creates the command-line argument parser for the inference script."""
    parser = argparse.ArgumentParser(
        description="Run inference using a trained EDANSA model.")

    # --- Required Inputs & Model --- #
    parser.add_argument(
        '--model_path',
        help='Path to the trained model checkpoint file (.pth).',
        required=True,
        type=str,
    )
    parser.add_argument(
        '-c',
        '--config_file',
        help='Path to the JSON configuration file associated with the model.',
        required=True,
        type=str,
    )

    # --- Input Source (Mutually Exclusive) --- #
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_files_list',
        help=
        'Path to a text file listing audio files to process (one path per line).',
        # required=True, # Now handled by the mutually exclusive group
        type=str,
    )
    input_group.add_argument(
        '--input_folder',
        help=
        'Path to a folder containing audio files to process (recursive search).',
        type=str,
    )

    # --- Output Control --- #
    parser.add_argument(
        '-O',
        '--output_folder',
        help=
        'Directory to save prediction or embedding files. (Default: ./outputs)',
        default=None,  # Default handled later if None
        type=str,
    )
    parser.add_argument(
        '--embeddings',
        help=
        'Generate and save embeddings instead of class predictions. (Flag, default: False)',
        action='store_true',
    )

    # --- Audio Processing Options --- #
    parser.add_argument(
        '--channel_selection_method',
        help=
        "Method for handling multi-channel audio: 'average' (default), 'clipping', or 'channel_N'.",
        default='average',  # Explicit default
        type=str,
    )
    parser.add_argument(
        '--skip_clipping_info',
        help=
        'Skip calculating and including clipping percentage in results. (Flag, default: False)',
        action='store_true',
    )

    # --- Optional Processing & Output --- Add device here
    parser.add_argument(
        '--device',
        help=
        "Device to use for computation (e.g., 'cpu', 'cuda', 'cuda:0'). Defaults to cuda if available, otherwise cpu.",
        type=str,
        default=None  # Default handled in _setup_device_and_config
    )

    # --- Add force overwrite flag (default is False) --- #
    parser.add_argument(
        '--force_overwrite',
        help=
        'Force processing of all files, even if their corresponding output file already exists. If not set, existing outputs will be skipped.',
        action='store_true',  # Default is False
    )

    # args = parser.parse_args() # Removed: Parsing happens in the calling script now

    # Handle default output folder path creation after parsing
    # if args.output_folder is None:
    #     output_path = Path("outputs")
    #     args.output_folder = str(output_path.resolve())
    #     logging.info(
    #         f"Output folder not specified, using default: {args.output_folder}")

    return parser  # Return the parser object


#%% Helper function to parse filenames
def parse_start_time_from_filename(filename: str) -> Optional[datetime]:
    """
    Attempts to parse the start date and time from a filename based on common PAM formats.

    Expected formats (case-insensitive for extension):
    1. {recorderid}_YYYYMMDD_HHMMSS.*
    2. YYYYMMDD_HHMMSS.*
    3. _{recorderid}_YYYYMMDD_HHMMSS.* (Handles potential leading underscore before ID)
    4. _YYYYMMDD_HHMMSS.* (Handles potential leading underscore before date)


    Args:
        filename: The filename string (e.g., "REC001_20230101_120000.wav").

    Returns:
        A datetime object if parsing is successful, otherwise None.
    """
    # Remove extension for easier matching
    stem = Path(filename).stem

    # Pattern 1 & 3: Optional leading underscore, recorder ID, date, time
    # Allows ID like 'REC001' or '.REC001' if underscore is present
    pattern1 = re.compile(r"^_?(.+)_(\d{8})_(\d{6})$")
    match1 = pattern1.match(stem)
    if match1:
        recorder_id, date_str, time_str = match1.groups()
        datetime_str = date_str + time_str
        try:
            # Validate date/time components before returning
            parsed_dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
            logging.debug(
                f"Parsed datetime '{parsed_dt}' from filename '{filename}' using pattern 1/3."
            )
            return parsed_dt
        except ValueError:
            logging.warning(
                f"Invalid date/time string '{datetime_str}' found in filename '{filename}'."
            )
            return None  # Invalid date/time components

    # Pattern 2 & 4: Optional leading underscore, date, time (no recorder ID part)
    pattern2 = re.compile(r"^_?(\d{8})_(\d{6})$")
    match2 = pattern2.match(stem)
    if match2:
        date_str, time_str = match2.groups()
        datetime_str = date_str + time_str
        try:
            parsed_dt = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
            logging.debug(
                f"Parsed datetime '{parsed_dt}' from filename '{filename}' using pattern 2/4."
            )
            return parsed_dt
        except ValueError:
            logging.warning(
                f"Invalid date/time string '{datetime_str}' found in filename '{filename}'."
            )
            return None  # Invalid date/time components

    logging.debug(
        f"Could not parse datetime from filename '{filename}' using known patterns."
    )
    return None


#%% Core functions


def _process_single_audio_file(
    audio_file_path: Path,
    config: Dict,
    model_saved: torch.nn.Module,
    get_data_loader: Callable,
    args: argparse.Namespace,
) -> Tuple[Union[pd.DataFrame, Dict[str, Union[Path, np.ndarray]], None], str]:
    """
    Processes a single audio file: loads, preprocesses, infers, and formats results.

    Args:
        audio_file_path: Path object for the audio file.
        config: Configuration dictionary.
        model_saved: Loaded model.
        get_data_loader: Function to create a DataLoader.
        args: Command-line arguments.

    Returns:
        A tuple containing:
        - results: Either a pandas DataFrame (for predictions) or a dictionary
                   (for embeddings) containing the inference results and relevant metadata.
                   Returns None if processing fails at any critical step.
        - result_type: A string indicating the type of result ('predictions' or 'embeddings').
                       Returns an empty string if processing fails.
    """
    audio_ins = None
    device = config['device']
    excerpt_len = config['excerpt_length']
    channel_selection_method = config.get('channel_selection_method',
                                          'clipping')
    clipping_threshold = 1.0  # Standard threshold
    result_type = 'embeddings' if args.embeddings else 'predictions'

    try:
        audio_ins = dataimport.Audio(str(audio_file_path))

        # --- Load Stereo Audio as Tensor --- #
        try:
            stereo_data, sr = audio_ins.load_data(
                mono=False,
                resample_rate=config['sampling_rate'],
                dtype=torch.float32,
                store=False)
            if sr is None:
                raise ValueError("Sample rate could not be determined.")
            if not isinstance(stereo_data, torch.Tensor):
                logging.warning(
                    f"load_data did not return a tensor for {audio_file_path.name}, type was {type(stereo_data)}. Converting."
                )
                stereo_data = torch.from_numpy(stereo_data).float()

            stereo_data = stereo_data.to(device)

        except Exception as load_err:
            logging.error(
                f"Failed to load audio for {audio_file_path.name}: {load_err}")
            return None, ""

        # --- Calculate Clipping (if needed) --- #
        clipping_per_segment_tensor = None
        mean_clipping_per_excerpt_tensor = None
        if not args.skip_clipping_info:
            try:
                clipping_per_segment_tensor = clippingutils.get_clipping_percent_file(
                    stereo_data, sr, excerpt_len, clipping_threshold).to(device)

                if clipping_per_segment_tensor.numel() > 0:
                    if clipping_per_segment_tensor.ndim > 1:
                        mean_clipping_per_excerpt_tensor = torch.mean(
                            clipping_per_segment_tensor.float(), dim=1)
                    elif clipping_per_segment_tensor.ndim == 1:
                        mean_clipping_per_excerpt_tensor = clipping_per_segment_tensor.float(
                        )
                    else:
                        mean_clipping_per_excerpt_tensor = torch.empty(
                            0, device=device)

            except Exception as clip_err:
                logging.warning(
                    f"Could not calculate clipping for {audio_file_path.name}: {clip_err}. Proceeding without clipping info."
                )
                clipping_per_segment_tensor = None
                mean_clipping_per_excerpt_tensor = None

        # --- Select Inference Channel --- #
        try:
            mono_data_for_inference = _select_inference_channel(
                stereo_data,
                channel_selection_method, clipping_per_segment_tensor,
                str(audio_file_path), sr, excerpt_len)
        except ValueError as chan_sel_err:
            logging.error(
                f"Channel selection failed for {audio_file_path.name}: {chan_sel_err}. Skipping file."
            )
            return None, ""

        # --- Prepare DataLoader --- #
        try:
            dataloader = get_data_loader(mono_data_for_inference, config)
        except Exception as dl_err:
            logging.error(
                f"Failed to create DataLoader for {audio_file_path.name}: {dl_err}"
            )
            return None, ""

        # --- Run Inference --- #
        try:
            preds_tensor = single_file_inference(dataloader, config,
                                                 model_saved)
        except Exception as infer_err:
            logging.error(
                f"Inference failed for {audio_file_path.name}: {infer_err}")
            return None, ""

        # --- Format Results --- #
        try:
            num_excerpts = preds_tensor.shape[0]
            if num_excerpts == 0:
                logging.warning(
                    f"No excerpts generated for {audio_file_path.name}. Skipping file."
                )
                return None, ""

            if result_type == 'embeddings':
                # For embeddings, return the tensor and the path
                # Extract embedding for the first excerpt
                if num_excerpts > 1:
                    logging.warning(
                        f"Multiple ({num_excerpts}) embeddings generated for {audio_file_path.name}, using only the first one."
                    )
                first_embedding_np = preds_tensor[0].cpu().numpy(
                )  # Get first row, move to CPU, convert to numpy
                embeddings_dict = {
                    'audio_file_path': audio_file_path,
                    'embeds': first_embedding_np
                }
                return embeddings_dict, result_type

            elif result_type == 'predictions':
                target_taxo = config['target_taxo']
                target_taxo_names = [
                    config['code2excell_names'][x] for x in target_taxo
                ]
                pred_col_names = ['pred_' + name for name in target_taxo_names]

                # --- Create Timestamp Index (Handle missing/invalid start_date_time) ---
                # Priority: 1. Filename parsing, 2. Metadata 'start_date_time', 3. Relative index
                timestamps_pd = None
                start_pd_timestamp = None  # Initialize

                # 1. Attempt to parse from filename first
                try:
                    filename_dt = parse_start_time_from_filename(
                        audio_file_path.name)
                    if filename_dt:
                        start_pd_timestamp = pd.Timestamp(filename_dt)
                        logging.debug(
                            f"Using start time from filename for {audio_file_path.name}: {start_pd_timestamp}"
                        )
                except Exception as fn_parse_err:
                    logging.warning(
                        f"Error parsing filename {audio_file_path.name} for timestamp: {fn_parse_err}"
                    )

                # 2. If filename parsing failed or didn't yield a result, try metadata
                # (Metadata fallback is now effectively removed as file_row is gone)
                if start_pd_timestamp is None:
                    pass  # Keep the outer 'if start_pd_timestamp is None:' structure

                # 3. Generate index based on whether we found an absolute start time
                if start_pd_timestamp:
                    try:
                        timestamps_pd = pd.date_range(
                            start=start_pd_timestamp,
                            periods=num_excerpts,
                            freq=pd.Timedelta(seconds=excerpt_len))
                        logging.debug(
                            f"Using absolute timestamps for {audio_file_path.name}"
                        )
                    except (ValueError, TypeError,
                            pd.errors.OutOfBoundsDatetime) as ts_err:
                        logging.warning(
                            f"Error creating date range for {audio_file_path.name} even with parsed start time {start_pd_timestamp}: {ts_err}. Falling back to relative time index."
                        )
                        timestamps_pd = None  # Fallback to relative index

                # 4. If absolute timestamp generation failed or no start time found, use relative index
                if timestamps_pd is None:
                    if start_pd_timestamp is None:  # Log only if we never found a start time
                        logging.warning(
                            f"Could not determine absolute start time for {audio_file_path.name} from filename or metadata. Using relative time index (float seconds from start)."
                        )
                    # Create an index of float seconds from the start
                    relative_seconds = np.arange(num_excerpts) * excerpt_len
                    timestamps_pd = pd.Index(relative_seconds, dtype='float64')

                # --- End Timestamp Index Creation ---

                results_data = {}
                preds_sig_tensor = sigmoid(preds_tensor)
                preds_sig_np = preds_sig_tensor.cpu().numpy()

                # Ensure preds_sig_np is 2D for consistent indexing, even with 1 class
                if preds_sig_np.ndim == 1:
                    preds_sig_np = preds_sig_np.reshape(-1, 1)

                for i, p_col in enumerate(pred_col_names):
                    results_data[p_col] = preds_sig_np[:, i]

                if mean_clipping_per_excerpt_tensor is not None:
                    mean_clipping_np = mean_clipping_per_excerpt_tensor.cpu(
                    ).numpy()
                    if len(mean_clipping_np) == num_excerpts:
                        results_data['clipping'] = mean_clipping_np
                    else:
                        logging.warning(
                            f"Clipping excerpt count ({len(mean_clipping_np)}) mismatch prediction excerpt count ({num_excerpts}) for {audio_file_path.name}. Padding clipping with NaN."
                        )
                        results_data['clipping'] = np.full(num_excerpts, np.nan)
                else:
                    results_data['clipping'] = np.full(num_excerpts, np.nan)

                results_df = pd.DataFrame(results_data, index=timestamps_pd)
                results_df.index.name = eio.TIMESTAMP_ARRAY_KEY
                return results_df, result_type

        except Exception as format_err:
            logging.error(
                f"Failed to format results for {audio_file_path.name}: {format_err}"
            )
            return None, ""

    except Exception as outer_err:
        logging.exception(
            f"Unexpected ERROR processing {audio_file_path.name}: {outer_err}")
        return None, ""
    finally:
        if audio_ins:
            audio_ins.unload_data()


def run_inference_on_dataframe(
        file_paths: List[Union[str, Path]],  # Allow Path objects now
        file_io: eio.IO,
        config: Dict,
        model_saved: torch.nn.Module,
        get_data_loader: Callable,
        args: argparse.Namespace,
        input_data_root: Path,  # Add root path for relative path calculation
):
    """
    Runs inference for each file specified in the list and saves results individually.

    Args:
        file_paths: List of audio file path strings or Path objects.
        file_io: IO handler instance for saving results.
        config: Configuration dictionary.
        model_saved: Loaded model.
        get_data_loader: Function to create a DataLoader.
        args: Command-line arguments.
        input_data_root: The root directory from which relative paths for output should be derived.
    """
    logging.info(f"Processing {len(file_paths)} files...")
    processed_count = 0
    error_count = 0
    skipped_count = 0  # Add counter for skipped files

    if args.embeddings:
        model_saved = replace_fc_layer(
            model_saved)  # Prepare model for embeddings

    for file_path in file_paths:  # Iterate over list (str or Path)
        # Ensure it's a Path object for consistency within the loop
        audio_file_path = Path(file_path)
        logging.debug(
            f"Considering: {audio_file_path.name}")  # Log file being considered

        # --- Check if output file exists (Skip if it exists AND force_overwrite is False) --- #
        if not args.force_overwrite:  # Only check if we are NOT forcing overwrite
            try:
                result_type_for_check = 'embeddings' if args.embeddings else 'predictions'
                # --- Use the IO method to get the expected path --- #
                expected_output_path = file_io.get_expected_output_path(
                    result_type=result_type_for_check,
                    audio_file_path=audio_file_path.resolve(
                    ),  # Ensure absolute path
                    input_data_root=input_data_root.resolve(
                    )  # Ensure absolute path
                )
                # --- End Use the IO method --- #
                if expected_output_path.exists():
                    logging.info(
                        f"Skipping {audio_file_path.name} as output {expected_output_path} already exists."
                    )
                    continue
            except Exception as check_err:
                logging.warning(
                    f"Error checking for existing output for {audio_file_path.name}, proceeding with inference: {check_err}"
                )

        # --- Process the file if not skipped ---
        logging.debug(
            f"Processing: {audio_file_path.name}")  # Log file being processed

        results, result_type = _process_single_audio_file(
            audio_file_path, config, model_saved, get_data_loader, args)

        if results is not None and result_type:
            try:
                # Call the new save function in file_io
                file_io.save_results_per_file(results, result_type,
                                              audio_file_path, input_data_root)
                processed_count += 1
            except Exception as save_err:
                logging.error(
                    f"ERROR saving results for {audio_file_path.name}: {save_err}"
                )
                error_count += 1
        else:
            # Error occurred during processing, already logged in _process_single_audio_file
            error_count += 1

    logging.info(
        f"Finished processing. Success: {processed_count}, Errors: {error_count}, Skipped: {skipped_count}."
    )


def _setup_device_and_config(args: argparse.Namespace, config: Dict,
                             model_saved: torch.nn.Module) -> torch.device:
    """Configures torch device based on user input or availability, and moves model to device."""

    if args.device:
        # User specified a device
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            logging.warning(
                f"User specified device '{args.device}' but CUDA is not available. Falling back to CPU."
            )
            device_str = 'cpu'
        elif args.device == 'cpu':
            device_str = 'cpu'
        elif args.device.startswith('cuda'):
            # Validate specific cuda device index if provided e.g. cuda:0
            try:
                # Basic validation: Check if it's 'cuda' or 'cuda:N' where N is int
                if ':' in args.device:
                    gpu_index = int(args.device.split(':')[1])
                    if gpu_index >= torch.cuda.device_count():
                        logging.warning(
                            f"CUDA device index {gpu_index} out of range ({torch.cuda.device_count()} available). Using default CUDA device."
                        )
                        device_str = 'cuda'  # Fallback to default cuda
                    else:
                        device_str = args.device  # Use specified cuda:N
                else:
                    device_str = 'cuda'  # Use default cuda device
            except (ValueError, IndexError):
                logging.warning(
                    f"Invalid CUDA device format '{args.device}'. Using default CUDA device."
                )
                device_str = 'cuda'  # Fallback
        else:
            logging.warning(
                f"Unrecognized device '{args.device}'. Falling back to default logic (CUDA if available, else CPU)."
            )
            # Fallback to default logic below
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        # Default logic: Use CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            device_str = 'cuda'
        else:
            device_str = 'cpu'

    device = torch.device(device_str)
    config['device'] = device
    logging.info(f"Using device: {device}")
    model_saved.to(device)
    return device


def _read_input_file_list(input_list_path_str: str) -> List[str]:
    """Reads file paths from the specified input list file."""
    if not input_list_path_str:
        # This case should ideally be caught by argparse 'required=True'
        logging.error(
            "--input_files_list argument is required but was not provided.")
        raise ValueError("--input_files_list argument is required.")

    try:
        with open(input_list_path_str, 'r') as f:
            # Read paths, stripping whitespace and ignoring empty lines
            file_paths = [line.strip() for line in f if line.strip()]
        if not file_paths:
            logging.warning(
                f"Input file list '{input_list_path_str}' is empty. No files to process."
            )
        else:
            logging.info(
                f"Read {len(file_paths)} files for processing from '{input_list_path_str}'."
            )
        return file_paths
    except FileNotFoundError:
        logging.error(f"Input file list '{input_list_path_str}' not found.")
        raise  # Reraise the FileNotFoundError


def _find_audio_files(input_folder_path_str: str) -> List[Path]:
    """Recursively finds audio files in the specified folder."""
    input_folder = Path(input_folder_path_str)
    if not input_folder.is_dir():
        logging.error(
            f"Input folder '{input_folder_path_str}' not found or is not a directory."
        )
        raise FileNotFoundError(
            f"Input folder '{input_folder_path_str}' not found or is not a directory."
        )

    supported_extensions = [".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff"]
    audio_files = []
    logging.info(f"Searching for audio files in '{input_folder_path_str}'...")
    for ext in supported_extensions:
        # Search case-insensitively
        audio_files.extend(input_folder.rglob(f"*[.{ext[1:].lower()}]"))
        audio_files.extend(input_folder.rglob(f"*[.{ext[1:].upper()}]"))

    # Remove duplicates that might arise from case-insensitive globbing
    audio_files = sorted(list(set(audio_files)))

    if not audio_files:
        logging.warning(f"No audio files found in '{input_folder_path_str}'.")
    else:
        logging.info(
            f"Found {len(audio_files)} audio files in '{input_folder_path_str}'."
        )
    return audio_files


def _determine_input_root(
        file_paths: List[Union[str, Path]],  # Allow Path objects
        input_list_path_str: Optional[str] = None,  # Made optional
        input_folder_path_str: Optional[str] = None) -> Path:
    """Determines the root directory for input files based on their common path."""
    if not file_paths:
        logging.warning(
            "Input file list was empty or no files found in folder. Using input source directory as root."  # Updated msg
        )
        # Determine root based on which input method was likely used
        if input_folder_path_str:
            return Path(input_folder_path_str)
        elif input_list_path_str:
            return Path(input_list_path_str).parent
        else:
            # Should not happen if group is required, but fallback just in case
            return Path(".").resolve()

    try:
        path_objects = [Path(p) for p in file_paths]
        # Find common ancestor path
        common_path = Path(os.path.commonpath(path_objects))
        # Check if common path is a file (it shouldn't be, but handle defensively)
        if common_path.is_file():
            input_data_root = common_path.parent
        else:
            input_data_root = common_path
        logging.info(
            f"Using input data root: {input_data_root} (derived from common path of input file list)"
        )
        return input_data_root

    except ValueError as e:
        # Can happen if paths are on different drives on Windows or other issues
        logging.warning(
            f"Could not determine common path for input files (Reason: {e}). Falling back to input source directory: %s",
            input_folder_path_str)
        return Path(input_folder_path_str)
    except Exception as e:
        # Determine fallback directory based on input method
        if input_folder_path_str:
            fallback_dir = Path(input_folder_path_str)
        elif input_list_path_str:
            fallback_dir = Path(input_list_path_str).parent
        else:
            fallback_dir = Path(".").resolve()

        # Catch any other unexpected errors during path processing
        logging.error(
            f"Unexpected error determining common path: {e}. Falling back to input source directory: %s",
            fallback_dir)
        return fallback_dir


def main(args: argparse.Namespace, setup_fnc: Callable,
         get_data_loader: Callable):
    """Main entry point for the inference script."""
    # 1. Setup: Load model, config, and IO handler
    # Let setup_fnc handle initial arg parsing related to model/config loading if needed
    model_saved, config, file_io = setup_fnc(args)

    # 2. Configure Device (GPU/CPU)
    _setup_device_and_config(args, config,
                             model_saved)  # config is updated in-place

    # 3. Read Input File List OR Find Files in Folder
    file_paths: List[Union[str, Path]] = []
    input_data_root: Optional[Path] = None  # Initialize

    if args.input_folder:
        try:
            file_paths = _find_audio_files(args.input_folder)
            # Convert Path objects to strings for consistency if needed downstream,
            # but let's keep them as Paths for now if possible.
            # file_paths = [str(p) for p in file_paths]
            input_data_root = Path(args.input_folder).resolve()
            logging.info(
                f"Using input data root: {input_data_root} (from --input_folder)"
            )
        except FileNotFoundError:
            # Error already logged by _find_audio_files, exit or handle
            logging.error("Exiting due to input folder issue.")
            return  # Or sys.exit(1)
        except Exception as e:
            logging.exception(
                f"Error finding files in folder {args.input_folder}: {e}")
            return  # Or sys.exit(1)

    elif args.input_files_list:
        try:
            # file_paths will be an empty list if the file is empty or doesn't exist (error raised before)
            file_paths = _read_input_file_list(args.input_files_list)
            # We proceed even if file_paths is empty, as _determine_input_root handles it.
            input_data_root = _determine_input_root(file_paths,
                                                    args.input_files_list, None)
        except FileNotFoundError:
            # Error logged by _read_input_file_list
            logging.error("Exiting due to input file list issue.")
            return
        except Exception as e:
            logging.exception(
                f"Error processing input file list {args.input_files_list}: {e}"
            )
            return

    # 4. Determine Input Root Directory ( Handled above for both cases )
    # input_data_root = _determine_input_root(file_paths, args.input_files_list)
    if input_data_root is None:
        logging.error("Could not determine input data root. Exiting.")
        return  # Or sys.exit(1)

    # 5. Run Inference (only if there are files to process)
    if file_paths:
        logging.info(f"Starting inference for {len(file_paths)} files.")
        run_inference_on_dataframe(
            file_paths=file_paths,  # Pass the list directly
            file_io=file_io,
            config=config,
            model_saved=model_saved,
            get_data_loader=get_data_loader,
            args=args,
            input_data_root=input_data_root  # Pass the determined root
        )
    else:
        # Log if no files were processed (either list was empty or read error occurred earlier)
        logging.warning(
            "No audio files found in the input list or list could not be read. Skipping inference run."
        )


# Ensure any top-level script execution logic remains outside main if needed,
# or is called within an `if __name__ == "__main__":` block.
