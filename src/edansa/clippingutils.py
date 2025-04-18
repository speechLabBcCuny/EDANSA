"""Utilities for clipping detection in audio.

'Clipping is a form of waveform distortion that occurs when an amplifier is
overdriven and attempts to deliver an output voltage or current beyond its
maximum capability.'(wikipedia)
Clipped samples of the audio signal does not carry any information.
We assume clipping happens when sample's value is +1 or -1 (threshold).


    Typical usage    
        Function run_task_save combines loading audio and get_clipping_percent.

    1)example by using list of files :
    ```[python]
    from nna import clippingutils
    test_area_files = ['./data/sound_examples/10minutes.mp3', 
                        './data/sound_examples/10seconds.mp3']

    all_results_dict, files_w_errors = clippingutils.run_task_save(
        test_area_files, "test_area", "./output_folder_path", 1.0)
    ```

    2) example by using a file_properties file :
    ```[python]
    from nna import clippingutils
    # file info
    import pandas as pd
    clipping_threshold=1.0
    file_properties_df_path = "../nna/data/prudhoeAndAnwr4photoExp_dataV1.pkl"
    file_properties_df = pd.read_pickle(file_properties_df_path)
    # where to save results
    clipping_results_path="./clipping_results/"
    location_ids=['11','12']
    for i,location_id in enumerate(location_ids):
        print(location_id,i)
        location_id_filtered=file_properties_df[file_properties_df.locationId==location_id]
        all_results_dict, files_w_errors = clippingutils.run_task_save(
                                                    location_id_filtered.index,
                                                    location_id,clipping_results_path,
                                                    clipping_threshold)
    ```
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Union
import traceback
import warnings
import logging

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None  # Handle case where torch is not installed

from edansa import audio
from edansa import io as eio

FILE_PATH_COL = 'file_path'


# file_properties_df_path = '../../data/prudhoeAndAnwr4photoExp_dataV1.pkl'
def load_metadata(metadata_file, version='v1'):
    version = version.lower()
    if not metadata_file or metadata_file == '.':
        return pd.DataFrame()
    if version == 'v1':
        return mio.AudioRecordingFiles(dataframe=metadata_file).files
    elif version == 'v2':
        return mio.AudioRecordingFiles(dataframe=metadata_file)
    else:
        raise ValueError(f'unknown version {version}')


def get_clipping_percent(sound_array: Union[np.ndarray, torch.Tensor],
                         threshold: float = 1.0) -> torch.Tensor:
    """Calculate clipping percentage comparing to (>=) threshold using PyTorch.

        Args:
            sound_array: a numpy array or torch tensor with shape of
                        (sample_count) or (..., sample_count)
            threshold: min and max values which samples are assumed to be
                      Clipped. 0 <= threshold <= 1.

        Returns:
            torch.Tensor: A tensor containing clipping percentages (float).
                          The shape matches the input tensor shape excluding the 
                          last (samples) dimension. E.g., input (B, C, S) -> output (B, C).
                          Tensor will be on the same device as the input tensor.
    """
    if torch is None:
        raise ImportError("PyTorch is required for this function.")

    # Convert numpy array to torch tensor if necessary
    if isinstance(sound_array, np.ndarray):
        try:
            sound_array_torch = torch.from_numpy(sound_array)
        except TypeError as e:
            raise TypeError(
                f"Could not convert numpy array to tensor: {e}") from e
    elif isinstance(sound_array, torch.Tensor):
        sound_array_torch = sound_array
    else:
        raise TypeError(
            "Input sound_array must be a numpy array or torch tensor.")

    original_device = sound_array_torch.device

    # --- Handle ALL empty tensor cases first ---
    if sound_array_torch.ndim == 0:
        # Treat 0-dim tensor as a single channel with 0% clipping -> return 0D tensor
        return torch.tensor(0.0, device=original_device)

    num_samples = sound_array_torch.shape[-1]
    if num_samples == 0:
        # For empty samples dim, return zeros with the shape of leading dimensions
        leading_dims_shape = sound_array_torch.shape[:-1]
        return torch.zeros(leading_dims_shape,
                           dtype=torch.float32,
                           device=original_device)

    # --- Proceed with calculation only if there are samples (num_samples > 0) ---
    threshold = abs(threshold)  # Ensure threshold is positive

    # Determine minval/maxval based on dtype
    # Note: Checks are performed on the original tensor device
    if torch.is_floating_point(sound_array_torch):
        # Check for warning by calculating max on original device
        # .item() brings the scalar result to CPU
        max_abs_val = torch.max(torch.abs(sound_array_torch)).item()  # Safe now
        if max_abs_val > 1.0 + 5e-2:
            warnings.warn(
                f"Input audio data is float type but has values outside the "
                f"typical normalized range [-1.0, 1.0] (max abs value: {max_abs_val:.4f}). "
                f"Clipping calculation assumes normalization; results may be inaccurate.",
                UserWarning)
        # Assume normalized float audio data (typically [-1.0, 1.0])
        minval = -threshold
        # Apply slight reduction to max threshold for floats near 1.0
        maxval = threshold * 0.9999
    elif sound_array_torch.dtype in (torch.int8, torch.int16, torch.int32,
                                     torch.int64):
        # Assume integer audio data
        info = torch.iinfo(sound_array_torch.dtype)  # Safe now
        # Scale the integer limits by the threshold
        # Cast info limits to float for multiplication, then back to int
        minval_f = info.min * threshold
        maxval_f = info.max * threshold

        # Ensure scaled values are within the original integer type limits
        minval = max(int(minval_f), info.min)
        maxval = min(int(maxval_f), info.max)
    else:
        raise TypeError(
            f"Unsupported audio dtype for clipping detection: {sound_array_torch.dtype}"
        )

    # --- Perform calculation ---
    # Threshold tensors need to be on the same device as the input tensor
    minval_tensor = torch.tensor(minval,
                                 dtype=sound_array_torch.dtype,
                                 device=original_device)
    maxval_tensor = torch.tensor(maxval,
                                 dtype=sound_array_torch.dtype,
                                 device=original_device)

    # Create boolean mask for clipped samples
    is_clipped = (sound_array_torch <= minval_tensor) | (sound_array_torch
                                                         >= maxval_tensor)

    # Sum the boolean mask over the sample dimension (last dim)
    clipped_count = torch.sum(is_clipped, dim=-1)

    # Calculate percentage
    results_tensor = clipped_count.float() / num_samples

    # Return the results tensor (shape matches input leading dims)
    return results_tensor


def run_task_save(
    input_files: List[Union[str, Path]],
    area_id: str,
    results_folder: Union[str, Path, None],
    clipping_threshold: float,
    segment_len: int = 10,
    save=True,
) -> Tuple[dict, list]:
    """Save clipping in dict to a file named as f"{area_id}_{threshold}.pkl"

        Computes clipping for only files
         that does not exist in the results pkl file if results_folder is provided.

        Args:
            input_files: List of files to calculate clipping.
            area_id: ID of the area/dataset, used in potential cache file name.
            results_folder: Where to save/load cached results. If None, no caching is done.
            clipping_threshold: Threshold for detecting clipping.
            segment_len: Length of segments to calculate clipping per.
            save: Explicit flag to control saving (if results_folder is provided).
        Returns:
            Tuple(all_results_dict ,files_w_errors)
                all_results_dict: Dict{a_file_path:np.array}
                files_w_errors: List[(index, a_file_path, exception),]
    """
    output_file_path = None
    error_file_path = None
    filename = None
    can_cache = results_folder is not None

    if can_cache:
        output_file_path, error_file_path, filename = get_output_file_path(
            results_folder, area_id, clipping_threshold)

    if can_cache and output_file_path is not None and output_file_path.exists():
        # Attempt to load from cache
        all_results_dict, file2process = load_previous_results(
            results_folder, filename, output_file_path, input_files)
    else:
        # No cache found or caching disabled, process all files
        file2process = input_files
        all_results_dict = {}

    files_w_errors = []
    # CALCULATE RESULTS
    for file_index, audio_file in enumerate(file2process):
        # Ensure audio_file is a string or Path
        audio_file_path = Path(audio_file)
        try:
            # Call edansa.audio.load directly, maintaining previous defaults
            y, sr = audio.load(audio_file_path)

            assert sr == int(sr)
            sr = int(sr)
            # Get result as tensor
            result_tensor = get_clipping_percent_file(y, sr, segment_len,
                                                      clipping_threshold)
            # Convert to numpy for saving
            all_results_dict[str(audio_file_path)] = result_tensor.cpu().numpy()

            # Save periodically only if caching is enabled and save is True
            if can_cache and save and output_file_path and error_file_path and file_index % 100 == 0:
                save2disk(all_results_dict, files_w_errors, output_file_path,
                          error_file_path)

        except Exception as e:
            # Use logging for errors
            logging.error(f'Error processing {audio_file_path}: {e}')
            # Optionally log traceback for debugging
            # logging.exception("Traceback for error processing file:")
            files_w_errors.append((str(audio_file_path), e))

    # Final save only if caching is enabled and save is True
    if can_cache and save and output_file_path and error_file_path:
        save2disk(all_results_dict, files_w_errors, output_file_path,
                  error_file_path)

    return all_results_dict, files_w_errors


def get_output_file_path(results_folder, area_id, clipping_threshold):
    clipping_threshold_str = str(clipping_threshold)
    clipping_threshold_str = clipping_threshold_str.replace('.', ',')
    filename = f'{area_id}_{clipping_threshold_str}.pkl'
    error_filename = f'{area_id}_{clipping_threshold_str}_error.pkl'
    if results_folder:
        results_folder = Path(results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)
        output_file_path = results_folder / filename
        error_file_path = results_folder / error_filename
    else:
        output_file_path = Path('.') / filename
        error_file_path = Path('.') / error_filename
    return output_file_path, error_file_path, filename


def load_previous_results(results_folder, filename, output_file_path,
                          input_files):
    input_files = [str(i) for i in input_files]
    # Use logging for cache info
    logging.info(
        f'Clipping file for {filename} exists at {results_folder}. Checking existing results.'
    )
    prev_results_dict = np.load(str(output_file_path), allow_pickle=True)
    prev_results_dict = dict(prev_results_dict[()])
    prev_results_keys = {str(i) for i in prev_results_dict.keys()}
    input_result_keys = set(input_files)
    new_result_keys = input_result_keys.difference(prev_results_keys)
    if len(new_result_keys) > 0:
        # Use logging
        logging.info(
            f'{len(new_result_keys)} files missing results, calculating only those.'
        )
    else:
        # Use logging
        logging.info('No new file from existing results, will exit.')
        return prev_results_dict, []
    file2process = new_result_keys
    all_results_dict = prev_results_dict

    return all_results_dict, file2process


def get_clipping_percent_file(audio_array: Union[np.ndarray, torch.Tensor],
                              sr: int, segment_len: int,
                              clipping_threshold: float) -> torch.Tensor:
    """Calculate clipping percentage for segments using vectorized PyTorch operations.
    
    Divides the audio into segments of `segment_len` seconds. If the last segment
    is shorter, it pads it with zeros to `segment_len` before calculating clipping.
    Uses padding similar to the prediction data pipeline.

    Args:
        audio_array: A numpy array or torch tensor containing audio data.
                     Assumes shape (..., samples).
        sr: Sample rate of the audio.
        segment_len: Length of each segment in seconds.
        clipping_threshold: Threshold for clipping detection.
        
    Returns:
        torch.Tensor: Tensor of clipping percentages. Shape will be 
                      (num_segments, ...) where ... is the shape of the 
                      input tensor's leading dimensions.
                      Result tensor is on the same device as the input tensor.
    """
    if torch is None:
        raise ImportError("PyTorch is required for this function.")

    # 1. Ensure input is a torch.Tensor on the correct device
    if isinstance(audio_array, np.ndarray):
        try:
            # Assume numpy arrays are CPU, result tensor will also be CPU
            sound_array_torch = torch.from_numpy(audio_array)
        except TypeError as e:
            raise TypeError(
                f"Could not convert numpy array to tensor: {e}") from e
    elif isinstance(audio_array, torch.Tensor):
        sound_array_torch = audio_array
    else:
        raise TypeError(
            "Input audio_array must be a numpy array or torch tensor.")

    original_device = sound_array_torch.device
    original_leading_dims = sound_array_torch.shape[:-1]
    num_samples_total = sound_array_torch.shape[-1]

    # 2. Calculate segment info and handle edge cases
    segment_samples = int(segment_len * sr)
    if segment_samples <= 0:
        warnings.warn(
            f"Segment length ({segment_len}s) results in non-positive samples ({segment_samples}) at sample rate {sr}. Returning empty tensor.",
            UserWarning)
        # Return empty tensor with shape (0, ...) matching leading dims
        return torch.empty((0,) + original_leading_dims,
                           dtype=torch.float32,
                           device=original_device)

    if num_samples_total < segment_samples:
        warnings.warn(
            "Audio array is shorter than segment length. Returning empty tensor.",
            UserWarning)
        # Return empty tensor with shape (0, ...) matching leading dims
        return torch.empty((0,) + original_leading_dims,
                           dtype=torch.float32,
                           device=original_device)

    # --- MODIFIED: 3. Pad audio instead of truncating --- #
    left_over = num_samples_total % segment_samples
    padded_tensor = sound_array_torch  # Assume no padding needed initially
    if left_over != 0:
        missing_element_count = segment_samples - left_over
        # Create padding tensor on the same device
        pad_shape = original_leading_dims + (missing_element_count,)
        padding = torch.zeros(pad_shape,
                              dtype=sound_array_torch.dtype,
                              device=original_device)
        padded_tensor = torch.cat([sound_array_torch, padding], dim=-1)

    # Now calculate number of segments based on padded length
    num_segments = padded_tensor.shape[-1] // segment_samples
    # total_samples_in_segments = num_segments * segment_samples # No longer needed

    # Original truncation code REMOVED:
    # num_segments = num_samples_total // segment_samples
    # total_samples_in_segments = num_segments * segment_samples
    # truncated_tensor = torch.narrow(sound_array_torch, ...)

    # 4. Reshape the PADDED tensor for segmentation view
    # New shape: original_leading_dims + (num_segments, segment_samples)
    reshaped_shape = original_leading_dims + (num_segments, segment_samples)
    # Use contiguous() before view for safety if padded_tensor might be non-contiguous
    segmented_view = padded_tensor.contiguous().view(reshaped_shape)

    # 5. Call get_clipping_percent ONCE on the segmented view
    # Result shape: original_leading_dims + (num_segments,)
    results_tensor_permuted = get_clipping_percent(segmented_view,
                                                   threshold=clipping_threshold)

    # 6. Permute dimensions to (num_segments, ...original_leading_dims...)
    num_dims_result = results_tensor_permuted.ndim
    if num_dims_result > 0:  # Check if the result is not a 0D tensor (original
        #  input was 1D)
        segment_dim_index = num_dims_result - 1  # Index of the segment dimension
        # Create permutation order: (segment_dim_index, 0, 1, ..., segment_dim_index - 1)
        permute_order = (segment_dim_index,) + tuple(range(segment_dim_index))
        results_tensor = results_tensor_permuted.permute(permute_order)
    else:  # Original input was 1D (S,), result is 0D. This shouldn't happen with current get_clipping_percent.
        # If input was (S,), segmented_view is (N, L), result is (N,). num_dims_result=1.
        # If num_dims_result is 1, segment_dim_index=0, permute_order=(0,), no change. Correct.
        results_tensor = results_tensor_permuted  # Should be shape (N,)

    return results_tensor


def save2disk(all_results_dict, files_w_errors, output_file_path,
              error_file_path):
    with open(output_file_path, 'wb') as f:
        np.save(f, all_results_dict)  # type: ignore
    if files_w_errors:
        with open(error_file_path, 'wb') as f:
            pickle.dump(files_w_errors, f, protocol=pickle.HIGHEST_PROTOCOL)
