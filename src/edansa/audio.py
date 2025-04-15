""" Audio related functions. """

import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional
import torch
import torchaudio
import logging

# Added logger definition
logger = logging.getLogger(__name__)


def load(
    filepath: Union[Path, str],
    dtype: Union[np.dtype, torch.dtype] = torch.float32,  # type: ignore
    resample_rate: int = -1,
    backend: Optional[str] = None,
    mono=False,
    normalize: bool = True,
    channels_first: bool = True,
    audio_format: Optional[str] = None,
) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    """Load audio file as numpy/torch array using torchaudio backend.
    

    Args:
        filepath: path to the file
        dtype: Data type to store audio file.
        backend: Which torchaudio backend to use. If None, torchaudio selects the backend.
                 Valid options: "ffmpeg", "sox", "soundfile". Default: None.
        mono: If True, convert to mono.
        normalize: If True, normalize the audio. [-1,1]
        channels_first: If True, return [channel, time], else [time, channel].
        audio_format: Optional format hint for torchaudio.load.

    Returns:
        A tuple of array storing audio and sampling rate.

    """

    allowed_np_dtypes = {np.int16, np.float32, np.float64}
    allowed_torch_dtypes = {torch.float32, torch.float64}
    allowed_dtypes = allowed_np_dtypes.union(allowed_torch_dtypes)
    np_to_torch_map = {
        np.int16: torch.int16,
        np.float32: torch.float32,
        np.float64: torch.float64,
    }

    if dtype not in allowed_dtypes:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Allowed types are: {allowed_dtypes}")

    if dtype in allowed_np_dtypes:
        torch_dtype = np_to_torch_map[dtype]  # type: ignore
    else:  # dtype is already an allowed torch.dtype
        torch_dtype = dtype

    # Add assertion to help linter confirm the type
    assert isinstance(
        torch_dtype,
        torch.dtype), f"Expected torch.dtype, got {type(torch_dtype)}"

    sound_array, sr = load_audio_torch(
        str(filepath),
        dtype=torch_dtype,
        resample_rate=resample_rate,
        mono=mono,
        backend=backend,  # Pass backend here
        normalize=normalize,
        channels_first=channels_first,
        audio_format=audio_format)
    if dtype != torch_dtype:
        sound_array = sound_array.numpy()

    return sound_array, sr


def load_audio_torch(
        filepath: Union[Path, str],
        dtype: Optional[torch.dtype] = None,  # Corrected type hint
        resample_rate: int = -1,
        mono=False,
        backend: Optional[str] = None,
        normalize: bool = True,
        channels_first: bool = True,
        audio_format: Optional[str] = None) -> Tuple[torch.Tensor, int]:
    """Load audio file as numpy array using torch backend.

    Depending on audio reading library handles data type conversions.

    # https://pytorch.org/audio/stable/_modules/torchaudio/backend/soundfile_backend.html#load
    
    Args:
        filepath: path to the file
        dtype: Data type to store audio file.
        mono: If True, convert to mono.
        backend: Torchaudio backend to use ("ffmpeg", "sox", "soundfile", None).
        normalize: If True, normalize the audio. [-1,1]
        channels_first: If True, return [channel, time], else [time, channel].
        audio_format: Optional format hint for torchaudio.load.

    Returns:
        A tuple of array storing audio and sampling rate.

    """

    filepath = str(filepath)
    try:
        # Note: Default backend (None) usually handles wav/flac well.
        # User can specify "ffmpeg", "sox", or "soundfile" if needed.
        waveform, sample_rate = torchaudio.load(
            str(filepath),
            normalize=normalize,
            channels_first=channels_first,
            format=audio_format,
            backend=backend)  # Pass backend to torchaudio.load
        logger.debug(
            f"Loaded {filepath} with backend='{backend or 'default'}', SR={sample_rate}, Shape={waveform.shape}"
        )
    except Exception as e:
        logger.error(f"Could not load audio file {filepath}: {e}")
        raise

    if resample_rate != -1:
        if normalize is False:
            raise ValueError("normalize should be True when resampling audio")
        waveform = torchaudio.functional.resample(waveform, sample_rate,
                                                  resample_rate)
        sample_rate = resample_rate
    if mono and waveform.shape[0] > 1:
        dim = 0 if channels_first else -1
        waveform = torch.mean(waveform, dim=dim, keepdim=True)

    if dtype is not None:
        waveform = waveform.type(dtype)

    return waveform, sample_rate
