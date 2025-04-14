""" Audio related functions. """

import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional
import torch
import torchaudio


def load(
    filepath: Union[Path, str],
    dtype: Union[np.dtype, torch.dtype] = torch.float32,  # type: ignore
    resample_rate: int = -1,
    backend: str = "torch_soxio",
    mono=False,
    normalize: bool = True,
    channels_first: bool = True,
    audio_format: Optional[str] = None,
) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    """Load audio file as numpy/torch array using torchaudio backend.
    

    Args:
        filepath: path to the file
        dtype: Data type to store audio file.
        backend: Which backend to use load the audio: torch_soxio, 
                                                     torch_soundfile

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

    if dtype is None:
        dtype = torch.float32

    if dtype not in allowed_dtypes:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Allowed types are: {allowed_dtypes}")

    if dtype in allowed_np_dtypes:
        torch_dtype = np_to_torch_map[dtype]
    else:  # dtype is already an allowed torch.dtype
        torch_dtype = dtype

    sound_array, sr = load_audio_torch(str(filepath),
                                       dtype=torch_dtype,
                                       resample_rate=resample_rate,
                                       mono=mono,
                                       backend=backend,
                                       normalize=normalize,
                                       channels_first=channels_first,
                                       audio_format=audio_format)
    if dtype != torch_dtype:
        sound_array = sound_array.numpy()

    return sound_array, sr


def load_audio_torch(
        filepath: Union[Path, str],
        dtype: torch.dtype = None,  # type: ignore
        resample_rate: int = -1,
        mono=False,
        backend=None,
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
        backend: Which backend to use load the audio.
        normalize: If True, normalize the audio. [-1,1]

    Returns:
        A tuple of array storing audio and sampling rate.

    """

    filepath = str(filepath)
    if backend == "torch_soxio" or backend == "sox_io":
        torchaudio.set_audio_backend("sox_io")
    elif backend == "torch_soundfile" or backend == "soundfile":
        torchaudio.set_audio_backend("soundfile")
    else:
        raise ValueError(
            f"no backend called {backend} for torchaudio options " +
            "are torch_soxio, torch_soundfile")

    sound_array, sr = torchaudio.load(  # type: ignore
        filepath,
        normalize=normalize,
        channels_first=channels_first,
        format=audio_format)
    if resample_rate != -1:
        if normalize is False:
            raise ValueError("normalize should be True when resampling audio")
        sound_array = torchaudio.functional.resample(sound_array, sr,
                                                     resample_rate)
        sr = resample_rate
    if mono and sound_array.shape[0] > 1:
        dim = 0 if channels_first else -1
        sound_array = torch.mean(sound_array, dim=dim, keepdim=True)

    if dtype is not None:
        sound_array = sound_array.type(dtype)

    return sound_array, sr
