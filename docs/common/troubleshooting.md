# Troubleshooting Guide

This page lists common issues encountered when installing or running the EDANSA inference scripts and suggests potential solutions.

## Installation Issues

### Conda Environment Creation Fails

*   **Symptom:** `conda env create -f environment.yml -n <name>` fails, often with messages about unsatisfiable dependencies or package conflicts.
*   **Possible Causes & Solutions:**
    *   **Outdated Conda/Mamba:** Ensure your Conda or Mamba installation is up-to-date: `conda update conda` or `mamba update mamba`.
    *   **Channel Conflicts:** Sometimes channel priorities can cause issues. Try creating the environment strictly from `conda-forge` (though this might miss specific builds):
        ```bash
        conda create -n <your_env_name> --channel conda-forge --file environment.yml 
        ```
    *   **Network Issues:** Temporary network problems can prevent downloading packages. Try again later.
    *   **Incompatible System Dependencies (GPU):** If creating the GPU environment (`environment.gpu.yml`, if using the older two-file approach) fails, ensure your NVIDIA drivers are compatible with the requested CUDA toolkit version (e.g., 12.1). Update drivers if necessary.

### Pip Installation Fails (PyTorch)

*   **Symptom:** Running the specific `pip install torch ...` command from the PyTorch website fails.
*   **Possible Causes & Solutions:**
    *   **Incorrect Command:** Double-check that you copied the exact command matching your OS, package manager (Pip), and compute platform (CPU/CUDA version) from [pytorch.org](https://pytorch.org/get-started/locally/).
    *   **Unsupported CUDA Version:** Ensure the CUDA version selected on the PyTorch website matches the CUDA toolkit version compatible with your NVIDIA drivers.
    *   **Network Issues:** Try again later.
    *   **Permissions:** Ensure you have permissions to install packages in the target environment.

### Pip Installation Fails (`pip install -r requirements.txt`)

*   **Symptom:** Fails after successfully installing PyTorch.
*   **Possible Causes & Solutions:**
    *   **Missing PyTorch:** Ensure you successfully completed the PyTorch installation step *before* running `pip install -r requirements.txt`.
    *   **Network Issues:** Try again later.

### `ImportError: edansa` or Module Not Found

*   **Symptom:** Running the inference script fails with an `ImportError` for `edansa` or one of its submodules.
*   **Possible Causes & Solutions:**
    *   **Environment Not Activated:** You forgot to activate the correct Conda or Pip virtual environment before running the script. Activate it (e.g., `conda activate <your_env_name>` or `source .venv/bin/activate`).
    *   **Package Not Installed:** You missed **Step 3** of the installation: `pip install -e .`. Run this command from the root directory of the repository (`EDANSA/`) while your environment is activated.

### `torchaudio` Backend Error / Cannot Load Audio

*   **Symptom:** Errors related to loading audio files, potentially mentioning missing backends like FFmpeg or SoX.
*   **Possible Causes & Solutions:**
    *   **Missing Backend:** `torchaudio` needs a backend library like FFmpeg to load audio. 
        *   If using Conda with `environment.yml`, `ffmpeg` should have been installed. Verify using `python -c "import torchaudio; print(torchaudio.list_audio_backends())"`. If `ffmpeg` isn't listed, try reinstalling the environment or explicitly installing ffmpeg: `conda install ffmpeg -c conda-forge`.
        *   If using Pip, you likely need to install `ffmpeg` (or `sox`, `libsndfile`) using your system's package manager (e.g., `sudo apt update && sudo apt install ffmpeg`, `brew install ffmpeg`).
    *   **Corrupt Audio File:** The specific audio file might be corrupted or incomplete.
    *   **Unsupported Format (for Pip):** If using Pip, ensure the audio format is supported by the backend you have installed.

## Runtime Errors

### CUDA Errors (GPU Usage)

*   **Symptom:** Errors mentioning CUDA, `RuntimeError: CUDA error: out of memory`, `AssertionError: Torch not compiled with CUDA enabled`, or devices not found.
*   **Possible Causes & Solutions:**
    *   **Incorrect Environment:** You might have installed the CPU-only version of PyTorch but are trying to run on a GPU (or specified `--device cuda`). Ensure you installed using the GPU instructions (e.g., via `environment.yml` on a GPU machine or the correct `pip` command).
    *   **GPU Memory:** The model or batch size requires more GPU memory than available. Try reducing batch size (if applicable to the script/config) or running on a GPU with more memory. This model may require ~5GB+ VRAM.
    *   **Incorrect Device Specified:** Check the `--device` argument (`cuda`, `cuda:0`, `cpu`). Ensure it matches your available hardware.
    *   **Driver/CUDA Toolkit Mismatch:** The installed NVIDIA drivers might be incompatible with the CUDA toolkit version PyTorch was compiled against. Update your drivers.
    *   **Environment Not Activated:** Ensure the correct conda/pip environment with GPU PyTorch is activated.

### File Not Found Errors

*   **Symptom:** Errors indicating `--model_path`, `--config_file`, `--input_folder`, `--input_files_list`, or `--output_folder` cannot be found.
*   **Possible Causes & Solutions:**
    *   **Typo:** Double-check the spelling and case sensitivity of the path.
    *   **Incorrect Path:** Ensure the path is correct relative to your current working directory, or use an absolute path.
    *   **File/Folder Missing:** Verify that the specified file or folder actually exists at that location.
    *   **Permissions:** Ensure you have read permissions for input files/folders and write permissions for the output folder.

### Errors During Inference (`_process_single_audio_file`)

*   **Symptom:** The script processes some files but fails on others, often logging errors to `failed_files.csv` in the output folder.
*   **Possible Causes & Solutions:**
    *   **Corrupt/Invalid Audio File:** The audio file might be zero-length, corrupted, or in an unexpected format not handled correctly by the backend.
    *   **Memory Issues (RAM/CPU):** Very long audio files could potentially consume large amounts of RAM during loading or processing.
    *   **Channel Selection Error:** An invalid `--channel_selection_method` was provided, or the chosen channel index doesn't exist.
    *   **Unexpected Audio Properties:** Files with unusual sample rates or channel counts might cause issues if not handled correctly by resampling or channel selection.
    *   **Check `failed_files.csv`:** This file in your output directory contains the specific error message for each failed file, which is crucial for diagnosis.

## Unexpected Output

### Timestamps are Relative (Seconds) Instead of Absolute

*   **Symptom:** The `timestamp` column in the output CSV shows `0.0`, `10.0`, `20.0`, etc., instead of dates and times.
*   **Cause:** The input audio filenames did not match the expected `recorderid_YYYYMMDD_HHMMSS` or `YYYYMMDD_HHMMSS` format.
*   **Solution:** This is expected behavior if filenames aren't parsable. If you need absolute timestamps, rename your input files according to the convention described in the [Providing Audio Data](./using_pretrained_model/providing_audio_data.md#recommended-timestamp-formats-in-filename) section *before* running inference.

### Low Confidence Scores / Poor Performance

*   **Symptom:** The model produces very low confidence scores or seems to perform poorly on your data.
*   **Possible Causes & Solutions:**
    *   **Domain Mismatch:** The acoustic characteristics of your data (environment, noise, species vocalizations) might be significantly different from the model's training data.
    *   **Incorrect Model:** Ensure you are using the correct model (`general` or `bird_species`) for your target sounds.
    *   **High Noise / Clipping:** Check the `clipping` column in the output. High noise or clipping can degrade performance.
    *   **Thresholding:** Default thresholds might not be optimal for your specific data or goals. Refer to the Model Card for recommended starting thresholds and consider adjusting them based on your own evaluation.
    *   **Audio Quality:** Poor quality recordings (low sample rate, high compression artifacts if using lossy formats) can affect results.
