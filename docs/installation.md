# Installation Guide

This guide covers how to set up the necessary environment and install the EDANSA package to run the pre-trained models.

## Prerequisites

*   **Git:** You need Git installed to clone the repository. You can download it from [git-scm.com](https://git-scm.com/).
*   **(Recommended) Micromamba/Mamba or Conda:** For managing environments and dependencies, we strongly recommend Micromamba or Mamba for significantly faster performance. Standard Conda (from Miniconda or Anaconda) can also be used. If you need to install one, see the [Micromamba installation guide](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or the [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html).
*   **(Optional) Python:** If not using Conda, you'll need Python installed (version >= 3.10 and < 3.15).
*   **(GPU Users) NVIDIA Drivers:** If you intend to use a GPU, ensure you have appropriate NVIDIA drivers installed *before* creating the conda environment.

## Step 1: Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/speechLabBcCuny/EDANSA.git
cd EDANSA
```

## Step 2: Install Dependencies

Choose **one** of the following methods (Conda is recommended).

### Method 1: Using Conda/Mamba (Recommended)

This method uses the provided environment files, which specify dependencies primarily from the `conda-forge` channel and ensure correct PyTorch (CPU/GPU) and `torchaudio` (with FFmpeg) setup. We recommend using Mamba or Micromamba for faster environment creation.

1.  **Choose your Conda-based Package Manager:**
    We recommend using a Mamba-based package manager for significantly faster environment creation and dependency resolution compared to standard Conda.
    *   **Micromamba (Recommended):** A fast, standalone Conda-compatible package manager. Follow the [official Micromamba installation instructions](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html). The subsequent commands in this guide will use `micromamba`.
    *   **Mamba (Alternative):** If you have an existing Conda installation ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda), you can install Mamba into your base environment:
        ```bash
        conda install mamba -n base -c conda-forge
        ```
        If you use Mamba, simply replace `micromamba` with `mamba` in the commands below (e.g., `mamba create ...`).
    *   **Conda:** You can also use standard `conda` (e.g., `conda env create ...`), though it may be slower.

2.  **Create Environment:** Choose the environment file based on your system (GPU or CPU). You **must** specify a name for your new environment using the `-n` flag (or `--name` for Mamba/Conda).

    *   **For GPU-enabled systems (e.g., HPC nodes with NVIDIA GPUs):**
        This environment uses `pytorch-gpu` and CUDA-enabled `torchaudio`. Ensure your NVIDIA drivers are installed and compatible.
        ```bash
        micromamba create -f environment_gpu.yml -n <your_env_name>
        ```

    *   **For CPU-only systems (e.g., local machines, CPU-only servers):**
        This environment uses CPU-only `pytorch` and `torchaudio`.
        ```bash
        micromamba create -f environment_cpu.yml -n <your_env_name>
        ```

3.  **Activate Environment:** Activate the environment using the name you chose:
    ```bash
    micromamba activate <your_env_name>
    ```
    You need to activate this environment every time you want to run the EDANSA scripts.

### Method 2: Using Pip

This method uses `pip` and specific `requirements_*.txt` files to install dependencies, including PyTorch and the EDANSA package.

1.  **Create and Activate Virtual Environment:**
    It's highly recommended to use a virtual environment to isolate your project dependencies.
    ```bash
    python -m venv .venv 
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```

2.  **Install Dependencies using Pip:**
    Choose the requirements file that matches your system (CPU or GPU with CUDA 12.8). Then, run the `pip install -r` command. This will install PyTorch, Torchvision, Torchaudio from the correct PyTorch index, all other dependencies, and the EDANSA package in editable mode.

    *   **For CPU-only systems:**
        ```bash
        pip install -r requirements_cpu.txt
        ```

    *   **For GPU-enabled systems (CUDA 12.8):**
        Ensure your NVIDIA drivers are installed and compatible with CUDA 12.8.
        ```bash
        pip install -r requirements_gpu_cu128.txt
        ```
        *   **Important for CUDA users:** If you are installing for GPU, you must ensure you have the correct NVIDIA drivers and the corresponding CUDA Toolkit (version 12.8 in this case) installed *manually* on your system. `pip` only installs the PyTorch library compiled for that CUDA version, not the CUDA Toolkit itself.

    This single command handles all Python package installations.

## Audio Backend for Torchaudio (FFmpeg)

`torchaudio`, the library used for loading audio files, requires an external backend library to perform the actual decoding. Supported backends include FFmpeg (recommended, cross-platform), SoX (Linux/macOS), and SoundFile. [\[torchaudio Backends Documentation\]](https://pytorch.org/audio/stable/torchaudio.html#backend)

*   **Conda/Mamba/Micromamba Users:** The provided `environment_gpu.yml` and `environment_cpu.yml` files include `ffmpeg`. This backend is installed automatically when you create the environment using these files.
*   **Pip Users:** If you install dependencies using `pip` (Method 2), you need to install `ffmpeg` or another backend (`libsndfile` for SoundFile, `sox`) separately.
    *   **Important FFmpeg Version Note:** `torchaudio` does not work with FFmpeg 7.x. It is recommended to install an FFmpeg 6.x or 5.x version.
    *   **Linux/macOS:** Use your system's package manager. You may need to specify a version, for example:
        *   `sudo apt update && sudo apt install ffmpeg` (check the version provided by your distribution, or look for `ffmpeg-6` or `ffmpeg-5` packages if available).
        *   `brew install ffmpeg@6` (or `ffmpeg@5` if `ffmpeg@6` is not available or if you prefer an older stable version).
        *   `sudo yum install ffmpeg` (similarly, check version or look for specific versioned packages).
    *   **Windows:**
        *   **Using a Package Manager:** If you use a Windows package manager like [Chocolatey](https://chocolatey.org/) or [Scoop](https://scoop.sh/), check if your package manager allows installing a specific version (e.g., `choco install ffmpeg --version=6.0`). You can typically install FFmpeg with:
            *   Chocolatey: `choco install ffmpeg`
            *   Scoop: `scoop install ffmpeg`
        *   **Manual Installation:** If you don't use a package manager, when downloading from the [official ffmpeg website](https://ffmpeg.org/download.html#build-windows), look for builds based on FFmpeg 6.x or 5.x. Extract the archive and add the `bin` directory (containing `ffmpeg.exe`, `ffprobe.exe`, etc.) to your system's `PATH` environment variable. You may need to restart your terminal or computer for the `PATH` change to take effect.

*   **Checking Available Backends:** Once you have installed dependencies and activated your environment (e.g., `<your_env_name>`), you can check which backends `torchaudio` can detect by running the following Python command:

    ```python
    python -c "import torchaudio; print(torchaudio.list_audio_backends())"
    ```
    Ensure that the output lists at least one available backend (e.g., `'ffmpeg'`, `'sox'`, `'soundfile'`).

## Installation Complete

You should now have the necessary environment and the `edansa` package installed. You can proceed to run inference using the pre-trained models as described in the [Running Inference](./using_pretrained_model/index.md) guide.