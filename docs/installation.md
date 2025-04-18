# Installation Guide

This guide covers how to set up the necessary environment and install the EDANSA package to run the pre-trained models.

## Prerequisites

*   **Git:** You need Git installed to clone the repository. You can download it from [git-scm.com](https://git-scm.com/).
*   **(Recommended) Conda:** We strongly recommend using Conda (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)/Mamba/Micromamba) for managing environments and dependencies. See the [Conda documentation](https://docs.conda.io/) for installation details if needed.
*   **(Optional) Python:** If not using Conda, you'll need Python installed (version >= 3.8 and < 3.12).
*   **(GPU Users) NVIDIA Drivers:** If you intend to use a GPU, ensure you have appropriate NVIDIA drivers installed *before* creating the conda environment.

## Step 1: Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/speechLabBcCuny/EDANSA.git
cd EDANSA
```

## Step 2: Install Dependencies

Choose **one** of the following methods (Conda is recommended).

### Method 1: Using Conda (Recommended)

This method uses the provided `environment.yml` file, which specifies dependencies primarily from the `conda-forge` channel following [PyTorch's recommendations](https://github.com/pytorch/pytorch/issues/138506).

1.  **Install Conda:** If you don't have Conda, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or the full [Anaconda](https://www.anaconda.com/products/distribution).

2.  **Create Environment:** Create a new Conda environment using the `environment.yml` file. You **must** specify a name for your new environment using the `-n` flag. The file is configured to request the standard `pytorch` package from `conda-forge`. Conda's solver should automatically detect your system configuration and install the appropriate version (CPU or GPU-enabled with CUDA).

    ```bash
    # Replace <your_env_name> with your desired environment name (e.g., edansa)
    conda env create -f environment.yml -n <your_env_name>
    ```
    
    ***Alternatively: Updating an Existing Environment***
    If you already have a Conda environment activated that you wish to use, you can update it with the required packages. Ensure the target environment is activated *before* running the update command:
    ```bash
    # Activate your existing environment first
    # conda activate your-env-name
    
    # Using conda
    conda env update --file environment.yml
    
    # Or using micromamba
    micromamba env update -f environment.yml
    ```

3.  **Activate Environment:** Activate the environment using the name you chose during creation:
    ```bash
    conda activate <your_env_name> 
    ```
    You need to activate this environment every time you want to run the EDANSA scripts.

### Method 2: Using Pip

If you prefer not to use Conda, you can use `pip` with a virtual environment. This method requires installing PyTorch separately according to your system's configuration (CPU or specific CUDA version) **before** installing the other requirements.

1.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv 
    source .venv/bin/activate # On Windows use `.venv\\Scripts\\activate`
    ```

2.  **Install PyTorch:** 
    *   Visit the [official PyTorch website installation guide](https://pytorch.org/get-started/locally/).
    *   Select your operating system (Linux, Mac, Windows), package manager (`Pip`), compute platform (CPU or the specific CUDA version compatible with your drivers), and Python version.
    *   Copy the generated `pip install` command (it will likely include `--index-url` or `--extra-index-url`).
    *   Run that command in your activated virtual environment. For example:
        ```bash
        # Example for CPU-only on Linux/Mac:
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        
        # Example for CUDA 12.1 on Linux/Windows:
        # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
        ```
    *   **Important:** You must ensure you have the correct NVIDIA drivers and CUDA toolkit installed *manually* if you choose a CUDA option. `pip` will only install the PyTorch library compiled for that CUDA version, not the toolkit itself.

3.  **Install Remaining Dependencies:** Once PyTorch is installed correctly for your system, install the other required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    `pip` should detect that PyTorch, torchvision, and torchaudio are already installed and handle the rest.

    **Note:** Managing PyTorch versions and CUDA compatibility can sometimes be complex with `pip`. Conda often handles this more automatically.

## Step 2b: Audio Backend for Torchaudio

`torchaudio`, the library used for loading audio files, requires an external backend library to perform the actual decoding. Supported backends include FFmpeg (recommended, cross-platform), SoX (Linux/macOS), and SoundFile. [\[torchaudio Backends Documentation\]](https://pytorch.org/audio/stable/torchaudio.html#backend)

*   **Conda Users:** The provided `environment.yml` file includes `ffmpeg`. This backend should be installed automatically when you create the environment.
*   **Pip Users:** If you install dependencies using `pip` (Method 2), you might need to install `ffmpeg` or another backend (`libsndfile` for SoundFile, `sox`) separately.
    *   **Linux/macOS:** Use your system's package manager (e.g., `sudo apt update && sudo apt install ffmpeg`, `brew install ffmpeg`, `sudo yum install ffmpeg`).
    *   **Windows:**
        *   **Using a Package Manager:** If you use a Windows package manager like [Chocolatey](https://chocolatey.org/) or [Scoop](https://scoop.sh/), you can typically install FFmpeg with:
            *   Chocolatey: `choco install ffmpeg`
            *   Scoop: `scoop install ffmpeg`
        *   **Manual Installation:** If you don't use a package manager, you can download the `ffmpeg` shared build for Windows from the [official ffmpeg website](https://ffmpeg.org/download.html#build-windows). Extract the archive and add the `bin` directory (containing `ffmpeg.exe`, `ffprobe.exe`, etc.) to your system's `PATH` environment variable. You may need to restart your terminal or computer for the `PATH` change to take effect.

*   **Checking Available Backends:** Once you have installed dependencies and activated your environment (e.g., `<your_env_name>`), you can check which backends `torchaudio` can detect by running the following Python command:

    ```python
    python -c "import torchaudio; print(torchaudio.list_audio_backends())"
    ```
    Ensure that the output lists at least one available backend (e.g., `'ffmpeg'`, `'sox'`, `'soundfile'`).

## Step 3: Install the EDANSA Package

After installing the dependencies using either Conda or Pip, you need to install the `edansa` package itself. Make sure your environment (e.g., `<your_env_name>`) is activated, and then run the following command from the **root directory** of the cloned repository (`EDANSA/`):

```bash
pip install -e .
```

## Installation Complete

You should now have the necessary environment and the `edansa` package installed. You can proceed to run inference using the pre-trained models as described in the [Running Inference](./using_pretrained_model/index.md) guide.