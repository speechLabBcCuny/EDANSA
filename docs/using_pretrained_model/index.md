# Running Inference with a Pretrained Model

This document explains how to use the inference script (`runs/augment/inference.py`) to generate predictions from audio files using a pretrained EDANSA model.

## Basic Usage

The core command structure involves specifying the model, its configuration, the input audio source, and the desired output location.

```bash
python runs/augment/inference.py \
    --model_path <path_to_model.pt> \
    --config_file <path_to_config.json> \
    --input_folder <path_to_your_audio_folder> \
    --output_folder <path_to_save_results>
```

## Example: Running the Main EDANSA Model (ID: 31m2plxv-V1)

Here is a concrete example using the primary pre-trained EDANSA model included in the `assets` directory. This command assumes:

*   You are running the command from the root directory of the `EDANSA-2019` repository.
*   Your audio files are located in a directory named `my_audio_files`.
*   You want to save the results to a directory named `inference_results`.

```bash
python runs/augment/inference.py \
    --model_path assets/31m2plxv-V1/model_info/best_model_370_val_f1_min=0.8028.pt \
    --config_file assets/31m2plxv-V1/model_info/model_config.json \
    --input_folder my_audio_files/ \
    --output_folder inference_results/
```

## Command-Line Arguments

The inference script accepts several arguments to control its behavior:

### Required Arguments

*   `--model_path <path>`: **Required.** Path to the trained model checkpoint file (e.g., `.pth`).
*   `--config_file <path>` / `-c <path>`: **Required.** Path to the JSON configuration file associated with the model. This file contains essential parameters like sampling rate, class labels, and excerpt length used during training.

### Input Source (Choose ONE)

You must specify one of the following options to provide the audio files for inference:

*   `--input_files_list <path>`: Path to a text file where each line contains the full path to an audio file to be processed.
*   `--input_folder <path>`: Path to a folder containing audio files. The script will recursively search this folder for audio files. Officially supported and tested formats are WAV (`.wav`) and FLAC (`.flac`). While it may attempt to load other formats like MP3, OGG, or AIFF, these are not guaranteed to work correctly.

### Output Control

*   `--output_folder <path>` / `-O <path>`: Directory where the prediction files will be saved.
    *   If not specified, defaults to a folder named `outputs` in the current working directory.
    *   The script will create one CSV file per input audio file. It creates subdirectories within the output folder that mirror the structure of the input folder or the paths provided in the input list.

### Execution Environment

*   `--device <device_name>`: Specify the computational device. Examples: `'cpu'`, `'cuda'`, `'cuda:0'`.
    *   If not specified, the script defaults to `'cuda'` if a CUDA-compatible GPU is detected, otherwise it uses `'cpu'`.

## Output Format

*   **Predictions:** Results are saved as CSV files, with one file generated for each input audio file.
    *   The directory structure within the `--output_folder` will mirror the structure of the input source (either the `--input_folder` or the paths from `--input_files_list`).
    *   Each CSV file contains:
        *   Timestamps for each prediction segment (either absolute datetime if parsed from filename, or relative seconds).
        *   Confidence scores per target class defined in the config file.
        *   Clipping percentage per segment (if not skipped via `--skip_clipping_info`).

## Error Handling

If the script encounters an error while processing a specific audio file (e.g., loading error, processing error, save error), it will:

1.  Log the error message to the console/log output.
2.  Record the failed file path and the error message in a CSV file named `failed_files.csv` located within the specified `--output_folder`. This allows you to easily identify and investigate problematic files after a large batch run.

## Advanced Settings

### Audio Processing Options

*   `--channel_selection_method <method>`: Specifies how to handle multi-channel (e.g., stereo) audio files. Options are:
    *   `'average'` (Default): Averages the channels to create a mono signal.
    *   `'clipping'`: Selects the channel with the least clipping *per segment* (defined by `excerpt_length` in the model config). Requires valid clipping data calculation. Falls back to `'average'` if clipping calculation fails or data is invalid.
    *   `'channel_N'`: Selects a specific channel by its index (e.g., `'channel_0'`, `'channel_1'`).
*   `--skip_clipping_info`: If this flag is present, the script will not calculate or include the percentage of clipped samples in the output results for prediction files. By default (flag absent), clipping information is calculated and included if possible.

### Optional Processing & Behavior

*   `--force_overwrite`: If this flag is present, the script will process all input files, even if a corresponding output file already exists in the output folder.
    *   By default (flag absent), the script checks for existing output files and skips processing for files where the output already exists.
