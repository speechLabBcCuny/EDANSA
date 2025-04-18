# Providing Audio Data for Inference

This page describes how to provide your audio recordings to the inference script and explains how filenames and file formats can influence the process and results.

## Input Methods: File List or Folder Scan

You can provide audio files to the inference script (`runs/augment/inference.py`) in two flexible ways:

1.  **`--input_files_list <path/to/list.txt>`**: Provide a plain text file where each line contains the **absolute path** to an audio file you want to process.
    *   The files listed can be located anywhere on your system; they don't need to share a common directory structure.

```text
/path/to/my/recordings/region_A/site_01/2022/REC001_20220701_100000.flac
/path/to/other/recordings/location_X/2023/AUDIOMOTH_20230515_083000.wav
/another/drive/misc_audio/test_clip.flac
...
```

2.  **`--input_folder <path/to/folder>`**: Provide the path to a directory. The script will **recursively** search this directory and all its subdirectories for audio files to process.
    *   Officially supported and tested formats are WAV (`.wav`) and FLAC (`.flac`).
    *   While the script might attempt to load other formats (like MP3, OGG, AIFF) due to underlying libraries, using WAV or FLAC is strongly recommended for reliable results.

## How Output Structure Mirrors Input

Although the script doesn't require a specific input folder structure, it intelligently organizes the output files to correspond to your input organization.

1.  **Determine Input Root**: The script identifies a common base directory (the "root") from all the processed input file paths.
2.  **Calculate Relative Path**: For each input file, it determines its path *relative* to that common root.
3.  **Create Mirrored Output**: It saves the resulting prediction CSV under the directory specified by `--output_folder`, recreating the relative path structure from the input root.

**Example:**

If `--output_folder` is `/data/edansa_outputs`:

*   **Using `--input_files_list`** with:
    *   `/mnt/recordings/anwr/site_A/2022/REC01_20220601_000000.flac`
    *   `/mnt/recordings/dalton/site_X/2023/AM05_20230810_060000.flac`

    The common root is likely `/mnt/recordings/`. Outputs appear as:

    *   `/data/edansa_outputs/anwr/site_A/2022/REC01_20220601_000000.csv`
    *   `/data/edansa_outputs/dalton/site_X/2023/AM05_20230810_060000.csv`

*   **Using `--input_folder /mnt/recordings`** (where the above files reside in subdirs) yields the same output structure.

This mirroring ensures that your output results remain organized consistently with your input data, regardless of how you structure the input.

## Generating Timestamps in Output (Optional Filename Convention)

The inference script can process audio files with **any filename**. However, if you want the output CSV files to contain absolute timestamps (e.g., `2023-10-27 14:30:00`), you need to follow a specific filename convention.

*   **If Filename Contains Recognizable Timestamp:** The script attempts to parse the date and time from the filename stem (the part before the extension like `.wav` or `.flac`). If successful, the `timestamp` column in the output CSV will contain absolute timestamps for the start of each prediction segment.

    **Example Output (Timestamp Parsed):**
```csv
timestamp,ClassA,ClassB,clipping
2022-08-02_22-59-08,0.1213,0.0324,0.122
2022-08-02_22-59-18,0.2013,0.0355,0.164
...
```

*   **If Filename Does Not Contain Recognizable Timestamp:** If the script cannot parse a date and time from the filename, the `timestamp` column will contain **relative time indices** in seconds, starting from 0.0 for the beginning of the file.

    **Example Output (Timestamp Not Parsed):**
```csv
timestamp,ClassA,ClassB,clipping
0.0,0.1213,0.0324,0.122
10.0,0.2013,0.0355,0.164
...
```

### Recommended Timestamp Formats in Filename

For the script to recognize and parse absolute timestamps, the filename stem should follow one of these formats (using underscores `_` as separators):

*   **Format 1:** `recorderid_YYYYMMDD_HHMMSS`
    *Example: `S4A10297_20190504_120000.flac`*
*   **Format 2:** `YYYYMMDD_HHMMSS`
    *Example: `20190504_120000.wav`*

**Details:**

*   `recorderid`: (Optional) Identifier for the device. Avoid underscores within the ID.
*   `YYYYMMDD`: **Required** 8 digits for a valid date.
*   `HHMMSS`: **Required** 6 digits for a valid 24-hour time.

!!! tip "Filename Impact on Timestamps"
    Using one of the recommended filename formats enables automatic absolute timestamps in the output. Any other filename format will result in relative timestamps (seconds from start).

## Recommended Audio File Formats

*   **Supported & Recommended:** WAV (`.wav`) and FLAC (`.flac`) are the officially supported and tested formats. We strongly recommend using one of these for reliable results.
*   **FLAC Advantage:** FLAC offers lossless compression, reducing file size compared to WAV without quality loss, which is ideal for storage and transfer.
*   **Conversion:** If your files are in other formats (e.g., MP3), consider converting them to WAV or FLAC using tools like `ffmpeg` or Audacity.

!!! warning "Metadata During Conversion"
    Standard conversion methods might not preserve all embedded metadata (e.g., GPS tags) from original files. This model **does not use** such metadata, but if you need it for other purposes, use conversion tools carefully.

!!! tip "Basic Conversion with ffmpeg"
```bash
# Convert WAV to FLAC
ffmpeg -i input_audio.wav output_audio.flac

# Convert MP3 to FLAC
ffmpeg -i input_audio.mp3 output_audio.flac 
```

