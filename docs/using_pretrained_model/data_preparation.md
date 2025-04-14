# Preparing Your Data for the Pre-trained Model

Before you can run the pre-trained EDANSA model on your audio recordings, you need to ensure they are in the correct file format and that their filenames follow specific conventions.

## Input Methods: File List or Folder Scan

You can provide audio files to the inference script in two ways:

1.  **`--input_files_list <path/to/list.txt>`**: Provide a plain text file where each line contains the **absolute path** to an audio file you want to process.

```text
/path/to/my/recordings/region_A/site_01/2022/REC001_20220701_100000.flac
/path/to/my/recordings/region_A/site_01/2022/REC001_20220701_110000.flac
/path/to/other/recordings/location_X/2023/AUDIOMOTH_20230515_083000.wav
...
```

2.  **`--input_folder <path/to/folder>`**: Provide the path to a directory. The script will **recursively** search this directory and its subdirectories for audio files with common extensions (like `.wav`, `.flac`, etc.).

## How Output Structure is Determined (No Required Input Structure)

The script **does not require** you to organize your input audio files into a specific folder structure like `region/location/year`. You can keep your files organized however you prefer (e.g., by deployment, recorder ID, or date).

However, the script *does* create an **output structure** that mirrors your input organization. Here's how it works:

1.  **Determine Input Root**: The script identifies a common base directory (root) from all the input file paths (either from the list or found in the folder).
2.  **Calculate Relative Path**: For each input audio file, it determines its path *relative* to that common root.
3.  **Create Mirrored Output**: It saves the resulting prediction CSV (or embedding file) under the directory specified by `--output_folder`, replicating the relative path structure.

**Example:**

Assume your `--output_folder` is set to `/data/edansa_outputs`.

*   **If you use `--input_files_list`** with files like:
    *   `/mnt/recordings/anwr/site_A/2022/REC01_20220601_000000.flac`
    *   `/mnt/recordings/anwr/site_B/2022/REC02_20220605_120000.flac`
    *   `/mnt/recordings/dalton/site_X/2023/AM05_20230810_060000.flac`

    The script might determine `/mnt/recordings/` as the common root. The outputs would be saved as:
    *   `/data/edansa_outputs/anwr/site_A/2022/REC01_20220601_000000.csv`
    *   `/data/edansa_outputs/anwr/site_B/2022/REC02_20220605_120000.csv`
    *   `/data/edansa_outputs/dalton/site_X/2023/AM05_20230810_060000.csv`

*   **If you use `--input_folder /mnt/recordings`** and that folder contains the same files as above within their respective subdirectories, the output structure under `/data/edansa_outputs` would be identical.

This approach allows you flexibility in organizing your input data while ensuring the outputs remain organized in a corresponding manner.

## Recommended File Naming for Timestamps (Not Required for Processing)

While the inference script can process audio files regardless of their filenames, **following a specific naming convention is highly recommended if you want absolute date and time information in your output CSVs.**

If the script can parse a start time from the filename, the `timestamp` column in the output CSV will contain absolute timestamps corresponding to the start of each analyzed segment.

**Example Output with Parsed Timestamp:**
```csv
timestamp,ClassA,ClassB,clipping
2022-08-02_22-59-08,0.1213,0.0324,0.122
2022-08-02_22-59-18,0.2013,0.0355,0.164
...
```

If the filename **does not** match the expected patterns, the script cannot determine the absolute start time. In this case, the `timestamp` column will contain relative time indices, representing seconds from the beginning of the audio file.

**Example Output without Parsed Timestamp:**
```csv
timestamp,ClassA,ClassB,clipping
0.0,0.1213,0.0324,0.122
10.0,0.2013,0.0355,0.164
...
```

### Recommended Timestamp Formats in Filename

To enable absolute timestamp generation, the core part of the filename (the "stem", without the file extension) should follow one of these two specific formats, separated by underscores (`_`):

*   **Format 1:** `recorderid_YYYYMMDD_HHMMSS` 
    *Example: `S4A10297_20190504_120000`*
*   **Format 2:** `YYYYMMDD_HHMMSS` 
    *Example: `20190504_120000`*

**Component Details:**

*   `recorderid`: (Optional) An identifier for the recording device. It can contain various characters but should not contain underscores itself.
*   `YYYYMMDD`: (Required for timestamp parsing) Must be exactly **8 digits** representing a valid date (Year, Month, Day).
*   `HHMMSS`: (Required for timestamp parsing) Must be exactly **6 digits** representing a valid time (Hour, Minute, Second) in 24-hour format.

!!! tip "Filename Format Impact"
    Following one of the recommended filename formats (exactly one or two underscores separating components including the required date and time) allows the script to automatically generate absolute timestamps in the output. Otherwise, relative timestamps (seconds from start) will be used.

## File Format: FLAC (`.flac`) Recommended
*   We strongly recommend using the Free Lossless Audio Codec (FLAC) format. FLAC provides lossless compression, meaning it reduces file size significantly compared to WAV without losing any audio quality. This is beneficial for storage and processing efficiency.
*   While the model might process other common lossless formats like WAV (`.wav`), using FLAC is preferred.
*   If your files are in a different format (like `.wav`), converting them to `.flac` is advised. Tools like `ffmpeg` (a free command-line tool) or Audacity (a free graphical audio editor) can perform this conversion.

!!! warning "Potential Metadata Loss During Conversion"
    Converting WAV files to FLAC while preserving embedded metadata (like recorder settings, GPS coordinates, etc.) can be challenging due to inconsistencies in how WAV files store metadata. 
    **This model does not require or use that embedded metadata.** However, if preserving this metadata is crucial for *your own* analysis or other projects, be aware that simple conversion tools (like the basic `ffmpeg` command shown) might not retain it. You may need specialized tools or scripts for metadata-preserving conversion, which are beyond the scope of this documentation.

!!! tip "Converting to Recommended FLAC using ffmpeg"
    If you have `ffmpeg` installed, you can convert a `.wav` file to `.flac` using a command like:
    ```bash
    ffmpeg -i input_audio.wav output_audio.flac
    ```
    You can often run this in batch across many files.

