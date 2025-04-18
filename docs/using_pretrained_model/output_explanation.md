# Understanding the Inference Results

After running the inference script (`runs/augment/inference.py`), you will find CSV files in your specified output directory. Each CSV file corresponds to one input audio file and contains the model's predictions segment by segment.

Here's a breakdown of the columns you'll typically find in the output CSV:

## Columns Explained

*   **`timestamp`**: 
    *   Indicates the start time of the prediction segment.
    *   If you followed the [recommended filename convention](./providing_audio_data.md#recommended-timestamp-formats-in-filename) for your input audio files, this column will contain **absolute timestamps** (e.g., `2023-10-27 14:30:00`).
    *   If the filename could not be parsed for a date and time, this column will contain **relative timestamps** in seconds from the beginning of the audio file (e.g., `0.0`, `10.0`, `20.0`).
    *   The duration of each segment is determined by the `excerpt_length` parameter in the model's configuration file (`--config_file`).

*   **Confidence Score Columns (e.g., `pred_Aircraft`, `pred_Songbird`, `pred_Bird`, ...)**:
    *   There will be one column for each target class the model was trained to detect. The class names are defined in the model's configuration file.
    *   The value in each column represents the model's **confidence score** (typically ranging from 0.0 to 1.0) that the corresponding class is present within that specific time segment.
    *   A score closer to 1.0 indicates higher confidence, while a score closer to 0.0 indicates lower confidence.
    *   You can use these scores to set a threshold for deciding whether a class is considered present (e.g., consider a class present if its score is > 0.7).

*   **`clipping`**:
    *   This column shows the **proportion (0.0 to 1.0) of audio samples that were clipped** within that specific time segment.
    *   **Audio clipping** occurs when the amplitude (loudness) of the sound exceeds the maximum level that can be recorded or represented digitally. This results in distortion of the waveform, essentially chopping off the peaks.
    *   A high clipping proportion (e.g., > 0.01-0.05, corresponding to 1-5%) might indicate:
        *   The recording gain was set too high.
        *   There were very loud sounds (potentially non-target noise like wind, rain, or handling noise) close to the microphone.
        *   Potential data quality issues for that segment.
    *   While the model processes these segments, high clipping can sometimes affect prediction accuracy. This column provides context for interpreting the results.
    *   This column will contain `NaN` (Not a Number) or be absent if clipping calculation was skipped using the `--skip_clipping_info` flag during inference.

## Example CSV Snippet (with Absolute Timestamps)

```csv
timestamp,pred_Aircraft,pred_Songbird,pred_Bird,clipping
2023-07-15 04:00:00,0.85,0.05,0.10,0.002
2023-07-15 04:00:10,0.88,0.03,0.12,0.001
2023-07-15 04:00:20,0.15,0.75,0.05,0.015
```

In this example, the model is highly confident about Aircraft presence in the first two 10-second segments but detects Songbird with higher confidence in the third segment, which also shows a slightly higher clipping proportion (0.015 or 1.5%).
