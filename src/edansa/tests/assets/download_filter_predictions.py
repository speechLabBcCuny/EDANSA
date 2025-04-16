import os
import glob
import pandas as pd
import soundfile as sf
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np  # soundfile dependency
import pathlib  # Using pathlib for consistency and better path handling
import subprocess  # For running rclone
import shutil  # For checking rclone existence
import argparse  # For command-line arguments

# --- Base Configuration (can be overridden or used in argument parsing) ---
BASE_ASSET_DIR = pathlib.Path("src/edansa/tests/assets")
AUDIO_BASE_DIR = BASE_ASSET_DIR / "audio/real"
# Path on Box up to the model ID level
NNA_BOX_REMOTE_PATH_BASE = "nnabox:NNA Project Products (Data, Code, Pubs)/Data products/sound_model_results/results/latest"
# --- End Base Configuration ---


def parse_audio_filename_to_datetime(filename):
    """Parses start timestamp from audio filename into a datetime object."""
    # Assumes filename format like S4A..._YYYYMMDD_HHMMSS.ext
    parts = os.path.splitext(filename)[0].split('_')
    if len(parts) >= 3:
        try:
            timestamp_str = f"{parts[-2]}_{parts[-1]}"
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            print(
                f"  [Warning] Could not parse datetime from audio filename: {filename}"
            )
            pass
        except IndexError:
            print(
                f"  [Warning] Could not parse datetime, unexpected parts in audio filename: {filename}"
            )
            pass
    return None


def get_audio_file_time_range(file_path):
    """Gets the start datetime and duration (in seconds) of an audio file."""
    filename = os.path.basename(file_path)
    start_time = parse_audio_filename_to_datetime(filename)
    if start_time is None:
        return None, None

    try:
        # Use pathlib Path object for soundfile
        with sf.SoundFile(str(file_path), 'r') as f:
            duration_seconds = len(f) / f.samplerate
            return start_time, duration_seconds
    except Exception as e:
        print(f"  [Warning] Could not read audio metadata for {file_path}: {e}")
        # Return start time even if duration fails, maybe useful later?
        return start_time, None


def download_predictions(required_loc_site_year_dirs, rclone_remote_base,
                         remote_subdir, model_id, local_dest_base,
                         force_download):
    """Downloads necessary prediction folders (location/site/year) using rclone.

    Args:
        required_loc_site_year_dirs (set): A set of relative 'location/site/year' strings.
        rclone_remote_base (str): The base path on the rclone remote up to 'latest'.
        remote_subdir (str): Optional subdirectory between 'latest' and model ID (e.g., 'birds').
        model_id (str): The specific model ID (e.g., "31m2plxv-V1").
        local_dest_base (pathlib.Path): The local base directory to download into (e.g., .../MODEL_ID/10s_csv).
        force_download (bool): If True, download even if the destination exists and is not empty.
    Returns:
        bool: True if download was successful, skipped, or appeared successful, False otherwise.
    """
    print(f"\n--- Rclone Download Check ---")

    # Check if download should be skipped
    if not force_download and local_dest_base.exists() and any(
            local_dest_base.iterdir()):
        print(
            f"[Info] Download destination {local_dest_base} already exists and is not empty."
        )
        print("[Info] Skipping download. Use --force-download to overwrite.")
        return True  # Treat skip as success for workflow
    elif force_download:
        print(f"[Info] --force-download specified. Proceeding with download...")
    else:
        print(
            f"[Info] Download destination {local_dest_base} does not exist or is empty. Proceeding with download..."
        )

    # Construct the full remote source path including optional subdir, model ID, and 10s_csv
    remote_path_parts = [rclone_remote_base]
    if remote_subdir:
        remote_path_parts.append(remote_subdir)
    remote_path_parts.append(model_id)
    remote_path_parts.append(
        "10s_csv/")  # Ensure trailing slash for directory copy
    rclone_remote_source = "/".join(remote_path_parts)

    print(
        f"\n--- Starting Rclone Download --- (Force Download: {force_download})"
    )
    print(f"Source: {rclone_remote_source}")
    print(f"Destination Base: {local_dest_base}")
    print(f"Required location/site/year folders: {required_loc_site_year_dirs}")

    if not shutil.which("rclone"):
        print(
            "[Error] rclone command not found. Please install rclone and ensure it's in your PATH."
        )
        return False

    # Ensure destination base directory exists - rclone creates subdirs but base needs to exist
    local_dest_base.mkdir(parents=True, exist_ok=True)

    # Construct rclone command with include filters
    # Use 'copyto' to potentially handle the destination path structure more cleanly
    # We copy specific folders from the remote source into the local destination base
    cmd = ["rclone", "copy", "--progress"]  # Start with base command

    # Add include filters for each required directory relative to the source
    for loc_site_year in sorted(list(required_loc_site_year_dirs)
                               ):  # Sort for consistent command generation
        include_filter = f"/{loc_site_year}/**"  # Filter for the specific year directory and its contents
        cmd.extend(["--include", include_filter])

    # Add an exclude filter to ignore everything else at the top level of the source
    cmd.extend(["--exclude", "/**"])

    # Add source and destination paths
    cmd.append(rclone_remote_source)  # Source directory on remote
    cmd.append(str(local_dest_base))  # Local base directory destination

    print(f"Running rclone command: {' '.join(cmd)}"
         )  # Show command for debugging

    try:
        # Using shell=False (default) is safer, pass cmd as a list
        result = subprocess.run(cmd,
                                check=True,
                                capture_output=True,
                                text=True,
                                encoding='utf-8')
        print("[Info] rclone download completed successfully.")
        # Print stderr as rclone often outputs transfer stats there
        if result.stderr:
            print("[Info] rclone output:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] rclone command failed with exit code {e.returncode}")
        print("Stderr:")  # Separate print
        print(e.stderr)  # Print variable directly
        print("Stdout:")  # Separate print
        print(e.stdout)  # Print variable directly
        return False
    except FileNotFoundError:
        print(
            "[Error] rclone command not found during subprocess execution. Is it installed and in PATH?"
        )
        return False
    except Exception as e:
        print(
            f"[Error] An unexpected error occurred during rclone execution: {e}"
        )
        return False


def find_matches_and_filter(model_id, remote_subdir, force_download):
    """Determines requirements, downloads, finds prediction files overlapping
       with audio files, filters them, saves the filtered results to the download
       directory (10s_csv), and cleans up original downloaded files.

    Args:
        model_id (str): The model ID to process.
        remote_subdir (str | None): Optional subdirectory on Box between base and model ID.
        force_download (bool): Whether to force rclone download if destination exists.
    """
    # --- Derive paths based on model_id ---
    results_base_dir = BASE_ASSET_DIR / model_id / "10s_csv"  # Download AND final output location
    # filtered_results_base_dir = BASE_ASSET_DIR / model_id / "filtered_10s_csv" # No longer needed

    # --- 1. Determine required prediction directories based on audio files ---
    print(
        f"Scanning audio files in: {AUDIO_BASE_DIR} to determine required prediction folders..."
    )
    audio_file_ranges = defaultdict(list)  # Store detailed info
    required_loc_site_year_dirs = set(
    )  # Store unique 'location/site/year' strings
    found_audio_files = 0
    processed_audio_files = 0
    skipped_audio_parse = 0
    skipped_audio_meta = 0

    for audio_file_path in AUDIO_BASE_DIR.rglob(
            '*'):  # Recurse and get all files initially
        if audio_file_path.is_file() and audio_file_path.suffix.lower() in [
                '.wav', '.flac'
        ]:
            found_audio_files += 1
            # Use pathlib for relative path
            try:
                # Calculate relative directory path based on AUDIO_BASE_DIR structure
                relative_dir_parts = audio_file_path.relative_to(
                    AUDIO_BASE_DIR).parts[:-1]  # Exclude filename
                # Extract location/site/year (full relative path)
                if len(relative_dir_parts
                      ) >= 3:  # Expect at least location/site/year
                    loc_site_year = os.path.join(*relative_dir_parts)
                    required_loc_site_year_dirs.add(loc_site_year)
                # Handle edge cases like dempster/23/file.wav (no year folder in audio path)
                elif len(relative_dir_parts) == 2 and relative_dir_parts[
                        0] == 'dempster' and relative_dir_parts[1] == '23':
                    # Assume predictions are in a year folder, check common years or default
                    # For this specific known case, we found audio from 2023
                    # Let's assume the predictions are in location/site/YYYY format
                    # We need the *predicted* year folder structure, which might differ slightly.
                    # Based on previous checks, predictions for dempster/23 were in a 2023 folder.
                    loc_site_year = os.path.join(relative_dir_parts[0],
                                                 relative_dir_parts[1], "2023")
                    required_loc_site_year_dirs.add(loc_site_year)
                    print(
                        f"  [Info] Assuming prediction path {loc_site_year} for audio path {os.path.join(*relative_dir_parts)}"
                    )

                else:
                    print(
                        f" [Warning] Cannot determine location/site/year from path structure: {audio_file_path} relative parts: {relative_dir_parts}"
                    )
                # Decide if you want to skip or handle differently
                # continue

                relative_dir = os.path.join(
                    *relative_dir_parts
                )  # Keep full relative dir for matching later

            except ValueError:
                print(
                    f"  [Warning] Could not determine relative path for {audio_file_path} based on {AUDIO_BASE_DIR}"
                )
                continue  # Skip if it's somehow not relative

            start_time, duration = get_audio_file_time_range(audio_file_path)

            if start_time is None:
                skipped_audio_parse += 1
                continue

            if duration is None:
                skipped_audio_meta += 1
                # Handle as a zero-duration event for filtering purposes if duration fails
                end_time = start_time + timedelta(
                    seconds=0.1)  # Add small delta to avoid zero range issues
                duration = 0
            else:
                # Ensure predictions exactly at the end time are included
                end_time = start_time + timedelta(seconds=duration)

            # Store range info keyed by the full relative directory path (e.g., 'anwr/47/2022')
            audio_file_ranges[relative_dir].append({
                'path': audio_file_path,
                'start': start_time,
                'end': end_time,  # Use inclusive end time for filtering
                'duration': duration
            })
            processed_audio_files += 1

    print(f"Found {found_audio_files} potential audio files (.wav, .flac).")
    print(
        f"Successfully processed metadata for {processed_audio_files} audio files."
    )
    if skipped_audio_parse > 0:
        print(
            f"Skipped {skipped_audio_parse} audio files due to timestamp parsing issues."
        )
    if skipped_audio_meta > 0:
        print(
            f"Skipped {skipped_audio_meta} audio files due to metadata reading issues (duration could not be determined)."
        )

    if not required_loc_site_year_dirs:
        print(
            "[Error] No audio files found or processed. Cannot determine prediction folders to download."
        )
        return

    print(
        f"\nRequired location/site/year directories based on audio files: {required_loc_site_year_dirs}"
    )

    # --- 2. Download required predictions using rclone ---
    if not download_predictions(required_loc_site_year_dirs,
                                NNA_BOX_REMOTE_PATH_BASE, remote_subdir,
                                model_id, results_base_dir, force_download):
        print("[Error] Failed to download predictions. Aborting filtering.")
        return

    # --- 3. Filter downloaded predictions (Saving to results_base_dir) ---
    print(f"\nFiltering downloaded prediction files in: {results_base_dir}")
    # print(f"Filtered results will be saved to: {results_base_dir}") # Clarify output location

    # Set to keep track of the *paths* of the filtered files we create
    generated_filtered_files = set()

    created_filtered_files_count = 0  # Count files created
    processed_audio_files_count = 0  # Count audio files for which predictions were found and saved
    total_prediction_files = 0
    processed_prediction_files = 0
    matched_prediction_files = 0  # Count files that had at least one match
    skipped_pred_files_read = 0
    skipped_pred_files_empty = 0
    skipped_pred_files_timestamp_col = 0
    skipped_pred_files_timestamp_parse = 0

    # Use pathlib for globbing prediction files
    for pred_file_path in results_base_dir.rglob('*_pred.csv'):
        if pred_file_path.is_file():
            total_prediction_files += 1
            try:
                # Calculate relative directory path based on RESULTS_BASE_DIR structure
                # Example: 31m2plxv-V1/10s_csv/anwr/47/2022 -> anwr/47/2022
                relative_dir_parts = pred_file_path.relative_to(
                    results_base_dir).parts[:-1]  # Exclude filename
                relative_dir = os.path.join(*relative_dir_parts)
            except ValueError:
                print(
                    f"  [Warning] Could not determine relative path for {pred_file_path} based on {results_base_dir}"
                )
                continue  # Skip if it's somehow not relative

            relevant_audio_list = audio_file_ranges.get(relative_dir, [])
            if not relevant_audio_list:
                # print(f"  [Info] No corresponding audio files found for directory: {relative_dir}. Skipping {pred_file_path}")
                continue  # No audio files in this directory structure to filter against

            try:
                df = pd.read_csv(
                    str(pred_file_path))  # Use string representation for pandas
                if df.empty:
                    skipped_pred_files_empty += 1
                    continue
                if 'timestamp' not in df.columns:
                    skipped_pred_files_timestamp_col += 1
                    continue

                # Convert timestamp column once, coercing errors
                df['datetime'] = pd.to_datetime(df['timestamp'],
                                                format='%Y-%m-%d_%H-%M-%S',
                                                errors='coerce')
                original_rows = len(df)
                df.dropna(subset=['datetime'],
                          inplace=True)  # Remove rows where conversion failed
                rows_after_parse = len(df)
                if original_rows > rows_after_parse:
                    print(
                        f"  [Info] Dropped {original_rows - rows_after_parse} rows from {pred_file_path.name} due to invalid timestamp format."
                    )  # Changed to Info

                if df.empty:
                    # All timestamps might have been invalid after coerce
                    skipped_pred_files_timestamp_parse += 1
                    continue

                # *** MODIFIED LOGIC START ***
                # Iterate through each relevant audio file for this prediction CSV's directory
                processed_prediction_files += 1  # Count that we are attempting to process this pred file
                found_match_for_this_pred_file = False

                for audio_info in relevant_audio_list:
                    # Filter rows where prediction timestamp falls within this specific audio file's range
                    mask = (df['datetime'] >= audio_info['start']) & (
                        df['datetime'] < audio_info['end']
                    )  # Use < end for non-inclusive end
                    filtered_chunk = df.loc[mask].copy(
                    )  # Use .copy() to avoid SettingWithCopyWarning later

                    if not filtered_chunk.empty:
                        found_match_for_this_pred_file = True
                        processed_audio_files_count += 1  # Increment count for each audio file match saved

                        # --- Define output path and filename based on audio start time ---
                        # Save directly into the results_base_dir (10s_csv folder)
                        output_dir = results_base_dir / relative_dir
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Format filename from audio start time: YYYY-MM-DD_HH-MM-SS_pred.csv
                        audio_start_str = audio_info['start'].strftime(
                            '%Y-%m-%d_%H-%M-%S')
                        output_filename = f"{audio_start_str}_pred.csv"
                        output_path = output_dir / output_filename

                        # Prepare final dataframe for saving
                        final_filtered_df = filtered_chunk.drop(
                            columns=['datetime'])
                        final_filtered_df = final_filtered_df.sort_values(
                            by='timestamp')
                        # No need for drop_duplicates here as it's per audio file

                        final_filtered_df.to_csv(
                            str(output_path),
                            index=False)  # Use string path for pandas
                        generated_filtered_files.add(
                            output_path)  # Track the generated file path
                        created_filtered_files_count += 1

                if found_match_for_this_pred_file:
                    matched_prediction_files += 1  # Count pred files that had at least one audio match

                # *** MODIFIED LOGIC END ***

            except pd.errors.EmptyDataError:
                skipped_pred_files_empty += 1
            except Exception as e:
                print(
                    f"  [Error] Could not read or process prediction file {pred_file_path}: {e}"
                )
                skipped_pred_files_read += 1

    # --- Update Summary ---
    print(f"\n--- Summary ---")
    print(f"Found {total_prediction_files} prediction files (*_pred.csv).")
    # Corrected the processed count logic slightly
    print(
        f"Attempted to process {processed_prediction_files} prediction files (non-empty, had timestamp col)."
    )
    print(
        f"{matched_prediction_files} prediction files contained data matching one or more audio file time ranges."
    )
    print(
        f"Found matches and saved filtered data for {processed_audio_files_count} audio files."
    )  # New metric

    if skipped_pred_files_read > 0:
        print(
            f"Skipped {skipped_pred_files_read} prediction files due to read/processing errors."
        )
    if skipped_pred_files_empty > 0:
        print(
            f"Skipped {skipped_pred_files_empty} prediction files because they were empty."
        )
    if skipped_pred_files_timestamp_col > 0:
        print(
            f"Skipped {skipped_pred_files_timestamp_col} prediction files due to missing 'timestamp' column."
        )
    if skipped_pred_files_timestamp_parse > 0:
        print(
            f"Skipped {skipped_pred_files_timestamp_parse} prediction files because all timestamps were invalid or parsing failed."
        )

    print(
        f"\nFinal filtered data ({created_filtered_files_count} files) is located in {results_base_dir}"
    )
    print("\nFiltering complete.")

    # --- 5. Clean up original downloaded files ---
    print(f"\nCleaning up original files in {results_base_dir}...")
    deleted_count = 0
    potential_originals = 0
    for existing_file in results_base_dir.rglob('*_pred.csv'):
        if existing_file.is_file():
            potential_originals += 1
            if existing_file not in generated_filtered_files:
                try:
                    os.remove(existing_file)
                    print(
                        f"  Deleted original/unfiltered file: {existing_file.relative_to(results_base_dir)}"
                    )
                    deleted_count += 1
                except OSError as e:
                    print(
                        f"  [Error] Could not delete file {existing_file}: {e}")

    print(
        f"Cleanup finished. Checked {potential_originals} files, deleted {deleted_count} original/unfiltered files."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Download specific NNA Box prediction folders via rclone and filter them based on local audio files."
    )
    parser.add_argument(
        "model_id",
        help=
        "Model ID string (e.g., '31m2plxv-V1', 'csezrx0a-V1'). This is used for local directory names and the remote path."
    )
    parser.add_argument(
        "--remote-subdir",
        help=
        "Optional subdirectory between the base remote path (.../latest/) and the model ID (e.g., 'birds').",
        default=None)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=
        "Force rclone download even if the destination directory (10s_csv) exists and is not empty."
    )

    args = parser.parse_args()

    print(f"--- Starting Script --- Model ID: {args.model_id} ---")
    if args.remote_subdir:
        print(f"Remote Subdirectory: {args.remote_subdir}")
    print(f"Force Download: {args.force_download}")

    # Ensure necessary libraries are installed: pip install pandas soundfile numpy
    find_matches_and_filter(args.model_id, args.remote_subdir,
                            args.force_download)
