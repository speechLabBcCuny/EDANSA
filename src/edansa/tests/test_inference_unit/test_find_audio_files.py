import pytest
from pathlib import Path
import sys
import os

# Add the src directory to the Python path to import edansa modules
# This assumes the tests directory is two levels up from this file
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Now import the function to be tested
# Need to find where _find_audio_files is defined. It's in src/edansa/inference.py
from edansa.inference import _find_audio_files


# Test function for _find_audio_files
def test_find_audio_files(tmp_path):
    """
    Tests the _find_audio_files function to ensure it correctly identifies
    supported audio files, handles case insensitivity, ignores unsupported files,
    and searches recursively.
    """
    # --- Test Setup ---
    # Create dummy files in the temporary directory provided by pytest
    base_dir = tmp_path / "audio_test_folder"
    base_dir.mkdir()
    sub_dir = base_dir / "subdir"
    sub_dir.mkdir()

    # List of filenames to create (relative paths within base_dir)
    files_to_create = [
        "test1.wav",
        "test2.WAV",
        "test3.flac",
        "test4.ogg",
        "test5.mp3",
        "test6.aif",
        "test7.aiff",
        "test8.CSV",  # Uppercase unsupported
        "test9.txt",
        "not_audio.log",
        Path("subdir") / "sub_test1.mp3",
        Path("subdir") / "sub_test2.FLAC",
        Path("subdir") / "sub_test3.csv",
    ]

    # Expected list of valid audio files (as Path objects, relative to tmp_path)
    # Important: Convert to absolute paths as _find_audio_files returns absolute paths
    expected_files = sorted([
        base_dir / "test1.wav",
        base_dir / "test2.WAV",
        base_dir / "test3.flac",
        base_dir / "test4.ogg",
        base_dir / "test5.mp3",
        base_dir / "test6.aif",
        base_dir / "test7.aiff",
        sub_dir / "sub_test1.mp3",
        sub_dir / "sub_test2.FLAC",
    ])

    # Create the files
    for file_rel_path in files_to_create:
        full_path = base_dir / file_rel_path
        # Ensure parent directory exists if it's a sub_dir file
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.touch()  # Creates an empty file

    # --- Call the function under test ---
    found_files = _find_audio_files(str(base_dir))

    # --- Assertions ---
    # Ensure found_files are absolute paths for comparison
    found_files_abs = sorted([f.resolve() for f in found_files])
    expected_files_abs = sorted([f.resolve() for f in expected_files])

    # 1. Check if the correct number of files were found
    assert len(found_files_abs) == len(expected_files_abs), \
        f"Expected {len(expected_files_abs)} audio files, but found {len(found_files_abs)}"

    # 2. Check if the found files match the expected files exactly
    assert found_files_abs == expected_files_abs, \
        f"Mismatch between found and expected files.\nFound: {found_files_abs}\nExpected: {expected_files_abs}"

    # 3. Explicitly check that unsupported files were *not* included
    unsupported_files = [
        base_dir / "test8.CSV",
        base_dir / "test9.txt",
        base_dir / "not_audio.log",
        sub_dir / "sub_test3.csv",
    ]
    for unsupported in unsupported_files:
        assert unsupported.resolve() not in found_files_abs, \
            f"Unsupported file '{unsupported.name}' was incorrectly included in the results."
