import pandas as pd
from pathlib import Path
import argparse


def compare_csv_files(file1_path, file2_path, tolerance=1e-5):
    """
    Compares two CSV files using pandas, allowing for floating-point tolerance.

    Args:
        file1_path (Path): Path to the first CSV file.
        file2_path (Path): Path to the second CSV file.
        tolerance (float): Relative and absolute tolerance for numeric comparisons.

    Returns:
        bool: True if files are considered equal within tolerance, False otherwise.
        str: A message describing the comparison result.
    """
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        # Handle potential extra last column (e.g., clipping) in expected results (df2)
        if len(df1.columns) != len(df2.columns):
            if len(df2.columns) == len(df1.columns) + 1:
                print(
                    "  Note: Expected results (df2) has one extra column. Dropping last column from expected results for comparison."
                )
                df2 = df2.iloc[:, :-1]  # Drop the last column from df2
            # If column counts differ by more than 1, or if df1 has more cols,
            # assert_frame_equal will likely fail due to different columns/shapes,
            # which is the desired behavior.

        # Check for column name mismatches and align if necessary
        if list(df1.columns) != list(df2.columns):
            print(
                "  Warning: Column names differ between test and expected files. Comparing based on column order."
            )
            print(f"    Test columns:     {list(df1.columns)}")
            print(f"    Expected columns: {list(df2.columns)}")
            # Rename df2 columns to match df1 for positional comparison
            df2.columns = df1.columns

        # Optional: Sort columns for consistent comparison if order might differ
        # Note: Sorting columns might interfere with positional comparison if names were different.
        # df1 = df1.reindex(sorted(df1.columns), axis=1)
        # df2 = df2.reindex(sorted(df2.columns), axis=1)

        # Optional: Set a common index if applicable (e.g., 'timestamp')
        # Assuming 'timestamp' column exists and should be the index
        if 'timestamp' in df1.columns and 'timestamp' in df2.columns:
            try:
                # Convert to datetime if not already, handle potential errors
                # Specify the exact format for timestamp parsing
                timestamp_format = '%Y-%m-%d_%H-%M-%S'
                df1['timestamp'] = pd.to_datetime(df1['timestamp'],
                                                  format=timestamp_format)
                df2['timestamp'] = pd.to_datetime(df2['timestamp'],
                                                  format=timestamp_format)
                df1.set_index('timestamp', inplace=True)
                df2.set_index('timestamp', inplace=True)
                # Sort by index after setting it
                df1.sort_index(inplace=True)
                df2.sort_index(inplace=True)
            except Exception as e:
                return False, f"Error setting or sorting index 'timestamp': {e}"

        pd.testing.assert_frame_equal(
            df1,
            df2,
            check_dtype=
            False,  # Allow minor dtype differences (e.g., int vs float)
            check_exact=False,  # Allow for floating-point differences
            atol=tolerance,
            rtol=tolerance,
            check_names=
            False,  # Allow different names as we compare by position after aligning
        )
        return True, "Files are numerically equivalent."

    except FileNotFoundError as e:
        return False, f"Error reading file: {e}"
    except ValueError as e:
        # Catch errors from assert_frame_equal related to shape/labels
        return False, f"Comparison failed (shape/labels mismatch?): {e}"
    except AssertionError as e:
        # Catch errors from assert_frame_equal related to content mismatch
        # The error message from pandas is usually quite informative.
        return False, f"Comparison failed:\n{e}"
    except Exception as e:
        # Catch any other unexpected errors during read/comparison
        return False, f"An unexpected error occurred: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare CSV files in two directories.")
    parser.add_argument(
        "--test_dir",
        type=str,
        default="src/edansa/tests/assets/csezrx0a-V1/test_results/10s_csv",
        help="Directory containing the test CSV files.")
    parser.add_argument(
        "--expected_dir",
        type=str,
        default=
        "src/edansa/tests/assets/csezrx0a-V1/expected_results/10s_csv",
        help="Directory containing the expected CSV files.")
    parser.add_argument("--tolerance",
                        type=float,
                        default=1e-5,
                        help="Tolerance for floating-point comparisons.")

    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    expected_dir = Path(args.expected_dir)
    tolerance = args.tolerance

    if not test_dir.is_dir():
        print(f"Error: Test directory not found: {test_dir}")
        return
    if not expected_dir.is_dir():
        print(f"Error: Expected directory not found: {expected_dir}")
        return

    # Use rglob for recursive search and relative paths for matching
    # Normalize filenames by removing '_pred' suffix from the stem for matching
    def normalize_path(rel_path):
        if rel_path.stem.endswith('_pred'):
            norm_stem = rel_path.stem[:-len('_pred')]
            return rel_path.parent / (norm_stem + rel_path.suffix)
        return rel_path

    test_files = {
        normalize_path(f.relative_to(test_dir)): f
        for f in test_dir.rglob('*.csv')
    }
    expected_files = {
        normalize_path(f.relative_to(expected_dir)): f
        for f in expected_dir.rglob('*.csv')
    }

    print(
        f"Found {len(test_files)} test files and {len(expected_files)} expected files (after normalization)."
    )

    all_files = set(test_files.keys()) | set(expected_files.keys())
    files_match = True

    print(
        f"Comparing CSVs in:\n  Test: {test_dir}\n  Expected: {expected_dir}\n")

    for filename in sorted(list(all_files)):
        print(f"--- Comparing: {filename} ---")
        test_file_path = test_files.get(filename)
        expected_file_path = expected_files.get(filename)

        if not test_file_path:
            print("Status: FAIL - File missing in test results directory.")
            files_match = False
        elif not expected_file_path:
            print("Status: FAIL - File missing in expected results directory.")
            files_match = False
        else:
            print(
                f"  Comparing:\n    Test:     {test_file_path}\n    Expected: {expected_file_path}"
            )
            are_equal, message = compare_csv_files(test_file_path,
                                                   expected_file_path,
                                                   tolerance)
            if are_equal:
                print(f"Status: OK - {message}")
            else:
                print(f"Status: FAIL - {message}")
                files_match = False
        print("-" * (len(str(filename)) + 16))  # Use str(filename) for length

    print("\n--- Summary ---")
    if files_match:
        print(
            "All corresponding files are numerically equivalent within tolerance."
        )
    else:
        print("Differences found between test and expected results.")


if __name__ == "__main__":
    main()
