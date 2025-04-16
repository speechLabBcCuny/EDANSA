import pytest
import sys
from pathlib import Path
import os

# --- Test Data Setup ---
# Get the absolute path to the directory containing this conftest.py
CONFTEST_DIR = Path(__file__).parent.resolve()
# Get the path to the parent directory (src/edansa/tests/)
TESTS_DIR = CONFTEST_DIR.parent
# Path to the download script
DOWNLOAD_SCRIPT_PATH = TESTS_DIR / "download_test_data.py"

# Add the directory containing the download script to sys.path temporarily
# so we can import from it.
sys.path.insert(0, str(TESTS_DIR))

try:
    # Import the necessary function and variables from the script
    from download_test_data import download_and_extract, TARGET_ASSET_DIR, MARKER_FILE
    DOWNLOAD_SCRIPT_IMPORTED = True
except ImportError as e:
    print(f"\n--- ERROR: Could not import test data download script ---")
    print(f"Attempted import from: {DOWNLOAD_SCRIPT_PATH}")
    print(f"Error details: {e}")
    print(f"E2E tests requiring audio assets will likely fail.")
    print(f"---")
    # Define dummy values so pytest doesn't crash trying to define the fixture
    TARGET_ASSET_DIR = Path("/tmp/dummy_path_import_error")
    MARKER_FILE = TARGET_ASSET_DIR / ".download_complete"

    def download_and_extract():
        return False

    DOWNLOAD_SCRIPT_IMPORTED = False
finally:
    # Remove the path we added to avoid potential side effects
    if str(TESTS_DIR) in sys.path:
        sys.path.remove(str(TESTS_DIR))


@pytest.fixture(scope="session", autouse=True)
def ensure_e2e_test_data(pytestconfig):
    """
    Session-scoped fixture to automatically download E2E test audio data
    if it's not already present before running tests in the 'test_e2e' directory.
    Only runs if the download script was imported successfully.
    Checks for a custom marker or command-line option to skip download.
    """
    # Check if the download script import failed
    if not DOWNLOAD_SCRIPT_IMPORTED:
        print("Skipping test data check because download script import failed.")
        return  # Don't attempt download if import failed

    # Allow skipping download via command line or environment variable (optional)
    skip_download = (pytestconfig.getoption("--skip-data-download",
                                            default=False) or
                     os.environ.get("SKIP_TEST_DATA_DOWNLOAD", "0") == "1")
    if skip_download:
        print(
            "Skipping test data download check due to command-line flag or environment variable."
        )
        # If skipped, we assume data exists or tests will handle absence
        if not MARKER_FILE.exists():
            print(
                f"Warning: Test data download skipped, but marker file {MARKER_FILE} not found."
            )
        return

    print("\nChecking for E2E test audio data...")
    if not MARKER_FILE.exists():
        print(f"Marker file '{MARKER_FILE}' not found.")
        print("Attempting to download test data...")
        success = download_and_extract()
        if not success:
            # Fail the pytest session if download is essential and fails
            pytest.fail(
                f"Failed to download/extract required E2E test data to {TARGET_ASSET_DIR}. "
                f"See script output above. Tests cannot continue.",
                pytrace=False)
        else:
            print("Test data download/extraction successful.")
    else:
        print(f"Test data found at '{TARGET_ASSET_DIR}'. Skipping download.")


# You can also define command line options here if needed
# Example: Allow skipping the download check
def pytest_addoption(parser):
    parser.addoption("--skip-data-download",
                     action="store_true",
                     default=False,
                     help="Skip the automatic download of test data.")


# --- End Test Data Setup ---
