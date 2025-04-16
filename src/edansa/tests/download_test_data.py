# src/edansa/tests/download_test_data.py
import requests
import zipfile
import io
from pathlib import Path
import sys
import time  # Added for potential retry logic display

# --- Configuration ---
DATA_URL = "https://github.com/speechLabBcCuny/EDANSA/releases/download/dev-test-data-v1.0/edansa-test-assets-pack-v1.zip"

# Define the target directory relative to the script's location
# Script is now in src/edansa/tests/, so project root is 3 levels up
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # Adjusted path
TARGET_ASSET_DIR = PROJECT_ROOT / "src" / "edansa" / "tests" / "assets"
# A marker file to indicate successful download/extraction
MARKER_FILE = TARGET_ASSET_DIR / ".download_complete"
# --- End Configuration ---


def download_and_extract(max_retries=2, initial_delay=5):
    """Downloads and extracts the test audio assets. Returns True on success, False on failure."""
    if not DATA_URL or "PASTE_YOUR_GITHUB_RELEASE_ASSET_URL_HERE" in DATA_URL:
        print(
            "Error: DATA_URL not configured in src/edansa/tests/download_test_data.py."
        )
        print(
            "Please ensure the GitHub Release asset URL is correctly set in the script."
        )
        return False  # Indicate failure

    if MARKER_FILE.exists():
        # print(f"Test data marker found at '{MARKER_FILE}'. Skipping download.") # Keep console less noisy
        return True  # Indicate success (already done)

    print(f"Target directory: {TARGET_ASSET_DIR.resolve()}")
    print(f"Downloading test data from: {DATA_URL}...")

    # Ensure target parent directory exists
    try:
        TARGET_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating target directory {TARGET_ASSET_DIR}: {e}")
        return False

    retries = 0
    while retries <= max_retries:
        try:
            response = requests.get(DATA_URL, stream=True, timeout=600)
            response.raise_for_status()  # Check for HTTP errors

            print("Download connection successful. Receiving data...")
            content = response.content  # Read the whole content (ensure sufficient memory)
            print(
                f"Received {len(content) / (1024*1024):.2f} MB. Extracting...")

            with zipfile.ZipFile(io.BytesIO(content)) as z:
                # Security check for path traversal
                for member in z.namelist():
                    member_path = Path(member)
                    if member_path.is_absolute() or '..' in member_path.parts:
                        raise ValueError(
                            f"Zip file contains potentially unsafe path: {member}"
                        )
                # Extract
                z.extractall(TARGET_ASSET_DIR)

            # Create marker file
            MARKER_FILE.touch()
            print(
                f"Successfully downloaded and extracted test data to '{TARGET_ASSET_DIR}'."
            )
            return True  # Success!

        except requests.exceptions.RequestException as e:
            print(
                f"Error during download (attempt {retries + 1}/{max_retries + 1}): {e}"
            )
        except (zipfile.BadZipFile, ValueError) as e:
            print(f"Error extracting zip file: {e}")
            # Don't usually retry on bad zip file
            break  # Exit retry loop
        except Exception as e:
            print(
                f"An unexpected error occurred (attempt {retries + 1}/{max_retries + 1}): {e}"
            )

        # If we got here, an error occurred
        retries += 1
        if retries <= max_retries:
            delay = initial_delay * (2**(retries - 1))
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print("Max retries reached. Download failed.")

    # If loop finished without returning True, it failed.
    # Clean up potentially partially created/extracted directory (use with caution)
    if TARGET_ASSET_DIR.exists():
        print(
            f"Cleaning up potentially incomplete directory: {TARGET_ASSET_DIR}")
        # Consider safety before uncommenting shutil.rmtree
        # try:
        #     shutil.rmtree(TARGET_ASSET_DIR)
        # except OSError as e:
        #     print(f"Error during cleanup: {e}")
        # Ensure marker file doesn't exist if cleanup wasn't perfect
        if MARKER_FILE.exists():
            try:
                MARKER_FILE.unlink()
            except OSError:
                pass  # Ignore error if marker file deletion fails

    return False  # Indicate failure


if __name__ == "__main__":
    # Allow standalone execution for manual downloads
    print("Running test data download script directly...")
    if not download_and_extract():
        print("Script finished with errors.")
        sys.exit(1)
    else:
        print("Script finished successfully.")
        sys.exit(0)
