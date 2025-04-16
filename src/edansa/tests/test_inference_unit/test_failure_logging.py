#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Test suite for failure logging.
import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import tempfile
import logging
from datetime import datetime
import os
import csv

# Assuming the function is in src.edansa.inference
# Adjust the import path based on your project structure
from edansa.inference import _log_failed_file_to_output_folder

# Disable logging messages below WARNING during tests
logging.disable(logging.CRITICAL)


class TestFailureLogging(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for test outputs."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_folder = Path(self.test_dir.name)
        self.log_file_path = self.output_folder / 'failed_files.csv'

    def tearDown(self):
        """Clean up the temporary directory."""
        self.test_dir.cleanup()
        logging.disable(logging.NOTSET)  # Re-enable logging

    @patch('edansa.inference.datetime')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_log_file_creation_and_write(self, mock_mkdir, mock_open_func,
                                         mock_datetime):
        """Test that the log file is created and the first entry is written correctly."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        timestamp_str = mock_now.strftime('%Y-%m-%d %H:%M:%S')

        audio_file = Path('/data/audio/test_file.wav')
        error_msg = "Failed to load"
        expected_log_line = f"{timestamp_str}\t{audio_file}\t{error_msg}\n"

        # Call the function
        _log_failed_file_to_output_folder(audio_file, error_msg,
                                          self.output_folder)

        # Assertions
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_open_func.assert_called_once_with(self.log_file_path,
                                               'a',
                                               newline='',
                                               encoding='utf-8')
        # Check that writerow was called correctly for header and data
        # We access the mock_calls on the file handle mock
        write_calls = mock_open_func().write.call_args_list
        # Depending on how mock_open and csv.writer interact, write might be called
        # with bytes or strings. We check for the data content.
        # Note: csv.writer often adds CRLF (\r\n) line endings.
        # Construct expected CSV rows (adjust quoting/newlines based on actual csv output)
        expected_header_write = '"Timestamp","FilePath","ErrorMessage"\r\n'
        expected_data_write = f'"{timestamp_str}","{audio_file}","{error_msg}"\r\n'

        # Verify that write was called at least twice (header + data)
        self.assertGreaterEqual(len(write_calls), 2)

        # Verify the content of the calls (flexible check)
        # Convert call args to string for easier comparison
        call_args_str = [str(call[0][0]) for call in write_calls]
        self.assertIn(expected_header_write, call_args_str)
        self.assertIn(expected_data_write, call_args_str)

    @patch('edansa.inference.datetime')
    def test_log_file_appending(self, mock_datetime):
        """Test that entries are appended to an existing log file."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        timestamp_str = mock_now.strftime('%Y-%m-%d %H:%M:%S')

        # Create initial header and content for CSV
        header = '"Timestamp","FilePath","ErrorMessage"\n'
        initial_content_row = [
            '2022-12-31 23:59:59', '/data/old_file.wav', 'Old error'
        ]
        with open(self.log_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Timestamp", "FilePath", "ErrorMessage"])
            writer.writerow(initial_content_row)

        audio_file = Path('/data/audio/another_file.flac')
        error_msg = "Inference failed"
        expected_new_log_row = [timestamp_str, str(audio_file), error_msg]

        # Call the function
        _log_failed_file_to_output_folder(audio_file, error_msg,
                                          self.output_folder)

        # Read file content and assert rows
        with open(self.log_file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], ["Timestamp", "FilePath", "ErrorMessage"])
        self.assertEqual(rows[1], initial_content_row)
        self.assertEqual(rows[2], expected_new_log_row)

    @patch('edansa.inference.datetime')
    def test_relative_path_calculation(self, mock_datetime):
        """Test that relative path is used when input_data_root is provided."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        timestamp_str = mock_now.strftime('%Y-%m-%d %H:%M:%S')

        # Simulate a realistic structure
        input_root = Path('/project/data/recordings')
        audio_file_abs = input_root / 'site_a' / '20230101_120000.wav'
        audio_file_rel = Path(
            'site_a') / '20230101_120000.wav'  # Expected relative
        error_msg = "Clipping error"

        expected_log_row = [timestamp_str, str(audio_file_rel), error_msg]

        # Call the function with input_data_root
        _log_failed_file_to_output_folder(audio_file_abs, error_msg,
                                          self.output_folder, input_root)

        # Read file content and assert rows
        with open(self.log_file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ["Timestamp", "FilePath", "ErrorMessage"])
        self.assertEqual(rows[1], expected_log_row)

    @patch('edansa.inference.datetime')
    def test_absolute_path_fallback_no_root(self, mock_datetime):
        """Test that absolute path is used when input_data_root is None."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        timestamp_str = mock_now.strftime('%Y-%m-%d %H:%M:%S')

        audio_file_abs = Path('/different/drive/data/file.mp3')
        error_msg = "Format error"
        expected_log_row = [timestamp_str, str(audio_file_abs), error_msg]

        # Call the function without input_data_root
        _log_failed_file_to_output_folder(audio_file_abs, error_msg,
                                          self.output_folder,
                                          None)  # Explicitly None

        # Read file content and assert
        with open(self.log_file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ["Timestamp", "FilePath", "ErrorMessage"])
        self.assertEqual(rows[1], expected_log_row)

    @patch('edansa.inference.datetime')
    def test_absolute_path_fallback_relative_error(self, mock_datetime):
        """Test absolute path use if relative_to fails (e.g., different drives)."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        timestamp_str = mock_now.strftime('%Y-%m-%d %H:%M:%S')

        # Simulate paths that would cause relative_to to fail (e.g., different roots/drives)
        # On Unix, just different absolute paths might still work, depending on commonality
        # Let's simulate the ValueError directly for robustness
        audio_file_abs = Path('/other/root/audio.wav')
        input_root = Path('/project/data')
        error_msg = "Relative path error simulation"

        # Make Path.relative_to raise ValueError when called
        with patch.object(Path,
                          'relative_to',
                          side_effect=ValueError("Cannot make relative")):
            _log_failed_file_to_output_folder(audio_file_abs, error_msg,
                                              self.output_folder, input_root)

        # Assert that the original absolute path was logged
        expected_log_row = [timestamp_str, str(audio_file_abs), error_msg]
        with open(self.log_file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ["Timestamp", "FilePath", "ErrorMessage"])
        self.assertEqual(rows[1], expected_log_row)

    @patch('builtins.open', new_callable=mock_open)
    @patch('logging.warning')
    def test_graceful_failure_on_log_write_error(self, mock_log_warning,
                                                 mock_open_func):
        """Test that a warning is logged if writing to the log file fails."""
        # Configure the mock open to raise IOError on write
        mock_open_func.return_value.write.side_effect = IOError(
            "Disk full simulation")

        audio_file = Path('/data/audio/problem_file.wav')
        error_msg = "Original processing error"

        # Call the function - expect it not to raise the IOError
        try:
            _log_failed_file_to_output_folder(audio_file, error_msg,
                                              self.output_folder)
        except IOError:
            self.fail(
                "_log_failed_file_to_output_folder raised an IOError unexpectedly."
            )

        # Assert that logging.warning was called
        mock_log_warning.assert_called_once()
        # Check that the warning message contains the exception details and correct filename
        call_args, _ = mock_log_warning.call_args
        self.assertIn("Failed to write to failed_files.csv", call_args[0])
        # Check the content of the first argument for the error string
        self.assertIn("Disk full simulation", call_args[0])


if __name__ == '__main__':
    unittest.main()
