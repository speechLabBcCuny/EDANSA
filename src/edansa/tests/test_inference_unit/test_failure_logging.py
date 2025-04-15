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
        self.log_file_path = self.output_folder / 'failed_files.log'

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
                                               encoding='utf-8')
        mock_open_func().write.assert_called_once_with(expected_log_line)

    @patch('edansa.inference.datetime')
    def test_log_file_appending(self, mock_datetime):
        """Test that entries are appended to an existing log file."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        timestamp_str = mock_now.strftime('%Y-%m-%d %H:%M:%S')

        # Create initial content
        initial_content = " preexisting log entry\n"
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(initial_content)

        audio_file = Path('/data/audio/another_file.flac')
        error_msg = "Inference failed"
        expected_log_line = f"{timestamp_str}\t{audio_file}\t{error_msg}\n"

        # Call the function
        _log_failed_file_to_output_folder(audio_file, error_msg,
                                          self.output_folder)

        # Read file content and assert
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertEqual(content, initial_content + expected_log_line)

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

        expected_log_line = f"{timestamp_str}\t{audio_file_rel}\t{error_msg}\n"

        # Call the function with input_data_root
        _log_failed_file_to_output_folder(audio_file_abs, error_msg,
                                          self.output_folder, input_root)

        # Read file content and assert
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, expected_log_line)

    @patch('edansa.inference.datetime')
    def test_absolute_path_fallback_no_root(self, mock_datetime):
        """Test that absolute path is used when input_data_root is None."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        timestamp_str = mock_now.strftime('%Y-%m-%d %H:%M:%S')

        audio_file_abs = Path('/different/drive/data/file.mp3')
        error_msg = "Format error"
        expected_log_line = f"{timestamp_str}\t{audio_file_abs}\t{error_msg}\n"

        # Call the function without input_data_root
        _log_failed_file_to_output_folder(audio_file_abs, error_msg,
                                          self.output_folder,
                                          None)  # Explicitly None

        # Read file content and assert
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, expected_log_line)

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
        expected_log_line = f"{timestamp_str}\t{audio_file_abs}\t{error_msg}\n"
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, expected_log_line)

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
        # Check that the warning message contains the exception details
        call_args, _ = mock_log_warning.call_args
        self.assertIn("Failed to write to failed_files.log", call_args[0])
        # Check the content of the first argument for the error string
        self.assertIn("Disk full simulation", call_args[0])


if __name__ == '__main__':
    unittest.main()
