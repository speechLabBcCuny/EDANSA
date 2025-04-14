"""Project-wide pytest configuration file."""

import pytest
import logging


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption("--log-debug",
                     action="store_true",
                     default=False,
                     help="Enable DEBUG level logging for app and libraries.")


@pytest.fixture(scope='session')
def debug_mode(request):
    """Fixture to check if --log-debug was passed."""
    return request.config.getoption("--log-debug")


@pytest.fixture(autouse=True, scope='session')
def setup_logging(debug_mode):
    """Set up logging based on debug mode for the entire test session.

    Sets root logger and console level based on --log-debug.
    If --log-debug is NOT passed, explicitly sets noisy libraries to WARNING.
    """
    app_level = logging.DEBUG if debug_mode else logging.INFO
    third_party_level = logging.WARNING

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(app_level)  # Set root level based on flag

    # Configure the console handler (ensure only one is added)
    existing_handlers = [
        h for h in root_logger.handlers
        if isinstance(h, logging.StreamHandler) and
        hasattr(h, '__fixture_added__')
    ]

    if not existing_handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            # Use a slightly simpler default format
            '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        console_handler.__fixture_added__ = True
        root_logger.addHandler(console_handler)
    else:
        console_handler = existing_handlers[0]

    # Set the handler's level
    console_handler.setLevel(app_level)

    # Conditionally set levels for noisy third-party libraries to WARNING
    if not debug_mode:
        noisy_libraries = [
            'torch',
            'torchaudio',
            'torchvision',
            'ignite',
            'matplotlib',
            'git',
            'git.cmd',
            'numba',
            'h5py',
            'anyio',
        ]
        for lib_name in noisy_libraries:
            logging.getLogger(lib_name).setLevel(third_party_level)
        print(
            f"\\n--- Logging configured: App Level={logging.getLevelName(app_level)}, Third-Party Level={logging.getLevelName(third_party_level)} ---"
        )
    else:
        # When debug_mode is True, libraries inherit the DEBUG level from root
        print(
            f"\\n--- Logging configured: App Level={logging.getLevelName(app_level)}, Third-Party Libraries inherit DEBUG ---"
        )

    return app_level


# You can keep other project-wide fixtures or hooks here if needed.
