"""Show the package's log messages without any logging setup by the user."""

import logging
import sys


class _Handler(logging.Handler):
    """Plain messages on stdout, next to what the Rust layer prints.

    Stands down when the root logger has handlers, so a user's own logging
    setup gets each record once. The stream is looked up on every record because
    notebooks and test runners swap `sys.stdout`.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if logging.getLogger().handlers:
            return
        stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
        print(self.format(record), file=stream, flush=True)


def attach(name: str) -> None:
    """Make INFO messages of the logger `name` visible by default.

    Args:
        name: The package's logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(_Handler())
