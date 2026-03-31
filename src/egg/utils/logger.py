import os
import datetime
import logging
from typing import Literal

from typing_extensions import override

DEFAULT_LOG_PATH: str = os.path.join(os.environ["HOME"], "logs")


class LogFormatter(logging.Formatter):
    """
    Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629.

    This class provides a logging formatter that adds colors to log messages
    based on their severity level.

    :ivar fmt: The format string used for log messages.
    :vartype fmt: str
    :ivar FORMATS: A dictionary mapping logging levels to their colored format strings.
    :vartype FORMATS: Dict[int, str]
    """

    COLOR_GREY: str = "\x1b[38;21m"
    COLOR_BLUE: str = "\x1b[38;5;39m"
    COLOR_YELLOW: str = "\x1b[38;5;226m"
    COLOR_RED: str = "\x1b[38;5;196m"
    COLOR_BOLD_RED: str = "\x1b[31;1m"
    COLOR_RESET: str = "\x1b[0m"

    def __init__(self, fmt: str):
        """
        Initialize the LogFormatter with the given format.

        :param fmt: The format string to be used for log messages.
        :type fmt: str
        """
        super().__init__()
        self.fmt: str = fmt
        self.FORMATS: dict[int, str] = {
            logging.DEBUG: self.COLOR_GREY
            + self.fmt
            + "%(message)s"
            + self.COLOR_RESET,
            logging.INFO: self.COLOR_BLUE + self.fmt + self.COLOR_RESET + "%(message)s",
            logging.WARNING: self.COLOR_YELLOW
            + self.fmt
            + "%(message)s"
            + self.COLOR_RESET,
            logging.ERROR: self.COLOR_RED + self.fmt + "%(message)s" + self.COLOR_RESET,
            logging.CRITICAL: self.COLOR_BOLD_RED
            + self.fmt
            + "%(message)s"
            + self.COLOR_RESET,
        }

    @override
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with the appropriate color based on its
        severity level.

        :param record: The log record to be formatted.
        :type record: logging.LogRecord

        :return: The formatted log message.
        :rtype: str
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def getLogger(
    name: str,
    consoleLevel: int = logging.INFO,
    fileLevel: int = logging.DEBUG,
    fmt: str = "%(asctime)s - %(name)s - [%(levelname)s]: ",
    log_path: str | None = None,
    log_file: str = f"default.log",
) -> logging.Logger:
    """
    Set up and return a logger with both console and file handlers.

    This function sets up a logger with specified name, console logging level, file
    logging level, format, log path, and log file name.
    It configures the logger to log messages both to the console (with colored
    formatting) and to a file.

    :param name: The name of the logger.
    :type name: str
    :param consoleLevel: The logging level for the console handler.
        Default is logging.INFO.
    :type consoleLevel: int
    :param fileLevel: The logging level for the file handler.
        Default is logging.DEBUG.
    :type fileLevel: int
    :param fmt: The format string to be used for log messages.
        Default is "%(asctime)s - %(name)s - [%(levelname)s]: ".
    :type fmt: str
    :param log_path: The path where log files will be saved.
        Default is a directory named with the current timestamp.
    :type log_path: str
    :param log_file: The name of the log file.
        Default is "default.log".
    :type log_file: str

    :return: The configured logger.
    :rtype: logging.Logger
    """
    # Set up logger
    if log_path is None:
        log_path = os.path.join(
            DEFAULT_LOG_PATH,
            f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        )
    log_full_path = os.path.join(log_path, log_file)
    os.makedirs(os.path.dirname(log_full_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # stdout handler for logging to the console
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(consoleLevel)
    stdout_handler.setFormatter(LogFormatter(fmt))
    # File handler for logging to a file
    file_handler = logging.FileHandler(log_full_path)
    file_handler.setLevel(fileLevel)
    file_handler.setFormatter(logging.Formatter(fmt + "%(message)s"))
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger
