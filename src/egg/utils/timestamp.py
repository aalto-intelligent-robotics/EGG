from datetime import datetime
import logging

from egg.utils.logger import getLogger
from egg.utils.geometry import Odometry, Position


logger: logging.Logger = getLogger(
    name=__name__,
    consoleLevel=logging.INFO,
    fileLevel=logging.DEBUG,
    log_file="utils/timestamp.log",
)


def ns_to_datetime(nanoseconds: int) -> datetime:
    seconds = nanoseconds // 1_000_000_000
    dt = datetime.fromtimestamp(seconds)
    return dt


def str_to_datetime(date_string: str) -> datetime:
    date_format = "%Y-%m-%d %H:%M:%S"
    datetime_object = datetime.strptime(date_string, date_format)
    return datetime_object


def datetime_to_ns(dt: datetime) -> int:
    timestamp_seconds = dt.timestamp()
    nanoseconds_part = dt.microsecond * 1_000
    nanoseconds_total = timestamp_seconds * 1_000_000_000 + nanoseconds_part
    return int(nanoseconds_total)


def print_timestamped_position(timestamped_position: dict[int, Position]) -> str:
    output_str = "\n"
    for timestamp_ns, pos in timestamped_position.items():
        output_str += f"\t{ns_to_datetime(timestamp_ns)}: {pos}\n"
    return output_str


def print_timestamped_observation_odom(
    timestamped_observation_odom: dict[int, Odometry],
    first_only: bool = True,
) -> str:
    output_str = "\n"
    for timestamp_ns, odom in timestamped_observation_odom.items():
        output_str += f"\t{ns_to_datetime(timestamp_ns)}:" + f"\n\t- Odom: {odom}\n"
        if first_only:
            break
    return output_str
