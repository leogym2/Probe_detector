"""Parsing of the dataset's file naming convention.

Files follow the pattern:
    {device_id}_{sequence}_{frame}_{n}flight_{timestamp}_{cam}.jpg
e.g. E300PREMP00002_00725_216_1flight_300_2.jpg
"""

import re

FILENAME_RE = re.compile(
    r"^(?P<device>[A-Za-z0-9]+)_(?P<sequence>\d+)_(?P<frame>\d+)_"
    r"(?P<n>\d+)flight_(?P<timestamp>\d+)_(?P<cam>\d+)\.jpg$"
)


def parse_filename(file_name: str) -> dict:
    """Parse a dataset file name into its components.

    Raises ValueError if the file name does not match the expected pattern.
    """
    match = FILENAME_RE.match(file_name)
    if match is None:
        raise ValueError(f"File name does not match expected pattern: {file_name!r}")
    return match.groupdict()


def group_key(file_name: str) -> tuple[str, str]:
    """Return the (device, sequence) group a file belongs to.

    Frames sharing a group come from the same flight clip and are temporally
    close (same wall, same lighting) -- they must stay together on the same
    side of any train/val split.
    """
    parts = parse_filename(file_name)
    return parts["device"], parts["sequence"]
