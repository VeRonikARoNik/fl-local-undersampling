"""List directory contents in sorted order, as Windows (NTFS) does.

The split scripts read every data file in a data set folder and concatenate the files in the order returned by
pathlib's glob. The original experiment ran on Windows, where this order is alphabetical (case-insensitive). On Linux
and macOS the order is arbitrary, which changes the row order of data sets distributed as several files (SVS: nine
files, CM: a training and a testing file) and therefore their cross-validation folds and client sets.
Importing this module before the split scripts restores the Windows order on any system.
"""
import pathlib


def _sorted(fn):
    def wrapper(self, *args, **kwargs):
        return iter(sorted(fn(self, *args, **kwargs), key=lambda p: str(p).upper()))
    return wrapper


if not getattr(pathlib.Path, "_windows_order", False):
    pathlib.Path.glob = _sorted(pathlib.Path.glob)
    pathlib.Path.rglob = _sorted(pathlib.Path.rglob)
    pathlib.Path._windows_order = True
