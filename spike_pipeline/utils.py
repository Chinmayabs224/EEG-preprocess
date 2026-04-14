"""
Utility functions for the spike encoding pipeline.

Provides:
    - Logging setup (file + console)
    - CHB-MIT summary file parser (seizure annotations)
    - File manifest builder (recursive EDF discovery)
    - Timer context manager
"""

import os
import re
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import namedtuple

# ── Named tuple for file metadata ────────────────────────────────
FileInfo = namedtuple("FileInfo", [
    "path",               # Absolute path to .edf file
    "subject_id",         # e.g. "chb01"
    "filename",           # e.g. "chb01_03.edf"
    "seizure_intervals",  # list of (start_sec, end_sec) tuples
    "has_seizure",        # bool
])


# ═════════════════════════════════════════════════════════════════
# Logging
# ═════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str, name: str = "spike_pipeline") -> logging.Logger:
    """
    Configure a logger that writes to both console and a timestamped log file.

    Parameters
    ----------
    log_dir : str
        Directory to store log files.
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    # File handler — DEBUG level (captures everything)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler — INFO level (cleaner output)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging initialized → {log_file}")
    return logger


# ═════════════════════════════════════════════════════════════════
# CHB-MIT Summary File Parser
# ═════════════════════════════════════════════════════════════════

def parse_summary_file(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse a CHB-MIT ``chbXX-summary.txt`` file to extract seizure intervals.

    The summary files have a repeating structure like::

        File Name: chb01_03.edf
        File Start Time: 13:43:04
        File End Time: 14:43:04
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds

    For files with multiple seizures, the keys are numbered::

        Seizure 1 Start Time: ...
        Seizure 1 End Time: ...
        Seizure 2 Start Time: ...
        Seizure 2 End Time: ...

    Parameters
    ----------
    summary_path : str
        Path to the ``chbXX-summary.txt`` file.

    Returns
    -------
    dict
        Mapping from filename (e.g. ``"chb01_03.edf"``) to a list of
        ``(start_seconds, end_seconds)`` tuples.
    """
    seizure_map: Dict[str, List[Tuple[int, int]]] = {}

    if not os.path.isfile(summary_path):
        return seizure_map

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split into per-file blocks using "File Name:" as delimiter
    blocks = re.split(r"(?=File Name:)", content)

    for block in blocks:
        # Extract filename
        fname_match = re.search(r"File Name:\s*(.+\.edf)", block, re.IGNORECASE)
        if not fname_match:
            continue
        fname = fname_match.group(1).strip()

        # Extract number of seizures
        n_seiz_match = re.search(
            r"Number of Seizures in File:\s*(\d+)", block
        )
        n_seizures = int(n_seiz_match.group(1)) if n_seiz_match else 0

        intervals = []
        if n_seizures > 0:
            # Pattern matches both "Seizure Start Time:" and "Seizure N Start Time:"
            starts = re.findall(
                r"Seizure\s*\d*\s*Start Time:\s*(\d+)\s*seconds", block
            )
            ends = re.findall(
                r"Seizure\s*\d*\s*End Time:\s*(\d+)\s*seconds", block
            )
            for s, e in zip(starts, ends):
                intervals.append((int(s), int(e)))

        seizure_map[fname] = intervals

    return seizure_map


# ═════════════════════════════════════════════════════════════════
# File Manifest Builder
# ═════════════════════════════════════════════════════════════════

def get_file_manifest(data_dir: str,
                      logger: Optional[logging.Logger] = None
                      ) -> List[FileInfo]:
    """
    Recursively discover all ``.edf`` files across ``chbXX/`` subdirectories,
    parse their seizure annotations from ``chbXX-summary.txt``, and return
    a flat list of :class:`FileInfo` named tuples.

    Parameters
    ----------
    data_dir : str
        Root directory containing ``chb01/`` through ``chb24/`` folders.
    logger : logging.Logger, optional
        Logger for status messages.

    Returns
    -------
    list of FileInfo
        Sorted by (subject_id, filename).
    """
    manifest: List[FileInfo] = []
    data_path = Path(data_dir)

    if not data_path.is_dir():
        msg = f"Data directory not found: {data_dir}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)

    # Find all chbXX directories
    subject_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and re.match(r"chb\d+", d.name)
    ])

    if not subject_dirs:
        msg = f"No chbXX directories found in {data_dir}"
        if logger:
            logger.warning(msg)
        return manifest

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name  # e.g. "chb01"

        # Parse summary file for this subject
        summary_path = subj_dir / f"{subject_id}-summary.txt"
        seizure_map = parse_summary_file(str(summary_path))

        if logger:
            logger.debug(
                f"  {subject_id}: summary parsed, "
                f"{sum(len(v) for v in seizure_map.values())} seizure events"
            )

        # Find all .edf files in this subject directory
        edf_files = sorted(subj_dir.glob("*.edf"))

        for edf_path in edf_files:
            fname = edf_path.name
            intervals = seizure_map.get(fname, [])

            manifest.append(FileInfo(
                path=str(edf_path),
                subject_id=subject_id,
                filename=fname,
                seizure_intervals=intervals,
                has_seizure=len(intervals) > 0,
            ))

    if logger:
        n_seizure_files = sum(1 for f in manifest if f.has_seizure)
        logger.info(
            f"File manifest: {len(manifest)} EDF files across "
            f"{len(subject_dirs)} subjects, "
            f"{n_seizure_files} files with seizures"
        )

    return manifest


# ═════════════════════════════════════════════════════════════════
# Timer
# ═════════════════════════════════════════════════════════════════

class Timer:
    """
    Context manager for timing code blocks.

    Usage::

        with Timer("Loading data") as t:
            load_data()
        print(t.elapsed)  # seconds
    """

    def __init__(self, label: str = "", logger: Optional[logging.Logger] = None):
        self.label = label
        self.logger = logger
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        if self.logger and self.label:
            self.logger.info(f"⏱ {self.label} — started")
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.logger and self.label:
            self.logger.info(
                f"⏱ {self.label} — finished in {self.elapsed:.2f}s"
            )
