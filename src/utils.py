"""
Utility functions: device detection, reproducibility seeding, config loading,
and run metadata logging.

Used by every other script in the project.
"""

import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
import yaml


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect the best available device: cuda > mps > cpu.

    On Mac, the env var PYTORCH_ENABLE_MPS_FALLBACK=1 should be set
    externally (e.g. in the shell profile) so ops without MPS kernels
    fall back to CPU transparently.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[utils] Using device: {device}")
    return device


def is_mac_mps(device: torch.device) -> bool:
    """Return True if running on Apple MPS — used to disable fp16."""
    return device.type == "mps"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't have a separate manual_seed call
    print(f"[utils] Seed set to {seed}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(*yaml_paths: str) -> dict:
    """Load and merge YAML config files.  Later files override earlier ones.

    Usage:
        cfg = load_config("configs/base.yaml", "configs/bart.yaml")
    """
    merged: dict = {}
    for path in yaml_paths:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        merged.update(data)
        print(f"[utils] Loaded config: {path}")
    return merged


def load_decoding_config(path: str = "configs/decoding.yaml") -> dict:
    """Load the fixed decoding config (kept separate for clarity)."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[utils] Loaded decoding config: {path}")
    return cfg


# ---------------------------------------------------------------------------
# Run metadata / logging
# ---------------------------------------------------------------------------

def get_library_versions() -> dict:
    """Capture versions of key libraries for the run record."""
    import transformers
    import datasets
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "platform": platform.platform(),
        "device": str(get_device()),
    }
    try:
        import evaluate as ev
        versions["evaluate"] = ev.__version__
    except ImportError:
        pass
    return versions


def save_run_record(
    record: dict,
    results_dir: str = "results",
    filename: Optional[str] = None,
) -> Path:
    """Save a JSON run record with seed, versions, hardware, timing, etc.

    Args:
        record: Dictionary of run metadata and results.
        results_dir: Directory to save into (created if needed).
        filename: Optional custom filename; defaults to a timestamped name.

    Returns:
        Path to the saved JSON file.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"run_{ts}.json"

    out_path = results_path / filename
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"[utils] Run record saved to {out_path}")
    return out_path


def get_peak_memory_mb() -> float:
    """Return approximate peak memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)


class Timer:
    """Simple context-manager timer for measuring wall-clock duration."""

    def __init__(self, label: str = ""):
        self.label = label
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.label:
            print(f"[timer] {self.label}: {self.elapsed:.2f}s")
