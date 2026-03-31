"""
Checkpoint management for pipeline recovery.

Tracks progress through pipeline stages for resumption after interruption.
"""

from pathlib import Path
from typing import Optional
import re


# Checkpoint names matching progress.log format
CHECKPOINT_NAMES = [
    "VIDEO_SCAN",
    "DISTORTION_CHECK",
    "UNDISTORTION",
    "POSE_INFERENCE",
    "ROI_INFERENCE",
    "SLP_TO_YAML",
    "DECISION_ANALYSIS",
    "GAIT_ANALYSIS",
]


def read_checkpoints(checkpoint_file: Path) -> dict[str, str]:
    """
    Read checkpoint statuses from progress.log file.

    Args:
        checkpoint_file: Path to progress.log

    Returns:
        Dictionary mapping checkpoint names to status strings
    """
    checkpoints = {}

    if not checkpoint_file.exists():
        return {name: "pending" for name in CHECKPOINT_NAMES}

    content = checkpoint_file.read_text()

    # Parse CHECKPOINT_XXX = "status" lines
    pattern = r'CHECKPOINT_(\w+)\s*=\s*["\'](\w+)["\']'
    for match in re.finditer(pattern, content):
        name = match.group(1)
        status = match.group(2)
        checkpoints[name] = status

    # Fill in any missing checkpoints
    for name in CHECKPOINT_NAMES:
        if name not in checkpoints:
            checkpoints[name] = "pending"

    return checkpoints


def update_checkpoint(
    checkpoint_file: Path,
    checkpoint_name: str,
    status: str,
) -> None:
    """
    Update a checkpoint status in progress.log.

    Args:
        checkpoint_file: Path to progress.log
        checkpoint_name: Name of checkpoint (e.g., "VIDEO_SCAN")
        status: New status ("pending", "running", "completed", "failed")
    """
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    content = checkpoint_file.read_text()

    # Pattern to match the checkpoint line
    pattern = rf'(CHECKPOINT_{checkpoint_name}\s*=\s*["\'])\w+(["\'])'
    replacement = rf'\g<1>{status}\g<2>'

    new_content = re.sub(pattern, replacement, content)

    checkpoint_file.write_text(new_content)


def get_next_pending(checkpoint_file: Path) -> Optional[str]:
    """
    Get the next pending checkpoint to run.

    Args:
        checkpoint_file: Path to progress.log

    Returns:
        Name of next pending checkpoint, or None if all complete
    """
    checkpoints = read_checkpoints(checkpoint_file)

    for name in CHECKPOINT_NAMES:
        if checkpoints.get(name) == "pending":
            return name

    return None


def mark_running(checkpoint_file: Path, checkpoint_name: str) -> None:
    """Mark a checkpoint as running."""
    update_checkpoint(checkpoint_file, checkpoint_name, "running")


def mark_completed(checkpoint_file: Path, checkpoint_name: str) -> None:
    """Mark a checkpoint as completed."""
    update_checkpoint(checkpoint_file, checkpoint_name, "completed")


def mark_failed(checkpoint_file: Path, checkpoint_name: str) -> None:
    """Mark a checkpoint as failed."""
    update_checkpoint(checkpoint_file, checkpoint_name, "failed")
