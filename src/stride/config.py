"""
Configuration management for the T-maze pipeline.

This module handles paths, thresholds, and pipeline settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class PipelineConfig:
    """Configuration for the T-maze analysis pipeline."""

    # Input/Output paths
    video_dir: Path = field(default_factory=lambda: Path("."))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    meta_csv: Optional[Path] = None

    # Model paths (for inference stages)
    pose_model_paths: list[Path] = field(default_factory=list)
    roi_model_paths: list[Path] = field(default_factory=list)

    # Video settings
    fps_default: float = 120.0
    video_extensions: tuple[str, ...] = (".mp4", ".avi")

    # Distortion check thresholds
    distortion_line_threshold: float = 2.0  # pixels
    distortion_spacing_threshold: float = 0.05  # coefficient of variation
    distortion_reproj_threshold: float = 5.0  # pixels

    # ROI inference settings
    roi_frames_to_check: int = 10
    roi_strict_keypoints: int = 23
    roi_relaxed_keypoints: int = 20

    # Decision analysis thresholds
    min_valid_sec: float = 0.5  # warm-up window
    seg4_leave_threshold: float = 30.0  # % coverage to consider "left start"
    seg4_leave_k_frames: int = 5
    strict_dwell_threshold: float = 95.0  # % for strict arm entry
    soft_dwell_threshold: float = 60.0  # % for soft arm entry
    min_dwell_run: int = 3  # consecutive frames
    dwell_gap_allow: int = 1  # bridge short gaps
    min_body_area: float = 1500.0  # pixels^2

    # Gait analysis settings
    gait_confidence_threshold: float = 0.3
    gait_keypoints: tuple[str, ...] = (
        "snout", "mouth",
        "forepawR2", "forepawR1", "forepawL1", "forepawL2",
        "hindpawR2", "hindpawR1", "hindpawL2", "hindpawL1",
        "tailbase", "tail1", "tail2", "tail3", "tailtip",
    )

    # Parallel processing
    n_workers: int = 4

    # Checkpoint file
    checkpoint_file: Optional[Path] = None

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.video_dir, str):
            self.video_dir = Path(self.video_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.meta_csv, str):
            self.meta_csv = Path(self.meta_csv)
        if isinstance(self.checkpoint_file, str):
            self.checkpoint_file = Path(self.checkpoint_file)

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        data = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "roi_slp").mkdir(exist_ok=True)
        (self.output_dir / "roi_yml").mkdir(exist_ok=True)
        (self.output_dir / "decisions").mkdir(exist_ok=True)
        (self.output_dir / "gait").mkdir(exist_ok=True)


# Default model paths for the GoPro T-maze setup
# ROI models detect 23 keypoints for T-maze regions (arm_right, junction, arm_left, segment1-4)
DEFAULT_ROI_MODELS = [
    Path("/root/vast/leo/2025-11-26-GoPro_roi_train/models/251126_191944.centroid.n=50"),
    Path("/root/vast/leo/2025-11-26-GoPro_roi_train/models_centered_scale0.5/centered_instance.n=50"),
]

# Pose models detect mouse body keypoints (snout, paws, tail, etc.)
# Using n=503 models (latest, trained on 503 frames)
DEFAULT_POSE_MODELS = [
    Path("/root/vast/leo/2025-11-04-GoPro_Tmaze_train/models_500/251204_003259.centroid.n=503"),
    Path("/root/vast/leo/2025-11-04-GoPro_Tmaze_train/models_500/centered_instance.n=503"),
]

# 23 required keypoints for horizontal T-maze ROI
STRICT_REQUIRED_23 = [
    "arm_right.top_left", "arm_right.top_right", "arm_right.centroid",
    "junction.top_left", "junction.top_right", "junction.bottom_left",
    "junction.bottom_right", "junction.centroid",
    "arm_left.bottom_left", "arm_left.bottom_right", "arm_left.centroid",
    "segment1.top_left", "segment1.bottom_left", "segment1.centroid",
    "segment2.top_left", "segment2.bottom_left", "segment2.centroid",
    "segment3.top_left", "segment3.bottom_left", "segment3.centroid",
    "segment4.top_left", "segment4.bottom_left", "segment4.centroid",
]

# Relaxed version: skip segment1 (20 nodes)
RELAXED_REQUIRED_20 = [
    "arm_right.top_left", "arm_right.top_right", "arm_right.centroid",
    "junction.top_left", "junction.top_right", "junction.bottom_left",
    "junction.bottom_right", "junction.centroid",
    "arm_left.bottom_left", "arm_left.bottom_right", "arm_left.centroid",
    "segment2.top_left", "segment2.bottom_left", "segment2.centroid",
    "segment3.top_left", "segment3.bottom_left", "segment3.centroid",
    "segment4.top_left", "segment4.bottom_left", "segment4.centroid",
]

# ROI color palette
ROI_PALETTE = {
    "arm_right": "#2ca02c",
    "junction": "#98df8a",
    "arm_left": "#d62728",
    "segment4": "#ff9896",
    "segment3": "#9467bd",
    "segment2": "#c5b0d5",
    "segment1": "#8c564b",
}

# Physical T-maze dimensions (cm)
MAZE_DIMENSIONS = {
    "arm_length_cm": 27.5,  # Length of each arm (L/R)
    "maze_width_cm": 10.0,  # Width of maze corridor
    "stem_length_cm": 55.0,  # Total length of stem (segment1-4)
}


def calculate_px_per_cm(roi_yaml_path: Path) -> dict:
    """
    Calculate pixels per cm from ROI polygon dimensions.

    Uses min(arm_width, arm_height) to estimate px_per_cm based on known arm
    length of 27.5 cm. This handles both horizontal and vertical T-maze
    orientations correctly.

    Note: Different T-maze orientations (horizontal vs vertical) will have
    different coordinate systems. This function uses the smaller dimension
    of the arm polygon to ensure consistent calibration regardless of
    orientation.

    Returns:
        Dict with px_per_cm, cm_per_px, and calibration details
    """
    import yaml
    import numpy as np

    with open(roi_yaml_path) as f:
        y = yaml.safe_load(f) or {}

    # Find arm_right polygon
    arm_right_coords = None
    junction_coords = None
    for roi in y.get("rois", []):
        if roi.get("name") == "arm_right":
            arm_right_coords = roi.get("coordinates", [])
        if roi.get("name") == "junction":
            junction_coords = roi.get("coordinates", [])

    if not arm_right_coords or len(arm_right_coords) < 4:
        return {"px_per_cm": None, "cm_per_px": None, "error": "arm_right not found"}

    # Calculate arm dimensions in pixels
    # Use min(width, height) to handle both horizontal and vertical T-maze orientations
    pts = np.array(arm_right_coords)
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    arm_width_px = x_coords.max() - x_coords.min()
    arm_height_px = y_coords.max() - y_coords.min()

    # Use the smaller dimension as the arm length reference
    # This ensures consistent calibration regardless of T-maze orientation
    arm_length_px = min(arm_width_px, arm_height_px)
    orientation = "horizontal" if arm_width_px > arm_height_px else "vertical"

    # Calculate junction width for maze width calibration
    if junction_coords and len(junction_coords) >= 4:
        jpts = np.array(junction_coords)
        jx_coords = jpts[:, 0]
        jy_coords = jpts[:, 1]
        junction_width_px = jx_coords.max() - jx_coords.min()
        junction_height_px = jy_coords.max() - jy_coords.min()
    else:
        junction_width_px = None
        junction_height_px = None

    # Calculate px_per_cm
    arm_length_cm = MAZE_DIMENSIONS["arm_length_cm"]
    px_per_cm = arm_length_px / arm_length_cm

    result = {
        "px_per_cm": float(px_per_cm),
        "cm_per_px": 1.0 / px_per_cm,
        "arm_length_px": float(arm_length_px),
        "arm_width_px": float(arm_width_px),
        "arm_height_px": float(arm_height_px),
        "arm_length_cm": arm_length_cm,
        "tmaze_orientation": orientation,
        "calibration_method": "arm_min_dimension",
    }

    if junction_width_px is not None:
        maze_width_cm = MAZE_DIMENSIONS["maze_width_cm"]
        # Use smaller junction dimension for width calibration
        junction_min_px = min(junction_width_px, junction_height_px)
        px_per_cm_width = junction_min_px / maze_width_cm
        result["junction_width_px"] = float(junction_width_px)
        result["junction_height_px"] = float(junction_height_px)
        result["maze_width_cm"] = maze_width_cm
        result["px_per_cm_from_width"] = float(px_per_cm_width)

    return result
