"""
ROI keypoint inference using SLEAP models.

Adapted from: 2025-12-08-GoPro-roi-inference/00.run-inference.py

This module runs inference on videos to detect ROI keypoints (23 points
for horizontal T-maze), then selects the best frame from the first N frames.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from stride.config import DEFAULT_ROI_MODELS, STRICT_REQUIRED_23, RELAXED_REQUIRED_20
from stride.utils.parallel import run_parallel, find_videos


def run_roi_inference_batch(
    video_dir: Path,
    output_dir: Path,
    model_paths: Optional[list[Path]] = None,
    overwrite: bool = False,
    frames_to_check: int = 10,
    worker_id: int = 0,
    num_workers: int = 1,
) -> dict:
    """
    Run ROI inference on all videos in a directory.

    For multi-GPU parallelism, run multiple processes with different worker_id values:
        # Terminal 1 - GPU 0
        tmaze roi-inference -i /videos -o /output --gpu 0 --worker-id 0 --num-workers 2

        # Terminal 2 - GPU 1
        tmaze roi-inference -i /videos -o /output --gpu 1 --worker-id 1 --num-workers 2

    Args:
        video_dir: Directory containing input videos
        output_dir: Directory for output .slp files
        model_paths: List of model paths [centroid, centered_instance]
        overwrite: Whether to overwrite existing outputs
        frames_to_check: Number of frames to check for best ROI
        worker_id: Worker ID for parallel processing (0-indexed)
        num_workers: Total number of parallel workers

    Returns:
        Summary dict with counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use default models if not specified
    if model_paths is None:
        model_paths = [str(p) for p in DEFAULT_ROI_MODELS if p.exists()]

    if not model_paths:
        raise ValueError("No ROI models found. Please specify --model paths.")

    all_videos = find_videos(video_dir, pattern="*.mp4")

    # Partition videos across workers - each worker processes every Nth video
    if num_workers > 1:
        videos = [v for idx, v in enumerate(all_videos) if idx % num_workers == worker_id]
        print(f"Worker {worker_id + 1}/{num_workers}: processing {len(videos)}/{len(all_videos)} videos")
    else:
        videos = all_videos

    results = []
    for i, video_path in enumerate(videos, 1):
        result = run_single_video_inference(
            video_path=video_path,
            output_dir=output_dir,
            model_paths=model_paths,
            overwrite=overwrite,
            frames_to_check=frames_to_check,
        )
        results.append(result)
        print(f"[{i}/{len(videos)}] {result.get('status', 'error').upper():5} {video_path.name}")

    # Summarize
    passed = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") not in ("ok", "skip"))
    skipped = sum(1 for r in results if r.get("status") == "skip")

    return {
        "total": len(videos),
        "total_all": len(all_videos),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "results": results,
    }


def run_single_video_inference(
    video_path: Path,
    output_dir: Path,
    model_paths: list[str],
    overwrite: bool = False,
    frames_to_check: int = 10,
) -> dict:
    """
    Run ROI inference on a single video.

    Returns:
        Result dict with status
    """
    output_path = output_dir / f"{video_path.stem}.preds.v2.best1.slp"

    if output_path.exists() and not overwrite:
        return {"status": "skip", "video": video_path.name, "reason": "exists"}

    try:
        import sleap_io as sio
        from sleap_nn.predict import run_inference

        # Run inference on first N frames
        tmp_path = output_dir / f"{video_path.stem}.tmp.slp"

        run_inference(
            data_path=str(video_path),
            model_paths=model_paths,
            output_path=str(tmp_path),
            frames=list(range(frames_to_check)),
        )

        # Load predictions and select best frame
        labels = sio.load_file(str(tmp_path))
        best_idx, best_lf = select_best_frame(labels, max_check=frames_to_check)

        if best_lf is None:
            return {"status": "fail", "video": video_path.name, "reason": "no_valid_frame"}

        # Save single-frame result
        labels_out = sio.Labels(
            labeled_frames=[best_lf],
            videos=labels.videos,
            skeletons=labels.skeletons if hasattr(labels, "skeletons") else [labels.skeleton],
        )
        sio.save_file(labels_out, str(output_path))

        # Clean up temp file
        tmp_path.unlink(missing_ok=True)

        return {"status": "ok", "video": video_path.name, "output": str(output_path)}

    except Exception as e:
        return {"status": "error", "video": video_path.name, "error": str(e)}


def select_best_frame(labels, max_check: int = 10):
    """
    Select best frame from predictions using strict (23) then relaxed (20) criteria.

    Returns:
        Tuple of (frame_index, labeled_frame) or (video_name, None) if none valid
    """
    import re

    def _norm(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def get_skeleton_node_names(pred_labels):
        if hasattr(pred_labels, "skeleton") and hasattr(pred_labels.skeleton, "nodes"):
            return [n.name for n in pred_labels.skeleton.nodes]
        try:
            lf0 = pred_labels[0]
            if lf0.instances and hasattr(lf0.instances[0], "skeleton"):
                return [n.name for n in lf0.instances[0].skeleton.nodes]
        except Exception:
            pass
        return []

    skeleton_nodes = get_skeleton_node_names(labels)
    if not skeleton_nodes:
        return "UNKNOWN", None

    exact = {name: i for i, name in enumerate(skeleton_nodes)}
    norm = {_norm(name): i for i, name in enumerate(skeleton_nodes)}

    def lookup_idx(name):
        if name in exact:
            return exact[name]
        return norm.get(_norm(name))

    def get_point_xy(instance, idx):
        if hasattr(instance, "points"):
            pts = instance.points
            if 0 <= idx < len(pts):
                rec = pts[idx]
                if hasattr(rec, "dtype") and hasattr(rec.dtype, "names") and "xy" in rec.dtype.names:
                    xy = rec["xy"]
                    arr = np.asarray(xy, dtype=np.float64).reshape(-1)
                    if arr.size >= 2:
                        return [float(arr[0]), float(arr[1])]
                try:
                    arr = np.asarray(rec, dtype=np.float64).reshape(-1)
                    if arr.size >= 2:
                        return [float(arr[0]), float(arr[1])]
                except Exception:
                    pass
        return [np.nan, np.nan]

    def is_finite_xy(pt):
        arr = np.asarray(pt, dtype=np.float64).reshape(-1)
        return arr.size >= 2 and np.isfinite(arr[:2]).all()

    def eval_frame(lf, required_nodes):
        if not getattr(lf, "instances", None):
            return False, -1.0
        inst = lf.instances[0]
        for name in required_nodes:
            j = lookup_idx(name)
            if j is None:
                return False, -1.0
            xy = get_point_xy(inst, j)
            if not is_finite_xy(xy):
                return False, -1.0
        inst_score = float(getattr(inst, "score", 0.0) or 0.0)
        return True, inst_score

    # Try strict first, then relaxed
    best_strict = None
    best_relaxed = None
    nframes = min(max_check, len(labels))

    for idx in range(nframes):
        lf = labels[idx]

        ok_s, score_s = eval_frame(lf, STRICT_REQUIRED_23)
        if ok_s:
            if best_strict is None or score_s > best_strict[0]:
                best_strict = (score_s, idx, lf)
            continue

        ok_r, score_r = eval_frame(lf, RELAXED_REQUIRED_20)
        if ok_r:
            if best_relaxed is None or score_r > best_relaxed[0]:
                best_relaxed = (score_r, idx, lf)

    if best_strict is not None:
        return best_strict[1], best_strict[2]
    if best_relaxed is not None:
        return best_relaxed[1], best_relaxed[2]

    return "NO_VALID_FRAME", None
