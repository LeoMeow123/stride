"""
Parallel processing utilities.

Provides a generic framework for batch processing videos.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Any, Optional
import logging


def run_parallel(
    items: list[Any],
    worker_fn: Callable[[Any], dict],
    n_workers: int = 4,
    desc: str = "Processing",
    logger: Optional[logging.Logger] = None,
) -> list[dict]:
    """
    Run a worker function on items in parallel.

    Args:
        items: List of items to process
        worker_fn: Function that takes an item and returns a dict with 'status' key
        n_workers: Number of parallel workers
        desc: Description for progress logging
        logger: Optional logger instance

    Returns:
        List of result dictionaries from worker_fn
    """
    log = logger or logging.getLogger(__name__)
    total = len(items)

    if total == 0:
        log.warning(f"{desc}: No items to process")
        return []

    log.info(f"{desc}: {total} items with {n_workers} workers")

    results = []
    n_done = 0
    counts = {"ok": 0, "error": 0, "skip": 0}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker_fn, item): item for item in items}

        for future in as_completed(futures):
            n_done += 1
            item = futures[future]

            try:
                result = future.result()
                results.append(result)

                status = result.get("status", "ok")
                if status == "ok":
                    counts["ok"] += 1
                elif status.startswith("skip"):
                    counts["skip"] += 1
                else:
                    counts["error"] += 1

            except Exception as e:
                log.error(f"Exception processing {item}: {e}")
                counts["error"] += 1
                results.append({"status": "error", "item": item, "error": str(e)})

            # Progress logging
            if n_done % 50 == 0 or n_done == total:
                log.info(
                    f"{desc}: {n_done}/{total} "
                    f"(ok={counts['ok']}, skip={counts['skip']}, error={counts['error']})"
                )

    log.info(
        f"{desc} complete: ok={counts['ok']}, skip={counts['skip']}, error={counts['error']}"
    )

    return results


def find_videos(
    video_dir: Path,
    extensions: tuple[str, ...] = (".mp4", ".avi"),
    pattern: Optional[str] = None,
) -> list[Path]:
    """
    Find video files in a directory.

    Args:
        video_dir: Directory to search
        extensions: Video file extensions
        pattern: Optional glob pattern (e.g., "Day*_*.mp4")

    Returns:
        Sorted list of video paths
    """
    if pattern:
        return sorted(video_dir.glob(pattern))

    videos = []
    for ext in extensions:
        videos.extend(video_dir.glob(f"*{ext}"))

    return sorted(set(videos))
