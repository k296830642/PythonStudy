#!/usr/bin/env python3
"""Benchmark InsightFace buffalo_sc and buffalo_l on CPU.

Example:
  python benchmark_insightface.py --config config.json --seconds 30 --sample-fps 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--video", type=Path)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--models", nargs="+", default=["buffalo_sc", "buffalo_l"])
    return parser.parse_args()


def memory_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return None


def load_video(config: dict, requested: Path | None) -> Path:
    if requested:
        return requested.resolve()
    input_dir = Path(config.get("input_dir", "input"))
    videos = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"})
    if not videos:
        raise SystemExit(f"No video found in {input_dir}")
    return videos[0].resolve()


def benchmark(model_name: str, video_path: Path, seconds: float, sample_fps: float) -> dict:
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install insightface onnxruntime psutil") from exc

    before = memory_mb()
    app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    loaded = memory_mb()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = max(0.1, float(capture.get(cv2.CAP_PROP_FPS)))
    duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    sample_count = max(1, int(min(seconds, duration) * sample_fps))
    elapsed: list[float] = []
    faces = 0
    try:
        for index in range(sample_count):
            timestamp = index / sample_fps
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            started = time.perf_counter()
            faces += len(app.get(frame))
            elapsed.append(time.perf_counter() - started)
    finally:
        capture.release()
    after = memory_mb()
    return {
        "model": model_name,
        "video": str(video_path),
        "samples": len(elapsed),
        "faces_detected": faces,
        "mean_ms": round(statistics.mean(elapsed) * 1000, 2) if elapsed else None,
        "p95_ms": round(sorted(elapsed)[max(0, int(len(elapsed) * 0.95) - 1)] * 1000, 2) if elapsed else None,
        "throughput_fps": round(1 / statistics.mean(elapsed), 2) if elapsed else None,
        "rss_before_mb": round(before, 1) if before is not None else None,
        "rss_after_load_mb": round(loaded, 1) if loaded is not None else None,
        "rss_after_run_mb": round(after, 1) if after is not None else None,
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    video = load_video(config, args.video)
    results = [benchmark(model, video, args.seconds, args.sample_fps) for model in args.models]
    print(json.dumps({"results": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
