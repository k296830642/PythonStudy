#!/usr/bin/env python3
"""GPU fixed-face video clipping with cat-image masking for non-target faces."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


PROJECT_DIR = Path(__file__).resolve().parent

# Keep handles alive so libraries loaded by absolute path remain available to
# ONNX Runtime when it later loads its CUDA execution-provider extension.
PRELOADED_CUDA_LIBRARIES: list[ctypes.CDLL] = []
CUDA_LIBRARY_FILENAMES = (
    "libcudart.so.12",
    "libnvrtc.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcurand.so.10",
    "libcufft.so.11",
    "libcudnn.so.9",
)


@dataclass(frozen=True)
class Match:
    timestamp: float
    similarity: float


@dataclass(frozen=True)
class Appearance:
    first_match: float
    last_match: float
    maximum_similarity: float


@dataclass(frozen=True)
class Clip:
    start: float
    end: float
    maximum_similarity: float


@dataclass(frozen=True)
class DetectedFace:
    similarity: float
    bbox: tuple[int, int, int, int]
    is_target: bool
    target_source: str = "other"


@dataclass
class TargetFaceTracker:
    """Keep a confirmed target face unmasked through brief recognition dropouts."""

    max_gap_frames: int
    minimum_iou: float
    maximum_center_distance_ratio: float
    maximum_size_ratio: float
    last_bbox: tuple[int, int, int, int] | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    missing_frames: int = 0

    def resolve(self, faces: list[DetectedFace]) -> list[DetectedFace]:
        confirmed = [face for face in faces if face.is_target]
        if confirmed:
            selected = max(confirmed, key=lambda face: face.similarity)
            self._update(selected.bbox)
            return [replace(face, target_source="recognized") if face.is_target else face for face in faces]

        if self.last_bbox is None or self.missing_frames >= self.max_gap_frames:
            self.missing_frames += 1
            return faces

        predicted = self._predicted_bbox()
        candidates = [face for face in faces if self._matches_prediction(face.bbox, predicted)]
        if not candidates:
            self.missing_frames += 1
            return faces

        tracked = max(candidates, key=lambda face: self._association_score(face.bbox, predicted))
        self._update(tracked.bbox)
        return [
            replace(face, is_target=True, target_source="tracked") if face is tracked else face
            for face in faces
        ]

    def _update(self, bbox: tuple[int, int, int, int]) -> None:
        if self.last_bbox is not None:
            previous_x, previous_y = face_center(self.last_bbox)
            current_x, current_y = face_center(bbox)
            self.velocity = (current_x - previous_x, current_y - previous_y)
        self.last_bbox = bbox
        self.missing_frames = 0

    def _predicted_bbox(self) -> tuple[int, int, int, int]:
        if self.last_bbox is None:
            raise RuntimeError("Target tracker has no previous bounding box")
        dx, dy = self.velocity
        x1, y1, x2, y2 = self.last_bbox
        return (round(x1 + dx), round(y1 + dy), round(x2 + dx), round(y2 + dy))

    def _matches_prediction(self, bbox: tuple[int, int, int, int], predicted: tuple[int, int, int, int]) -> bool:
        predicted_width, predicted_height = face_size(predicted)
        width, height = face_size(bbox)
        size_ratio = max(width / predicted_width, predicted_width / width, height / predicted_height, predicted_height / height)
        if size_ratio > self.maximum_size_ratio:
            return False
        center_x, center_y = face_center(bbox)
        predicted_x, predicted_y = face_center(predicted)
        distance = math.hypot(center_x - predicted_x, center_y - predicted_y)
        diagonal = math.hypot(predicted_width, predicted_height)
        return bbox_iou(bbox, predicted) >= self.minimum_iou or distance <= diagonal * self.maximum_center_distance_ratio

    def _association_score(self, bbox: tuple[int, int, int, int], predicted: tuple[int, int, int, int]) -> float:
        center_x, center_y = face_center(bbox)
        predicted_x, predicted_y = face_center(predicted)
        diagonal = math.hypot(*face_size(predicted))
        distance_score = 1.0 - min(1.0, math.hypot(center_x - predicted_x, center_y - predicted_y) / max(1.0, diagonal))
        return bbox_iou(bbox, predicted) + distance_score


def face_size(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return (max(1.0, float(bbox[2] - bbox[0])), max(1.0, float(bbox[3] - bbox[1])))


def face_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
    second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
    return intersection / float(first_area + second_area - intersection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    parser.add_argument("--video", type=Path, help="Process one video only.")
    parser.add_argument("--scan-only", action="store_true", help="Find appearances without exporting clips.")
    return parser.parse_args()


def resolved_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read configuration: {exc}") from exc
    required = {
        "reference_faces_dir", "input_dir", "output_dir", "model_cache_dir", "model_pack", "onnx_providers",
        "require_gpu", "cuda_library_dirs",
        "recognition_mode", "mode_thresholds", "detection_size", "detection_threshold", "sample_fps",
        "before_seconds", "after_seconds", "clip_overlap_strategy", "minimum_appearance_seconds",
        "maximum_gap_seconds", "minimum_face_size", "mask_other_faces", "mask_image_path", "mask_scale",
        "mask_minimum_size", "mask_alpha", "video_codec", "video_preset", "video_cq", "audio_codec",
        "overwrite_output", "insightface_detection_log", "detection_log_interval_seconds",
        "detection_log_include_similarity", "target_tracking_enabled", "target_tracking_max_gap_frames",
        "target_tracking_minimum_iou", "target_tracking_maximum_center_distance_ratio",
        "target_tracking_maximum_size_ratio",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(missing)}")
    if config["recognition_mode"] not in config["mode_thresholds"]:
        raise ValueError("recognition_mode must exist in mode_thresholds")
    if config["clip_overlap_strategy"] not in {"allow", "trim"}:
        raise ValueError("clip_overlap_strategy must be allow or trim")
    if float(config["sample_fps"]) <= 0 or int(config["detection_size"]) <= 0:
        raise ValueError("sample_fps and detection_size must be greater than 0")
    if not 0 < float(config["mask_alpha"]) <= 1:
        raise ValueError("mask_alpha must be in (0, 1]")
    if float(config["detection_log_interval_seconds"]) <= 0:
        raise ValueError("detection_log_interval_seconds must be greater than 0")
    if int(config["target_tracking_max_gap_frames"]) < 0:
        raise ValueError("target_tracking_max_gap_frames must not be negative")
    if not 0 <= float(config["target_tracking_minimum_iou"]) <= 1:
        raise ValueError("target_tracking_minimum_iou must be in [0, 1]")
    if float(config["target_tracking_maximum_center_distance_ratio"]) <= 0:
        raise ValueError("target_tracking_maximum_center_distance_ratio must be greater than 0")
    if float(config["target_tracking_maximum_size_ratio"]) < 1:
        raise ValueError("target_tracking_maximum_size_ratio must be at least 1")
    return config, config_path.resolve().parent


def configure_cuda_library_path(config: dict[str, Any]) -> None:
    """Load configured CUDA libraries before ONNX Runtime initializes CUDA."""
    library_dirs = [str(path) for path in config.get("cuda_library_dirs", []) if Path(path).is_dir()]
    project_nvidia_root = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
    )
    if project_nvidia_root.is_dir():
        library_dirs.extend(
            str(path) for path in project_nvidia_root.glob("*/lib") if path.is_dir()
        )
    library_dirs = list(dict.fromkeys(library_dirs))
    if library_dirs:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(library_dirs + ([current] if current else []))
        preload_configured_cuda_libraries(library_dirs)


def preload_configured_cuda_libraries(library_dirs: list[str]) -> None:
    """Preload CUDA dependencies by absolute path for an already-started Python process."""
    loaded_names: list[str] = []
    for filename in CUDA_LIBRARY_FILENAMES:
        candidate = next((Path(directory) / filename for directory in library_dirs if (Path(directory) / filename).is_file()), None)
        if candidate is None:
            continue
        try:
            PRELOADED_CUDA_LIBRARIES.append(ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL))
            loaded_names.append(filename)
        except OSError as exc:
            print(f"Warning: unable to preload {candidate}: {exc}", file=sys.stderr)
    if loaded_names:
        print(f"Preloaded configured CUDA libraries: {', '.join(loaded_names)}", flush=True)


def verify_gpu_provider(config: dict[str, Any]) -> None:
    available = ort.get_available_providers()
    if bool(config["require_gpu"]) and "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Check libcublasLt/cuDNN libraries and "
            "onnxruntime-gpu compatibility; refusing CPU fallback because require_gpu=true. "
            f"Available providers: {available}"
        )
    print(f"ONNX Runtime providers available: {available}", flush=True)


def preload_cuda_libraries() -> None:
    """Preload CUDA/cuDNN libraries from NVIDIA Python packages when supported."""
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(directory="")
        print("ONNX Runtime CUDA/cuDNN libraries preloaded", flush=True)
    except Exception as exc:
        print(f"Warning: ONNX Runtime library preload failed: {exc}", file=sys.stderr)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Face feature vector has zero length")
    return vector / norm


class GpuFaceMatcher:
    def __init__(self, config: dict[str, Any], base_dir: Path) -> None:
        self.threshold = float(config["mode_thresholds"][config["recognition_mode"]])
        self.minimum_face_size = int(config["minimum_face_size"])
        self.app = FaceAnalysis(
            name=str(config["model_pack"]),
            root=str(resolved_path(base_dir, config["model_cache_dir"])),
            providers=list(config["onnx_providers"]),
        )
        size = int(config["detection_size"])
        self.app.prepare(ctx_id=0, det_thresh=float(config["detection_threshold"]), det_size=(size, size))
        active_providers = []
        for model in self.app.models.values():
            session = getattr(model, "session", None)
            if session is not None:
                active_providers.extend(session.get_providers())
        if bool(config["require_gpu"]) and not any(provider == "CUDAExecutionProvider" for provider in active_providers):
            raise RuntimeError(
                "InsightFace models did not activate CUDAExecutionProvider; refusing CPU fallback. "
                f"Active providers: {sorted(set(active_providers))}"
            )
        print(f"InsightFace active providers: {sorted(set(active_providers))}", flush=True)
        self.reference_embedding = self._load_reference_embedding(resolved_path(base_dir, config["reference_faces_dir"]))

    def _load_reference_embedding(self, faces_dir: Path) -> np.ndarray:
        image_paths = [path for path in faces_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        embeddings: list[np.ndarray] = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            faces = self.app.get(image)
            if not faces:
                continue
            face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
            embeddings.append(normalize(face.embedding.astype(np.float32)))
        if not embeddings:
            raise ValueError(f"No usable reference face found in {faces_dir}")
        return normalize(np.mean(embeddings, axis=0))

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        detected: list[DetectedFace] = []
        for face in self.app.get(frame):
            x1, y1, x2, y2 = (round(float(value)) for value in face.bbox)
            width, height = x2 - x1, y2 - y1
            if min(width, height) < self.minimum_face_size:
                continue
            similarity = float(np.dot(normalize(face.embedding.astype(np.float32)), self.reference_embedding))
            is_target = similarity >= self.threshold
            detected.append(DetectedFace(similarity, (x1, y1, x2, y2), is_target, "recognized" if is_target else "other"))
        return detected


def video_duration(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frames <= 0:
        raise ValueError("Cannot determine video duration")
    return frames / fps


def log_detected_faces(stage: str, timestamp: float, frame_number: int, fps: float, faces: list[DetectedFace], config: dict[str, Any]) -> None:
    if not bool(config["insightface_detection_log"]):
        return
    interval_frames = max(1, round(float(config["detection_log_interval_seconds"]) * fps))
    if frame_number != 0 and (frame_number + 1) % interval_frames != 0:
        return
    recognized = sum(face.target_source == "recognized" for face in faces)
    tracked = sum(face.target_source == "tracked" for face in faces)
    masked = sum(not face.is_target for face in faces)
    message = (
        f"  [{stage}] t={timestamp:.2f}s faces={len(faces)} "
        f"target_recognized={recognized} target_tracked={tracked} masks={masked}"
    )
    if bool(config["detection_log_include_similarity"]):
        similarities = ", ".join(f"{face.similarity:.3f}" for face in faces)
        message += f" similarities=[{similarities}]"
    print(message, flush=True)


def target_tracker(config: dict[str, Any]) -> TargetFaceTracker:
    return TargetFaceTracker(
        max_gap_frames=int(config["target_tracking_max_gap_frames"]),
        minimum_iou=float(config["target_tracking_minimum_iou"]),
        maximum_center_distance_ratio=float(config["target_tracking_maximum_center_distance_ratio"]),
        maximum_size_ratio=float(config["target_tracking_maximum_size_ratio"]),
    )


def scan_video(video_path: Path, matcher: GpuFaceMatcher, config: dict[str, Any]) -> tuple[float, list[Match]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        duration = video_duration(capture)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        sample_fps = float(config["sample_fps"])
        interval = 1.0 / sample_fps
        total = max(1, math.ceil(duration / interval))
        matches: list[Match] = []
        next_sample_timestamp = 0.0
        frame_number = 0
        sample_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_number / source_fps
            frame_number += 1
            if timestamp + (0.5 / source_fps) < next_sample_timestamp:
                continue
            faces = matcher.detect(frame)
            target_faces = [face for face in faces if face.is_target]
            if target_faces:
                matches.append(Match(timestamp, max(face.similarity for face in target_faces)))
            log_detected_faces("scan", timestamp, sample_number, sample_fps, faces, config)
            sample_number += 1
            next_sample_timestamp += interval
            if sample_number == 1 or sample_number % max(1, math.ceil(sample_fps * 10)) == 0 or sample_number >= total:
                print(f"  scan {sample_number}/{total}, target matches: {len(matches)}", flush=True)
        return duration, matches
    finally:
        capture.release()


def group_appearances(matches: list[Match], config: dict[str, Any]) -> list[Appearance]:
    if not matches:
        return []
    interval = 1.0 / float(config["sample_fps"])
    max_gap = float(config["maximum_gap_seconds"]) + interval
    minimum_duration = float(config["minimum_appearance_seconds"])
    groups: list[list[Match]] = [[matches[0]]]
    for match in matches[1:]:
        if match.timestamp - groups[-1][-1].timestamp <= max_gap:
            groups[-1].append(match)
        else:
            groups.append([match])
    return [
        Appearance(group[0].timestamp, group[-1].timestamp + interval, max(item.similarity for item in group))
        for group in groups
        if group[-1].timestamp + interval - group[0].timestamp >= minimum_duration
    ]


def make_clips(appearances: list[Appearance], duration: float, config: dict[str, Any]) -> list[Clip]:
    clips = [Clip(max(0.0, item.first_match - float(config["before_seconds"])), min(duration, item.last_match + float(config["after_seconds"])), item.maximum_similarity) for item in appearances]
    if config["clip_overlap_strategy"] != "trim":
        return clips
    clips = list(clips)
    for index in range(len(clips) - 1):
        current, following = clips[index], clips[index + 1]
        if current.end > following.start:
            boundary = (current.end + following.start) / 2
            clips[index] = Clip(current.start, boundary, current.maximum_similarity)
            clips[index + 1] = Clip(boundary, following.end, following.maximum_similarity)
    return clips


def time_label(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}_{milliseconds:03d}"


def overlay_mask(frame: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], config: dict[str, Any]) -> None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    face_width, face_height = max(1, x2 - x1), max(1, y2 - y1)
    size = max(int(config["mask_minimum_size"]), round(max(face_width, face_height) * float(config["mask_scale"])))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    left, top = center_x - size // 2, center_y - size // 2
    right, bottom = left + size, top + size
    clipped_left, clipped_top = max(0, left), max(0, top)
    clipped_right, clipped_bottom = min(frame_width, right), min(frame_height, bottom)
    if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
        return
    resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA if size < mask.shape[0] else cv2.INTER_LINEAR)
    roi = frame[clipped_top:clipped_bottom, clipped_left:clipped_right]
    mask_x1, mask_y1 = clipped_left - left, clipped_top - top
    mask_x2, mask_y2 = mask_x1 + roi.shape[1], mask_y1 + roi.shape[0]
    crop = resized[mask_y1:mask_y2, mask_x1:mask_x2]
    if crop.shape[2] == 4:
        alpha = (crop[:, :, 3].astype(np.float32) / 255.0) * float(config["mask_alpha"])
        foreground = crop[:, :, :3]
    else:
        alpha = np.full(crop.shape[:2], float(config["mask_alpha"]), dtype=np.float32)
        foreground = crop
    alpha = alpha[:, :, None]
    roi[:] = (foreground.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def export_clip(video_path: Path, clip: Clip, sequence: int, config: dict[str, Any], output_dir: Path, matcher: GpuFaceMatcher, mask_image: np.ndarray) -> Path:
    output_path = output_dir / f"{video_path.stem}__clip_{sequence:03d}__{time_label(clip.start)}_{time_label(clip.end)}.mp4"
    if output_path.exists() and not bool(config["overwrite_output"]):
        print(f"  output exists, skipped: {output_path.name}", flush=True)
        return output_path
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video for export: {video_path}")
    process: subprocess.Popen[bytes] | None = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = max(0.001, clip.end - clip.start)
        capture.set(cv2.CAP_PROP_POS_MSEC, clip.start * 1000)
        command = [
            "ffmpeg", "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}", "-framerate", f"{fps:.12g}", "-i", "pipe:0",
            "-ss", f"{clip.start:.3f}", "-t", f"{duration:.3f}", "-i", str(video_path), "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
            "-c:v", str(config["video_codec"]), "-preset", str(config["video_preset"]), "-cq", str(config["video_cq"]), "-pix_fmt", "yuv420p",
            "-c:a", str(config["audio_codec"]), "-shortest", "-movflags", "+faststart", str(output_path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        if process.stdin is None:
            raise RuntimeError("Cannot create FFmpeg pipe")
        expected_frames = max(1, math.ceil(duration * fps))
        tracker = target_tracker(config) if bool(config["target_tracking_enabled"]) else None
        for frame_number in range(expected_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if bool(config["mask_other_faces"]):
                faces = matcher.detect(frame)
                if tracker is not None:
                    faces = tracker.resolve(faces)
                log_detected_faces("export", clip.start + frame_number / fps, frame_number, fps, faces, config)
                for face in faces:
                    if not face.is_target:
                        overlay_mask(frame, mask_image, face.bbox, config)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        capture.release()
        if process is not None and process.poll() is None:
            process.kill()
    return output_path


def select_videos(config: dict[str, Any], base_dir: Path, selected: Path | None) -> list[Path]:
    if selected:
        return [selected.resolve()]
    input_dir = resolved_path(base_dir, config["input_dir"])
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"})


def main() -> int:
    args = parse_args()
    config, base_dir = load_config(args.config.resolve())
    configure_cuda_library_path(config)
    preload_cuda_libraries()
    verify_gpu_provider(config)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    output_dir = resolved_path(base_dir, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    matcher = GpuFaceMatcher(config, base_dir)
    mask_image = cv2.imread(str(resolved_path(base_dir, config["mask_image_path"])), cv2.IMREAD_UNCHANGED)
    if bool(config["mask_other_faces"]) and mask_image is None:
        raise ValueError("Cannot read mask_image_path")
    report: dict[str, Any] = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "threshold": matcher.threshold, "videos": []}
    for video_path in select_videos(config, base_dir, args.video):
        print(f"Scanning: {video_path.name}", flush=True)
        duration, matches = scan_video(video_path, matcher, config)
        appearances = group_appearances(matches, config)
        clips = make_clips(appearances, duration, config)
        result: dict[str, Any] = {"source_video": str(video_path), "duration_seconds": duration, "matches": len(matches), "appearances": [asdict(item) for item in appearances], "clips": []}
        for sequence, clip in enumerate(clips, start=1):
            if args.scan_only:
                result["clips"].append({**asdict(clip), "output_video": None})
                continue
            print(f"  export {sequence}/{len(clips)}: {clip.start:.2f}s - {clip.end:.2f}s", flush=True)
            output_path = export_clip(video_path, clip, sequence, config, output_dir, matcher, mask_image)
            result["clips"].append({**asdict(clip), "output_video": str(output_path)})
        report["videos"].append(result)
    (output_dir / "gpu_face_clip_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
