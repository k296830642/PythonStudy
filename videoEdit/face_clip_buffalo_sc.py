#!/usr/bin/env python3
"""Find a reference face with InsightFace buffalo_sc and export each appearance."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Match:
    timestamp: float
    similarity: float


@dataclass(frozen=True)
class FrameMatch:
    similarity: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class DetectedFace:
    similarity: float
    bbox: tuple[int, int, int, int]
    is_target: bool
    target_source: str = "other"


@dataclass
class TargetFaceTracker:
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
        candidates = [face for face in faces if self._matches(face.bbox, predicted)]
        if not candidates:
            self.missing_frames += 1
            return faces
        tracked = max(candidates, key=lambda face: bbox_iou(face.bbox, predicted))
        self._update(tracked.bbox)
        return [replace(face, is_target=True, target_source="tracked") if face is tracked else face for face in faces]

    def _update(self, bbox: tuple[int, int, int, int]) -> None:
        if self.last_bbox is not None:
            old = face_center(self.last_bbox)
            new = face_center(bbox)
            self.velocity = (new[0] - old[0], new[1] - old[1])
        self.last_bbox = bbox
        self.missing_frames = 0

    def _predicted_bbox(self) -> tuple[int, int, int, int]:
        if self.last_bbox is None:
            raise RuntimeError("Target tracker has no previous box")
        dx, dy = self.velocity
        x1, y1, x2, y2 = self.last_bbox
        return tuple(round(value + delta) for value, delta in ((x1, dx), (y1, dy), (x2, dx), (y2, dy)))

    def _matches(self, bbox: tuple[int, int, int, int], predicted: tuple[int, int, int, int]) -> bool:
        pw, ph = face_size(predicted)
        w, h = face_size(bbox)
        if max(w / pw, pw / w, h / ph, ph / h) > self.maximum_size_ratio:
            return False
        cx, cy = face_center(bbox)
        px, py = face_center(predicted)
        distance = math.hypot(cx - px, cy - py)
        return bbox_iou(bbox, predicted) >= self.minimum_iou or distance <= math.hypot(pw, ph) * self.maximum_center_distance_ratio


def face_size(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])


def face_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if not intersection:
        return 0.0
    first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
    second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
    return intersection / float(first_area + second_area - intersection)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config_buffalo_sc.json")
    parser.add_argument("--video", type=Path, help="Process only one video file.")
    parser.add_argument("--scan-only", action="store_true", help="Detect appearances without exporting clips.")
    return parser.parse_args()


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    if not config_path.is_file():
        raise ValueError(f"Configuration file does not exist: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    required = {
        "reference_faces_dir", "input_dir", "output_dir", "model_dir", "detector_model",
        "recognizer_model", "detector_model_url", "recognizer_model_url",
        "recognition_mode", "mode_thresholds", "sample_fps", "before_seconds", "after_seconds",
        "minimum_appearance_seconds", "maximum_gap_seconds", "clip_overlap_strategy", "minimum_face_size",
        "annotate_matched_face", "annotation_show_similarity", "annotation_box_color_bgr",
        "annotation_box_thickness", "annotation_label_font_scale", "annotation_label_thickness",
        "detection_threshold", "detection_nms_threshold", "video_extensions", "video_codec", "video_crf",
        "video_preset", "audio_codec", "overwrite_output", "mask_other_faces", "mask_image_path",
        "mask_scale", "mask_minimum_size", "mask_alpha", "insightface_detection_log",
        "detection_log_interval_seconds", "detection_log_include_similarity", "target_tracking_enabled",
        "target_tracking_max_gap_frames", "target_tracking_minimum_iou",
        "target_tracking_maximum_center_distance_ratio", "target_tracking_maximum_size_ratio",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Configuration is missing required fields: {', '.join(missing)}")
    if config["recognition_mode"] not in config["mode_thresholds"]:
        raise ValueError("recognition_mode must exist in mode_thresholds")
    if config.get("reference_match_strategy", "max") not in {"max", "mean_top3"}:
        raise ValueError("reference_match_strategy must be 'max' or 'mean_top3'")
    if config["sample_fps"] <= 0:
        raise ValueError("sample_fps must be greater than 0")
    if config["minimum_appearance_seconds"] < 0 or config["maximum_gap_seconds"] < 0:
        raise ValueError("Duration values cannot be negative")
    if config["clip_overlap_strategy"] not in {"allow", "trim"}:
        raise ValueError("clip_overlap_strategy must be 'allow' or 'trim'")
    box_color = config["annotation_box_color_bgr"]
    if not isinstance(box_color, list) or len(box_color) != 3 or any(not isinstance(value, int) or value < 0 or value > 255 for value in box_color):
        raise ValueError("annotation_box_color_bgr must be three integers between 0 and 255")
    if int(config["annotation_box_thickness"]) < 1 or int(config["annotation_label_thickness"]) < 1:
        raise ValueError("Annotation thickness values must be at least 1")
    if float(config["annotation_label_font_scale"]) <= 0:
        raise ValueError("annotation_label_font_scale must be greater than 0")
    if not 0 < float(config["mask_alpha"]) <= 1 or float(config["mask_scale"]) <= 0:
        raise ValueError("mask_alpha must be in (0, 1] and mask_scale must be positive")
    if float(config["detection_log_interval_seconds"]) <= 0 or int(config["target_tracking_max_gap_frames"]) < 0:
        raise ValueError("Invalid detection log or tracking interval")
    if not 0 <= float(config["target_tracking_minimum_iou"]) <= 1:
        raise ValueError("target_tracking_minimum_iou must be in [0, 1]")
    if float(config["target_tracking_maximum_center_distance_ratio"]) <= 0 or float(config["target_tracking_maximum_size_ratio"]) < 1:
        raise ValueError("Invalid target tracking limits")
    return config, config_path.resolve().parent


def resolved_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def normalize(vector: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    if magnitude == 0:
        raise ValueError("A face embedding with zero length was returned")
    return vector / magnitude


class FaceMatcher:
    def __init__(self, config: dict[str, Any], base_dir: Path) -> None:
        self.minimum_face_size = int(config["minimum_face_size"])
        self.threshold = float(config["mode_thresholds"][config["recognition_mode"]])
        self.reference_match_strategy = str(config.get("reference_match_strategy", "max"))
        self.model_name = str(config.get("insightface_model", "buffalo_sc"))
        if self.model_name != "buffalo_sc":
            raise ValueError("face_clip_buffalo_sc.py only supports insightface_model='buffalo_sc'")
        self.detector_size = tuple(int(value) for value in config.get("insightface_detector_size", [640, 640]))
        if len(self.detector_size) != 2 or min(self.detector_size) <= 0:
            raise ValueError("insightface_detector_size must contain two positive integers")
        self.app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=self.detector_size, det_thresh=float(config["detection_threshold"]))
        self.reference_embeddings = self._load_reference_embeddings(
            resolved_path(base_dir, config["reference_faces_dir"])
        )
        print(
            f"InsightFace model: {self.model_name}; provider: CPUExecutionProvider; detector size: {self.detector_size[0]}x{self.detector_size[1]}; usable reference faces: {len(self.reference_embeddings)}",
            flush=True,
        )

    def _detect_faces(self, image: np.ndarray) -> list[Any]:
        return self.app.get(image)

    @staticmethod
    def _embedding(face: Any) -> np.ndarray:
        return normalize(np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1))

    def _load_reference_embeddings(self, faces_dir: Path) -> list[np.ndarray]:
        image_paths = [
            path for path in faces_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ] if faces_dir.is_dir() else []
        if not image_paths:
            raise ValueError(f"No reference face image found in: {faces_dir}")

        embeddings: list[np.ndarray] = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: unreadable reference image skipped: {image_path.name}", file=sys.stderr)
                continue
            faces = self._detect_faces(image)
            if len(faces) == 0:
                print(f"Warning: no face found in reference image: {image_path.name}", file=sys.stderr)
                continue
            largest_face = max(faces, key=self._face_area)
            embeddings.append(self._embedding(largest_face))

        if not embeddings:
            raise ValueError("No usable face was detected in the reference image directory")
        return embeddings

    @staticmethod
    def _face_area(face: Any) -> float:
        x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32)
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        detected: list[DetectedFace] = []
        for face in self._detect_faces(frame):
            x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32)
            width = x2 - x1
            height = y2 - y1
            if min(width, height) < self.minimum_face_size:
                continue
            embedding = self._embedding(face)
            similarities = [float(np.dot(embedding, reference)) for reference in self.reference_embeddings]
            if self.reference_match_strategy == "mean_top3":
                similarity = float(np.mean(sorted(similarities, reverse=True)[: min(3, len(similarities))]))
            else:
                similarity = max(similarities)
            bbox = (max(0, round(float(x1))), max(0, round(float(y1))), max(1, round(float(x2))), max(1, round(float(y2))))
            detected.append(DetectedFace(similarity, bbox, similarity >= self.threshold, "recognized" if similarity >= self.threshold else "other"))
        return detected

    def match(self, frame: np.ndarray) -> FrameMatch | None:
        targets = [face for face in self.detect(frame) if face.is_target]
        if not targets:
            return None
        face = max(targets, key=lambda item: item.similarity)
        return FrameMatch(face.similarity, face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])


def video_duration(capture: cv2.VideoCapture) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        raise ValueError("Could not determine the video duration")
    return frame_count / fps


def scan_video(video_path: Path, matcher: FaceMatcher, sample_fps: float) -> tuple[float, list[Match]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        duration = video_duration(capture)
        interval = 1.0 / sample_fps
        total = max(1, math.ceil(duration / interval))
        matches: list[Match] = []
        for index in range(total):
            timestamp = min(index * interval, max(0.0, duration - 0.001))
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                print(f"Warning: unable to decode {timestamp:.2f}s in {video_path.name}", file=sys.stderr)
                continue
            frame_match = matcher.match(frame)
            if frame_match is not None:
                matches.append(Match(timestamp=timestamp, similarity=frame_match.similarity))
            if index == 0 or (index + 1) % max(1, math.ceil(sample_fps * 10)) == 0 or index + 1 == total:
                print(f"  scan {index + 1}/{total} ({timestamp:.1f}s), matches: {len(matches)}", flush=True)
        return duration, matches
    finally:
        capture.release()


def group_appearances(matches: list[Match], sample_fps: float, max_gap: float, minimum_duration: float) -> list[Appearance]:
    if not matches:
        return []
    interval = 1.0 / sample_fps
    appearances: list[Appearance] = []
    group: list[Match] = [matches[0]]
    for match in matches[1:]:
        if match.timestamp - group[-1].timestamp <= max_gap + interval:
            group.append(match)
            continue
        appearance = build_appearance(group, interval)
        if appearance.last_match - appearance.first_match >= minimum_duration:
            appearances.append(appearance)
        group = [match]
    appearance = build_appearance(group, interval)
    if appearance.last_match - appearance.first_match >= minimum_duration:
        appearances.append(appearance)
    return appearances


def build_appearance(matches: list[Match], interval: float) -> Appearance:
    return Appearance(
        first_match=matches[0].timestamp,
        last_match=matches[-1].timestamp + interval,
        maximum_similarity=max(match.similarity for match in matches),
    )


def make_clips(appearances: list[Appearance], duration: float, config: dict[str, Any]) -> list[Clip]:
    clips = [
        Clip(
            start=max(0.0, appearance.first_match - float(config["before_seconds"])),
            end=min(duration, appearance.last_match + float(config["after_seconds"])),
            maximum_similarity=appearance.maximum_similarity,
        )
        for appearance in appearances
    ]
    if config.get("clip_overlap_strategy", "allow") != "trim":
        return clips

    # Short gaps cannot preserve full context on both clips without duplicate video.
    # Split an overlap at its midpoint so every exported time range is unique.
    trimmed = list(clips)
    for index in range(len(trimmed) - 1):
        current = trimmed[index]
        following = trimmed[index + 1]
        if current.end <= following.start:
            continue
        boundary = (current.end + following.start) / 2
        trimmed[index] = Clip(current.start, boundary, current.maximum_similarity)
        trimmed[index + 1] = Clip(boundary, following.end, following.maximum_similarity)
    return trimmed


def time_label(seconds: float) -> str:
    total_milliseconds = round(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}_{milliseconds:03d}"


def output_path_for_clip(video_path: Path, clip: Clip, sequence: int, output_dir: Path) -> Path:
    return output_dir / (
        f"{video_path.stem}__clip_{sequence:03d}__{time_label(clip.start)}_{time_label(clip.end)}.mp4"
    )


def annotate_frame(frame: np.ndarray, frame_match: FrameMatch, config: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    x1 = min(max(0, frame_match.x), width - 1)
    y1 = min(max(0, frame_match.y), height - 1)
    x2 = min(width - 1, x1 + frame_match.width)
    y2 = min(height - 1, y1 + frame_match.height)
    color = tuple(int(value) for value in config["annotation_box_color_bgr"])
    thickness = int(config["annotation_box_thickness"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if not bool(config["annotation_show_similarity"]):
        return
    label = f"MATCH {frame_match.similarity:.3f}"
    font_scale = float(config["annotation_label_font_scale"])
    label_thickness = int(config["annotation_label_thickness"])
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness
    )
    label_top = max(0, y1 - label_height - baseline - 6)
    label_right = min(width - 1, x1 + label_width + 6)
    cv2.rectangle(frame, (x1, label_top), (label_right, y1), color, cv2.FILLED)
    cv2.putText(
        frame, label, (x1 + 3, y1 - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA,
    )


def overlay_mask(frame: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], config: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    size = max(int(config["mask_minimum_size"]), round(max(x2 - x1, y2 - y1) * float(config["mask_scale"])))
    left, top = (x1 + x2 - size) // 2, (y1 + y2 - size) // 2
    right, bottom = left + size, top + size
    cl, ct, cr, cb = max(0, left), max(0, top), min(width, right), min(height, bottom)
    if cl >= cr or ct >= cb:
        return
    resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA if size < mask.shape[0] else cv2.INTER_LINEAR)
    crop = resized[ct - top:cb - top, cl - left:cr - left]
    roi = frame[ct:cb, cl:cr]
    if crop.ndim == 3 and crop.shape[2] == 4:
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0 * float(config["mask_alpha"])
        foreground = crop[:, :, :3].astype(np.float32)
    else:
        alpha = np.full((*crop.shape[:2], 1), float(config["mask_alpha"]), dtype=np.float32)
        foreground = crop.astype(np.float32)
    roi[:] = np.clip(foreground * alpha + roi.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)


def log_detected_faces(stage: str, timestamp: float, frame_number: int, fps: float, faces: list[DetectedFace], config: dict[str, Any]) -> None:
    if not bool(config["insightface_detection_log"]):
        return
    interval = max(1, round(float(config["detection_log_interval_seconds"]) * fps))
    if frame_number and (frame_number + 1) % interval:
        return
    recognized = sum(face.target_source == "recognized" for face in faces)
    tracked = sum(face.target_source == "tracked" for face in faces)
    masked = sum(not face.is_target for face in faces)
    message = f"  [{stage}] t={timestamp:.2f}s faces={len(faces)} target_recognized={recognized} target_tracked={tracked} masks={masked}"
    if bool(config["detection_log_include_similarity"]):
        message += " similarities=[" + ", ".join(f"{face.similarity:.3f}" for face in faces) + "]"
    print(message, flush=True)


def export_annotated_clip(
    video_path: Path, clip: Clip, config: dict[str, Any], output_path: Path, matcher: FaceMatcher
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video for annotation: {video_path}")
    process: subprocess.Popen[bytes] | None = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Could not determine video dimensions for annotation: {video_path}")
        duration = max(0.001, clip.end - clip.start)
        expected_frames = max(1, math.ceil(duration * fps))
        capture.set(cv2.CAP_PROP_POS_MSEC, clip.start * 1000)
        command = [
            "ffmpeg", "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", f"{fps:.12g}", "-i", "pipe:0",
            "-ss", f"{clip.start:.3f}", "-t", f"{duration:.3f}", "-i", str(video_path),
            "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-c:v", str(config["video_codec"]),
            "-crf", str(config["video_crf"]), "-preset", str(config["video_preset"]),
            "-pix_fmt", "yuv420p", "-c:a", str(config["audio_codec"]), "-shortest",
            "-movflags", "+faststart", str(output_path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        if process.stdin is None:
            raise RuntimeError("Could not open FFmpeg video input pipe")
        tracker = TargetFaceTracker(
            int(config["target_tracking_max_gap_frames"]), float(config["target_tracking_minimum_iou"]),
            float(config["target_tracking_maximum_center_distance_ratio"]), float(config["target_tracking_maximum_size_ratio"]),
        ) if bool(config["target_tracking_enabled"]) else None
        mask_image = cv2.imread(str(resolved_path(Path(__file__).resolve().parent, config["mask_image_path"])), cv2.IMREAD_UNCHANGED)
        if bool(config["mask_other_faces"]) and mask_image is None:
            raise ValueError(f"Could not read mask image: {config['mask_image_path']}")
        for frame_number in range(expected_frames):
            ok, frame = capture.read()
            if not ok:
                break
            faces = matcher.detect(frame)
            if tracker is not None:
                faces = tracker.resolve(faces)
            log_detected_faces("export", clip.start + frame_number / fps, frame_number, fps, faces, config)
            targets = [face for face in faces if face.is_target]
            if targets and bool(config["annotate_matched_face"]):
                target = max(targets, key=lambda item: item.similarity)
                annotate_frame(frame, FrameMatch(target.similarity, target.bbox[0], target.bbox[1], target.bbox[2] - target.bbox[0], target.bbox[3] - target.bbox[1]), config)
            if bool(config["mask_other_faces"]):
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


def export_clip(
    video_path: Path, clip: Clip, sequence: int, config: dict[str, Any], output_dir: Path, matcher: FaceMatcher
) -> Path:
    output_path = output_path_for_clip(video_path, clip, sequence, output_dir)
    if output_path.exists() and not bool(config["overwrite_output"]):
        print(f"  output exists, skipped: {output_path.name}", flush=True)
        return output_path
    if bool(config["annotate_matched_face"]) or bool(config["mask_other_faces"]):
        export_annotated_clip(video_path, clip, config, output_path, matcher)
        return output_path
    duration = max(0.001, clip.end - clip.start)
    command = [
        "ffmpeg", "-hide_banner", "-y", "-ss", f"{clip.start:.3f}", "-i", str(video_path),
        "-t", f"{duration:.3f}", "-map", "0:v:0?", "-map", "0:a?", "-map_metadata", "0",
        "-c:v", str(config["video_codec"]), "-crf", str(config["video_crf"]),
        "-preset", str(config["video_preset"]), "-c:a", str(config["audio_codec"]),
        "-movflags", "+faststart", str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def select_videos(config: dict[str, Any], base_dir: Path, requested_video: Path | None) -> list[Path]:
    if requested_video:
        video = requested_video.resolve()
        if not video.is_file():
            raise ValueError(f"Video does not exist: {video}")
        return [video]
    input_dir = resolved_path(base_dir, config["input_dir"])
    extensions = {extension.lower() for extension in config["video_extensions"]}
    videos = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    if not videos:
        raise ValueError(f"No supported videos found in: {input_dir}")
    return videos


def main() -> int:
    args = parse_args()
    config, base_dir = load_config(args.config.resolve())
    if shutil.which("ffmpeg") is None and not args.scan_only:
        raise RuntimeError("ffmpeg was not found in PATH")
    output_dir = resolved_path(base_dir, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    matcher = FaceMatcher(config, base_dir)
    videos = select_videos(config, base_dir, args.video)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recognition_mode": config["recognition_mode"],
        "similarity_threshold": matcher.threshold,
        "clip_overlap_strategy": config["clip_overlap_strategy"],
        "annotate_matched_face": config["annotate_matched_face"],
        "videos": [],
    }

    for video_path in videos:
        print(f"Scanning: {video_path.name}", flush=True)
        duration, matches = scan_video(video_path, matcher, float(config["sample_fps"]))
        appearances = group_appearances(
            matches, float(config["sample_fps"]), float(config["maximum_gap_seconds"]),
            float(config["minimum_appearance_seconds"]),
        )
        clips = make_clips(appearances, duration, config)
        entry: dict[str, Any] = {
            "source_video": str(video_path), "duration_seconds": duration,
            "matches": len(matches),
            "match_samples": [asdict(item) for item in matches],
            "appearances": [asdict(item) for item in appearances],
            "clips": [],
        }
        for sequence, clip in enumerate(clips, start=1):
            if args.scan_only:
                entry["clips"].append({**asdict(clip), "output_video": None})
                continue
            print(f"  export clip {sequence}/{len(clips)}: {clip.start:.2f}s - {clip.end:.2f}s", flush=True)
            output_path = export_clip(video_path, clip, sequence, config, output_dir, matcher)
            entry["clips"].append({**asdict(clip), "output_video": str(output_path)})
        report["videos"].append(entry)

    report_path = output_dir / "face_clip_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, subprocess.CalledProcessError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
