#!/usr/bin/env python3
"""Merge videos recursively, ordered by their parent directory name descending."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="包含视频或子目录的目录")
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 MP4 路径。默认输出到输入目录：merged_by_folder_desc.mp4",
    )
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="重新编码为 H.264/AAC；用于源视频编码、分辨率或帧率不一致的情况。",
    )
    parser.add_argument(
        "--include-root-videos",
        action="store_true",
        help="包含输入目录根目录下的视频；默认只合并子目录中的视频。",
    )
    return parser.parse_args()


def find_videos(input_dir: Path, output_path: Path, include_root_videos: bool) -> list[Path]:
    candidates = [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and path.resolve() != output_path.resolve()
        and (include_root_videos or path.parent != input_dir)
    ]
    # Stable sorting keeps filenames ascending inside each parent folder while
    # the parent folder names themselves are ordered descending.
    candidates.sort(key=lambda path: path.name.casefold())
    candidates.sort(key=lambda path: path.parent.name.casefold(), reverse=True)
    return candidates


def concat_list_line(path: Path) -> str:
    # FFmpeg concat syntax uses a single-quoted filename; escape embedded quotes.
    return "file '" + path.resolve().as_posix().replace("'", "'\\''") + "'\n"


def build_command(list_path: Path, output_path: Path, reencode: bool) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
    ]
    if reencode:
        command.extend(["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "aac", "-b:a", "192k"])
    else:
        command.extend(["-c", "copy"])
    command.append(str(output_path))
    return command


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在：{input_dir}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，请先安装并加入 PATH")

    output_path = (args.output or input_dir / "merged_by_folder_desc.mp4").expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    videos = find_videos(input_dir, output_path, args.include_root_videos)
    if not videos:
        raise ValueError("没有找到可合并的视频")

    print("合并顺序（目录名降序）：")
    for index, path in enumerate(videos, start=1):
        print(f"{index:03d}. {path.relative_to(input_dir)}")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        list_path = Path(handle.name)
        handle.writelines(concat_list_line(path) for path in videos)

    try:
        command = build_command(list_path, output_path, args.reencode)
        print("开始合并：", output_path)
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        if not args.reencode:
            raise RuntimeError(
                "无重编码拼接失败。源视频的编码参数可能不一致，请使用 --reencode 重试。"
            ) from exc
        raise
    finally:
        list_path.unlink(missing_ok=True)

    print(f"合并完成：{output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
