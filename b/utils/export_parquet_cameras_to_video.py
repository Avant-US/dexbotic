#!/usr/bin/env python3
"""
Export camera videos from parquet episodes.

Key features:
- Export three camera streams in one run (default):
  `head_rgb`, `left_wrist_rgb`, `right_wrist_rgb`
- Place output videos in the same directory as source parquet files
- Name output videos as `<parquet_stem>_<camera>.mp4`
- Optional frame sampling via `--frame-step`, `--start-frame`, `--max-frames`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")


@dataclass
class ExportStats:
    output_path: Path
    raw_rows: int = 0
    sampled_rows: int = 0
    written_frames: int = 0
    missing_bytes: int = 0
    decode_failures: int = 0
    shape_mismatches: int = 0
    skipped_existing: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export videos from parquet image-byte columns. "
            "Output videos are saved beside each parquet file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Parquet file or directory containing parquet files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="episode_*.parquet",
        help="Filename pattern used when --input is a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively search parquet files under input directory.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help=(
            "Camera columns to export. Accepts space-separated values or "
            "comma-separated values."
        ),
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Sample one frame every N rows. 1 means no sampling.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start row index for sampling.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames per output video. 0 means no limit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Parquet batch size for streaming reads.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help="Video codec used by imageio ffmpeg writer.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing output videos.",
    )
    return parser.parse_args()


def normalize_cameras(raw_cameras: Sequence[str]) -> List[str]:
    cameras: List[str] = []
    for item in raw_cameras:
        for camera in item.split(","):
            camera = camera.strip()
            if camera and camera not in cameras:
                cameras.append(camera)
    if not cameras:
        raise ValueError("No valid cameras provided.")
    return cameras


def collect_parquet_files(input_path: Path, pattern: str, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".parquet":
            raise ValueError(f"Input file is not parquet: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if recursive:
        files = sorted(input_path.rglob(pattern))
    else:
        files = sorted(input_path.glob(pattern))

    return [path for path in files if path.is_file() and path.suffix.lower() == ".parquet"]


def get_struct_fields_with_bytes(schema: pa.Schema) -> List[str]:
    names: List[str] = []
    for field in schema:
        if not pa.types.is_struct(field.type):
            continue
        subfield_names = set(field.type.names)
        if "bytes" in subfield_names:
            names.append(field.name)
    return names


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        frame = frame[:, :, :3]
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def iter_image_bytes(
    parquet_file: pq.ParquetFile,
    camera: str,
    batch_size: int,
) -> Iterable[bytes | None]:
    for batch in parquet_file.iter_batches(columns=[camera], batch_size=batch_size):
        struct_col = batch.column(0)
        if not pa.types.is_struct(struct_col.type):
            raise ValueError(
                f"Column `{camera}` is not struct type in file {parquet_file}"
            )

        if "bytes" not in struct_col.type.names:
            raise ValueError(f"Column `{camera}` has no `bytes` subfield")

        bytes_col = struct_col.field("bytes")
        for row_id in range(batch.num_rows):
            yield bytes_col[row_id].as_py()


def export_one_camera(
    parquet_path: Path,
    parquet_file: pq.ParquetFile,
    camera: str,
    fps: float,
    frame_step: int,
    start_frame: int,
    max_frames: int,
    batch_size: int,
    codec: str,
    force: bool,
) -> ExportStats:
    output_path = parquet_path.with_name(f"{parquet_path.stem}_{camera}.mp4")
    stats = ExportStats(output_path=output_path)

    if output_path.exists() and not force:
        stats.skipped_existing = True
        return stats

    if output_path.exists() and force:
        output_path.unlink()

    writer = None
    expected_shape = None

    try:
        for row_idx, image_bytes in enumerate(
            iter_image_bytes(parquet_file=parquet_file, camera=camera, batch_size=batch_size)
        ):
            stats.raw_rows += 1
            if row_idx < start_frame:
                continue
            if frame_step > 1 and ((row_idx - start_frame) % frame_step != 0):
                continue
            if max_frames > 0 and stats.written_frames >= max_frames:
                break

            stats.sampled_rows += 1
            if image_bytes is None:
                stats.missing_bytes += 1
                continue

            try:
                frame = imageio.imread(BytesIO(image_bytes))
                frame = to_uint8_rgb(np.asarray(frame))
            except Exception:
                stats.decode_failures += 1
                continue

            if expected_shape is None:
                expected_shape = frame.shape
                writer = imageio.get_writer(
                    output_path.as_posix(),
                    format="ffmpeg",
                    mode="I",
                    fps=fps,
                    codec=codec,
                    macro_block_size=1,
                )
            elif frame.shape != expected_shape:
                stats.shape_mismatches += 1
                continue

            writer.append_data(frame)
            stats.written_frames += 1
    finally:
        if writer is not None:
            writer.close()

    if stats.written_frames == 0 and output_path.exists():
        output_path.unlink()

    return stats


def main() -> int:
    args = parse_args()
    # Keep user-provided path semantics (do not resolve symlinks).
    input_path = Path(args.input).expanduser()

    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    cameras = normalize_cameras(args.cameras)
    parquet_files = collect_parquet_files(
        input_path=input_path,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    if not parquet_files:
        print(f"[ERROR] No parquet files found under: {input_path}")
        return 1

    print(f"[INFO] Found {len(parquet_files)} parquet files.")
    print(f"[INFO] Cameras: {cameras}")
    print(
        f"[INFO] Sampling: start={args.start_frame}, step={args.frame_step}, "
        f"max_frames={args.max_frames}"
    )

    total_written = 0
    total_outputs = 0
    total_skipped_existing = 0

    for parquet_path in parquet_files:
        print(f"\n[FILE] {parquet_path}")
        parquet_file = pq.ParquetFile(parquet_path)
        available_cameras = get_struct_fields_with_bytes(parquet_file.schema_arrow)
        print(f"[INFO] Available image columns: {available_cameras}")

        for camera in cameras:
            if camera not in available_cameras:
                print(f"[WARN] Camera `{camera}` not found in this parquet, skip.")
                continue

            stats = export_one_camera(
                parquet_path=parquet_path,
                parquet_file=parquet_file,
                camera=camera,
                fps=args.fps,
                frame_step=args.frame_step,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                batch_size=args.batch_size,
                codec=args.codec,
                force=args.force,
            )

            total_outputs += 1
            if stats.skipped_existing:
                total_skipped_existing += 1
                print(f"[SKIP] {stats.output_path} already exists (use --force to overwrite).")
                continue

            total_written += stats.written_frames
            print(
                f"[OK] {camera} -> {stats.output_path.name} | "
                f"written={stats.written_frames}, raw_rows={stats.raw_rows}, "
                f"sampled={stats.sampled_rows}, missing={stats.missing_bytes}, "
                f"decode_fail={stats.decode_failures}, shape_mismatch={stats.shape_mismatches}"
            )

    print("\n[SUMMARY]")
    print(f"  requested_outputs: {total_outputs}")
    print(f"  skipped_existing:  {total_skipped_existing}")
    print(f"  total_written:     {total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
