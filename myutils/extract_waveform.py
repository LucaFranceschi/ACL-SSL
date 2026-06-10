#!/usr/bin/env python3
"""
Traverses a directory recursively, finds all .wav files,
and generates a waveform visualization saved as waveform.jpeg
in the same directory as each .wav file.
"""

import sys
import wave
import struct
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_wav(filepath: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (samples, sample_rate)."""
    with wave.open(str(filepath), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

    fmt_map = {1: "b", 2: "h", 4: "i"}
    fmt = fmt_map.get(sample_width)
    if fmt is None:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    samples = np.array(struct.unpack(f"<{n_frames * n_channels}{fmt}", raw), dtype=np.float64)

    # Normalise to [-1, 1]
    max_val = 2 ** (8 * sample_width - 1)
    samples /= max_val

    # Mix down to mono if stereo
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sample_rate


def generate_waveform_image(wav_path: Path) -> Path:
    """Generate a waveform JPEG for *wav_path* and save it alongside the file."""
    samples, sample_rate = read_wav(wav_path)
    duration = len(samples) / sample_rate
    time_axis = np.linspace(0, duration, num=len(samples))

    # Downsample for very long files to keep rendering fast
    max_points = 200_000
    if len(samples) > max_points:
        step = len(samples) // max_points
        samples = samples[::step]
        time_axis = time_axis[::step]

    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor="white")
    ax.set_facecolor("white")

    # Grey filled waveform
    ax.fill_between(time_axis, samples, alpha=0.75, color="#aaaaaa", linewidth=0)
    ax.plot(time_axis, samples, color="#888888", linewidth=0.4, alpha=0.9)

    ax.set_xlim(0, duration)
    ax.set_ylim(-1.05, 1.05)

    # Remove all axes, ticks, labels and title
    ax.axis("off")

    # Green rounded rectangle over the first 3 seconds only
    box_margin = 0.08
    box_y = -1.05 + box_margin
    box_h = 2.1 - 2 * box_margin
    box_w = min(3.0, duration)
    corner_radius = 0.04 * duration  # scale radius to audio length

    rect = mpatches.FancyBboxPatch(
        (0, box_y),
        box_w,
        box_h,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        linewidth=1.8,
        edgecolor="#22c55e",   # Tailwind green-500
        facecolor="none",
    )
    ax.add_patch(rect)

    fig.tight_layout(pad=0)

    out_path = wav_path.parent / "waveform.jpg"
    fig.savefig(out_path, format="jpg", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return out_path


def process_directory(root: Path, verbose: bool = True) -> None:
    """Walk *root* recursively and generate a waveform image for every .wav file."""
    wav_files = sorted(root.rglob("*.wav"))

    if not wav_files:
        print(f"No .wav files found under: {root}")
        return

    print(f"Found {len(wav_files)} .wav file(s) under: {root}\n")

    success, failed = 0, 0
    for wav_path in wav_files:
        try:
            out = generate_waveform_image(wav_path)
            if verbose:
                print(f"  ✔  {wav_path.relative_to(root)}  →  {out.name}")
            success += 1
        except Exception as exc:
            print(f"  ✘  {wav_path.relative_to(root)}  —  {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone: {success} succeeded, {failed} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate waveform.jpeg for every .wav file found in a directory tree."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Root directory to search (default: current directory)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress per-file output",
    )
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    process_directory(root, verbose=not args.quiet)


if __name__ == "__main__":
    main()