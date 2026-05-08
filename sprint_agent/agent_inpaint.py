#!/usr/bin/env python3
"""30-minute MVP: one inpainting method + tiny agent loop + evaluation.

This intentionally avoids Python package dependencies. It uses the system
`pngtopnm` and `pnmtopng` tools for PNG I/O, then does the image math in plain
Python lists. The agent is heuristic today, but its JSON action schema is the
same shape you can later swap for an LLM response.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Image:
    width: int
    height: int
    pixels: list[list[float]]


@dataclass
class Attempt:
    iteration: int
    thought: str
    tool: str
    params: dict
    psnr: float
    masked_psnr: float
    ssim: float
    reflection: str
    output: str


def read_png(path: Path) -> Image:
    """Read a PNG as grayscale floats in [0, 255] via pngtopnm."""
    proc = subprocess.run(
        ["pngtopnm", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    data = proc.stdout
    magic, pos = _read_token(data, 0)
    if magic not in {b"P5", b"P6"}:
        raise ValueError(f"Expected P5/P6 PNM from pngtopnm, got {magic!r}")
    width_raw, pos = _read_token(data, pos)
    height_raw, pos = _read_token(data, pos)
    maxval_raw, pos = _read_token(data, pos)
    width, height, maxval = int(width_raw), int(height_raw), int(maxval_raw)
    if maxval != 255:
        raise ValueError(f"Only 8-bit images are supported; got maxval={maxval}")

    channels = 1 if magic == b"P5" else 3
    expected = width * height * channels
    raw = data[pos : pos + expected]
    if len(raw) != expected:
        raise ValueError(f"PNM payload is truncated for {path}")

    pixels: list[list[float]] = []
    idx = 0
    for _y in range(height):
        row: list[float] = []
        for _x in range(width):
            if channels == 1:
                row.append(float(raw[idx]))
                idx += 1
            else:
                r, g, b = raw[idx], raw[idx + 1], raw[idx + 2]
                row.append(0.299 * r + 0.587 * g + 0.114 * b)
                idx += 3
        pixels.append(row)
    return Image(width, height, pixels)


def write_png(image: Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P5\n{image.width} {image.height}\n255\n".encode()
    payload = bytearray()
    for row in image.pixels:
        for val in row:
            payload.append(max(0, min(255, int(round(val)))))
    proc = subprocess.run(
        ["pnmtopng"],
        input=header + bytes(payload),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    path.write_bytes(proc.stdout)


def _read_token(data: bytes, pos: int) -> tuple[bytes, int]:
    while pos < len(data):
        c = data[pos]
        if c == 35:  # comment
            while pos < len(data) and data[pos] not in (10, 13):
                pos += 1
        elif chr(c).isspace():
            pos += 1
        else:
            break
    start = pos
    while pos < len(data) and not chr(data[pos]).isspace():
        pos += 1
    token = data[start:pos]
    while pos < len(data) and chr(data[pos]).isspace():
        pos += 1
        break
    return token, pos


def infer_mask(damaged: Image, gt: Image, threshold: float = 8.0) -> list[list[bool]]:
    _assert_same_size(damaged, gt)
    dark_mask = [[damaged.pixels[y][x] <= 2.0 for x in range(gt.width)] for y in range(gt.height)]
    dark_pixels = sum(sum(1 for v in row if v) for row in dark_mask)
    if dark_pixels > 0:
        return dark_mask
    return [
        [abs(damaged.pixels[y][x] - gt.pixels[y][x]) > threshold for x in range(gt.width)]
        for y in range(gt.height)
    ]


def directional_neighbor_fill(
    damaged: Image,
    mask: list[list[bool]],
    *,
    horizontal_weight: float = 0.5,
    vertical_weight: float = 0.5,
) -> Image:
    """Fill masked pixels by interpolating from nearest known row/column pixels."""
    height, width = damaged.height, damaged.width
    current = [row[:] for row in damaged.pixels]

    left_val = [[None for _ in range(width)] for _ in range(height)]
    right_val = [[None for _ in range(width)] for _ in range(height)]
    up_val = [[None for _ in range(width)] for _ in range(height)]
    down_val = [[None for _ in range(width)] for _ in range(height)]

    for y in range(height):
        last = None
        for x in range(width):
            if not mask[y][x]:
                last = damaged.pixels[y][x]
            left_val[y][x] = last
        last = None
        for x in range(width - 1, -1, -1):
            if not mask[y][x]:
                last = damaged.pixels[y][x]
            right_val[y][x] = last

    for x in range(width):
        last = None
        for y in range(height):
            if not mask[y][x]:
                last = damaged.pixels[y][x]
            up_val[y][x] = last
        last = None
        for y in range(height - 1, -1, -1):
            if not mask[y][x]:
                last = damaged.pixels[y][x]
            down_val[y][x] = last

    for y in range(height):
        for x in range(width):
            if not mask[y][x]:
                continue
            horizontal = _mean_known([left_val[y][x], right_val[y][x]])
            vertical = _mean_known([up_val[y][x], down_val[y][x]])
            values = []
            weights = []
            if horizontal is not None:
                values.append(horizontal)
                weights.append(horizontal_weight)
            if vertical is not None:
                values.append(vertical)
                weights.append(vertical_weight)
            if values and sum(weights) > 0:
                current[y][x] = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    return Image(width, height, current)


def _mean_known(values: Iterable[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    if not known:
        return None
    return sum(known) / len(known)


def psnr(restored: Image, gt: Image, mask: list[list[bool]] | None = None) -> float:
    mse = mean_squared_error(restored, gt, mask)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def mean_squared_error(restored: Image, gt: Image, mask: list[list[bool]] | None = None) -> float:
    _assert_same_size(restored, gt)
    total = 0.0
    count = 0
    for y in range(gt.height):
        for x in range(gt.width):
            if mask is not None and not mask[y][x]:
                continue
            diff = restored.pixels[y][x] - gt.pixels[y][x]
            total += diff * diff
            count += 1
    return total / max(1, count)


def global_ssim(restored: Image, gt: Image) -> float:
    """Simple global SSIM approximation, dependency-free."""
    _assert_same_size(restored, gt)
    xs = [v for row in restored.pixels for v in row]
    ys = [v for row in gt.pixels for v in row]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs) / max(1, n - 1)
    var_y = sum((y - mean_y) ** 2 for y in ys) / max(1, n - 1)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / max(1, n - 1)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return ((2 * mean_x * mean_y + c1) * (2 * cov + c2)) / (
        (mean_x * mean_x + mean_y * mean_y + c1) * (var_x + var_y + c2)
    )


def choose_action(history: list[Attempt]) -> dict:
    """Tiny agent policy. Replace this with an LLM call later."""
    candidates = [
        {"horizontal_weight": 0.5, "vertical_weight": 0.5},
        {"horizontal_weight": 0.8, "vertical_weight": 0.2},
        {"horizontal_weight": 0.2, "vertical_weight": 0.8},
    ]
    tried = {json.dumps(a.params, sort_keys=True) for a in history}
    for params in candidates:
        if json.dumps(params, sort_keys=True) not in tried:
            reason = "start sharp/local" if not history else "try a smoother wider neighborhood"
            return {
                "thought": reason,
                "tool": "directional_neighbor_fill",
                "params": params,
            }
    best = max(history, key=lambda a: a.masked_psnr)
    return {
        "thought": "no untried directional-fill configs remain; stop with best result",
        "tool": "stop",
        "params": {"best_iteration": best.iteration},
    }


def reflect(current: Attempt, previous_best: Attempt | None) -> str:
    if previous_best is None:
        return "First scored reconstruction; use it as the baseline for the next decision."
    delta = current.masked_psnr - previous_best.masked_psnr
    if delta > 0.25:
        return f"Improved masked PSNR by {delta:.2f} dB; this parameter direction is promising."
    if delta < -0.25:
        return f"Masked PSNR dropped by {-delta:.2f} dB; avoid this smoothing level next."
    return "Quality is roughly tied with the best attempt; prefer the simpler/sharper result."


def run_agent(damaged_path: Path, gt_path: Path, run_dir: Path, max_iters: int) -> None:
    damaged = read_png(damaged_path)
    gt = read_png(gt_path)
    mask = infer_mask(damaged, gt)
    write_png(Image(damaged.width, damaged.height, [[255.0 if v else 0.0 for v in row] for row in mask]), run_dir / "mask.png")

    attempts: list[Attempt] = []
    for iteration in range(1, max_iters + 1):
        action = choose_action(attempts)
        if action["tool"] == "stop":
            break

        previous_best = max(attempts, key=lambda a: a.masked_psnr) if attempts else None
        restored = directional_neighbor_fill(damaged, mask, **action["params"])
        output_path = run_dir / f"attempt_{iteration:02d}_{action['tool']}.png"
        write_png(restored, output_path)
        attempt = Attempt(
            iteration=iteration,
            thought=action["thought"],
            tool=action["tool"],
            params=action["params"],
            psnr=psnr(restored, gt),
            masked_psnr=psnr(restored, gt, mask),
            ssim=global_ssim(restored, gt),
            reflection="",
            output=str(output_path.relative_to(ROOT)),
        )
        attempt.reflection = reflect(attempt, previous_best)
        attempts.append(attempt)

    best = max(attempts, key=lambda a: a.masked_psnr)
    best_image = read_png(ROOT / best.output)
    write_png(best_image, run_dir / "best.png")
    (run_dir / "agent_log.json").write_text(
        json.dumps(
            {
                "damaged": str(damaged_path.relative_to(ROOT)),
                "ground_truth": str(gt_path.relative_to(ROOT)),
                "mask_pixels": sum(sum(1 for v in row if v) for row in mask),
                "attempts": [asdict(a) for a in attempts],
                "best_iteration": best.iteration,
                "best_masked_psnr": best.masked_psnr,
                "best_ssim": best.ssim,
            },
            indent=2,
        )
    )
    write_metrics_csv(attempts, run_dir / "metrics.csv")
    print(f"Best iteration: {best.iteration}")
    print(f"Best masked PSNR: {best.masked_psnr:.3f} dB")
    print(f"Best global PSNR: {best.psnr:.3f} dB")
    print(f"Best SSIM approx: {best.ssim:.5f}")
    print(f"Run dir: {run_dir}")


def write_metrics_csv(attempts: list[Attempt], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "tool", "params", "psnr", "masked_psnr", "ssim", "reflection"])
        for attempt in attempts:
            writer.writerow(
                [
                    attempt.iteration,
                    attempt.tool,
                    json.dumps(attempt.params, sort_keys=True),
                    f"{attempt.psnr:.6f}",
                    f"{attempt.masked_psnr:.6f}",
                    f"{attempt.ssim:.6f}",
                    attempt.reflection,
                ]
            )


def _assert_same_size(left: Image, right: Image) -> None:
    if left.width != right.width or left.height != right.height:
        raise ValueError(f"Image sizes differ: {left.width}x{left.height} vs {right.width}x{right.height}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-method agentic inpainting sprint.")
    parser.add_argument("--damaged", type=Path, default=ROOT / "data/Image0-missing_large_region.png")
    parser.add_argument("--gt", type=Path, default=ROOT / "data/Image0-GT.png")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/quick_sprint")
    parser.add_argument("--max-iters", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_agent(args.damaged.resolve(), args.gt.resolve(), args.run_dir.resolve(), args.max_iters)


if __name__ == "__main__":
    main()
