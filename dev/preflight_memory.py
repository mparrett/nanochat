#!/usr/bin/env python3
"""Flag heavy memory consumers before launching training on memory-bound machines.

Designed for M2 24GB hygiene per CLAUDE.md. Run before `scripts/base_train.py`,
`scripts/chat_sft.py`, etc. Output is informational and a one-line
recommendation; exit code is 1 if free memory is critically low so the script
can gate launches in a shell wrapper.

Lesson from the 2026-05-10 d8 overnight run: background workloads (Docker
daemon, server.py, faprox.py, dashboard, crypto_bot dry-run) consumed enough
unified memory to push training into 38-65 s step-time spikes. Killing them
dropped pages active+inactive from 13 GB to 2.8 GB and restored ~6 s step
times — a 4× wall-clock difference for ~5 min of cleanup.

Usage:
  python3 dev/preflight_memory.py
"""
import re
import subprocess
import sys

PAGE = 16384  # macOS uses 16 KB pages on Apple Silicon


def vm_stat_summary() -> dict[str, float]:
    out = subprocess.check_output(["vm_stat"], text=True)
    stats = {}
    for line in out.splitlines():
        m = re.match(r"Pages (\w+).*?(\d+)\.", line)
        if m:
            stats[m.group(1)] = int(m.group(2)) * PAGE / (1024**3)
    return stats


def top_consumers(n: int = 10, min_mb: float = 100) -> list[tuple[float, str, str]]:
    out = subprocess.check_output(
        ["ps", "-axo", "pid,rss,comm"], text=True
    ).splitlines()
    rows = []
    for line in out[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rss_mb = int(parts[1]) / 1024
        except ValueError:
            continue
        if rss_mb >= min_mb:
            rows.append((rss_mb, parts[0], parts[2]))
    rows.sort(reverse=True)
    return rows[:n]


def suspect_processes() -> dict[str, list[str]]:
    # Patterns observed during 2026-05-10 to cause paging spikes during training.
    # Add to this list when new offenders surface.
    patterns = ["docker", "Docker", "server.py", "dashboard.py",
                "crypto_bot", "faprox", "node"]
    found = {}
    for pat in patterns:
        try:
            out = subprocess.check_output(
                ["pgrep", "-lf", pat], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except subprocess.CalledProcessError:
            continue
        if out:
            lines = [l for l in out.splitlines() if "preflight_memory" not in l]
            if lines:
                found[pat] = lines[:3]
    return found


def main() -> int:
    print("=== Memory state ===")
    stats = vm_stat_summary()
    for k in ("free", "active", "inactive", "wired"):
        if k in stats:
            print(f"  {k:9s} {stats[k]:5.2f} GB")

    print("\n=== Top memory consumers (RSS ≥ 100 MB) ===")
    for rss, pid, comm in top_consumers():
        print(f"  pid {pid:>6}  {rss:7.1f} MB  {comm}")

    print("\n=== Notable suspect processes ===")
    suspects = suspect_processes()
    if suspects:
        for pat, lines in suspects.items():
            print(f"  ⚠ {pat}:")
            for line in lines:
                print(f"      {line}")
    else:
        print("  ✓ None of the usual suspects running")

    print("\n=== Recommendation ===")
    free_gb = stats.get("free", 0)
    active = stats.get("active", 0)
    inactive = stats.get("inactive", 0)
    pressure = active + inactive
    if free_gb < 1:
        print(f"  ❌ Free memory {free_gb:.2f} GB — expect HEAVY paging during training.")
        print(f"     pages active+inactive = {pressure:.1f} GB. Kill suspects above or")
        print(f"     accept 4-10× slower step times. Returning exit 1.")
        return 1
    elif free_gb < 4 or pressure > 8:
        print(f"  ⚠ Free {free_gb:.2f} GB, pressure {pressure:.1f} GB — some paging likely.")
        print(f"     Consider killing suspects before a long run.")
    else:
        print(f"  ✓ Free {free_gb:.2f} GB, pressure {pressure:.1f} GB — looks OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
