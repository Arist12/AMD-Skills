#!/usr/bin/env python3
"""Verification gate for inference optimization.

Run after each optimization phase to check whether the latency target is met.
Exit code 0 = target met (agent may STOP), exit code 1 = target NOT met (agent MUST CONTINUE).

Usage:
    python verification-gate.py --target-ms 24.0 --benchmark-cmd "python benchmark.py"

Or inline (without a separate benchmark command):
    python verification-gate.py --target-ms 24.0 --p50-ms 43.2 --phase "Phase 1: CUDAGraph"
"""

import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Verification gate for latency optimization")
    parser.add_argument("--target-ms", type=float, required=True, help="Target p50 latency in ms")
    parser.add_argument("--p50-ms", type=float, default=None, help="Current p50 latency (if already measured)")
    parser.add_argument("--phase", type=str, default="unknown", help="Current optimization phase name")
    parser.add_argument("--benchmark-cmd", type=str, default=None,
                        help="Command to run benchmark (must print JSON with 'p50' key)")
    parser.add_argument("--baseline-ms", type=float, default=None, help="Baseline p50 for comparison")
    args = parser.parse_args()

    p50 = args.p50_ms

    if p50 is None and args.benchmark_cmd:
        result = subprocess.run(args.benchmark_cmd, shell=True, capture_output=True, text=True)
        try:
            data = json.loads(result.stdout)
            p50 = data.get("p50")
        except (json.JSONDecodeError, KeyError):
            for line in result.stdout.splitlines():
                if "p50" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "p50" in part.lower() and i + 1 < len(parts):
                            try:
                                p50 = float(parts[i + 1].replace("ms", "").replace(":", ""))
                            except ValueError:
                                continue

    if p50 is None:
        print("ERROR: Could not determine p50 latency. Provide --p50-ms or --benchmark-cmd.")
        sys.exit(2)

    print("=" * 60)
    print(f"VERIFICATION GATE - {args.phase}")
    print("=" * 60)
    print(f"  Current p50:  {p50:.2f} ms")
    print(f"  Target:       {args.target_ms:.2f} ms")
    if args.baseline_ms:
        improvement = (args.baseline_ms - p50) / args.baseline_ms * 100
        print(f"  Baseline:     {args.baseline_ms:.2f} ms")
        print(f"  Improvement:  {improvement:.1f}%")
    gap = p50 - args.target_ms
    print(f"  Gap:          {gap:+.2f} ms")
    print()

    if p50 <= args.target_ms:
        print(f"TARGET MET: {p50:.2f}ms <= {args.target_ms:.2f}ms")
        print("Agent MAY choose STOP.")
        sys.exit(0)
    else:
        print(f"TARGET NOT MET: {p50:.2f}ms > {args.target_ms:.2f}ms")
        print("Agent MUST CONTINUE to next optimization phase.")
        print(f"Remaining gap: {gap:.2f}ms ({gap/args.target_ms*100:.0f}% of target)")
        sys.exit(1)


if __name__ == "__main__":
    main()
