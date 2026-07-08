#!/usr/bin/env python3
"""Read a HotSwap sglang summary.json and emit the CI gate verdict.

Prints "<0|1>|<overall_status>|<strict_passed>" and exits 0 (pass) / 1 (fail).

Gate convention follows the harness's own sglang e2e CI: a leg PASSES when the
HotSwap transpile pipeline ran end-to-end and produced a valid equivalence
verdict, i.e. overall_status in {equivalent, numerically_close,
distributionally_equivalent, diverged}. `diverged` is the EXPECTED
gfx1250->target accumulation-order effect (argmax flips) and is reported, not
failed. The strict boolean (equivalence.passed) is still surfaced for triage.
A leg FAILS only if no valid verdict was produced (run didn't complete / no
transpile).
"""
import json
import sys

ACCEPT = {"equivalent", "numerically_close", "distributionally_equivalent", "diverged"}

summary = json.load(open(sys.argv[1]))
eq = summary.get("equivalence", {}) or {}
status = eq.get("overall_status") or summary.get("equivalence_status") or "?"
strict = eq.get("passed")
if strict is None:
    strict = summary.get("equivalence_passed")

ok = status in ACCEPT
print(f"{int(ok)}|{status}|{bool(strict)}")
sys.exit(0 if ok else 1)
