#!/usr/bin/env python3
"""Read a HotSwap PyTorch-path summary.json and emit the CI gate verdict.

Prints "<0|1>|proof=<status>|equiv=<status>|equiv_passed=<bool>" and exits
0 (pass) / 1 (fail).

Gate follows the harness's pytorch e2e convention: a leg PASSES when the
HotSwap transpile pipeline ran end-to-end cleanly — the native (local) branch
passed, the hotswap branch passed, and there were zero failed translations
(hotswap_fail == 0). Numerical equivalence is surfaced but NOT gated (the
gfx1250->target path legitimately drifts; equivalence is a separate quality
signal). A leg FAILS if a branch crashed or any kernel failed to translate.
"""
import json
import sys

s = json.load(open(sys.argv[1]))
l = s.get("local", {}) or {}
h = s.get("hotswap", {}) or {}
eq = s.get("equivalence", {}) or {}

local_ok = l.get("passed") is True
hotswap_ok = h.get("passed") is True
hotswap_fail = int(h.get("hotswap_fail", 0) or 0)
proof = h.get("proof_status") or s.get("proof_status") or "?"
eq_status = eq.get("status") or eq.get("overall_status") or "?"
eq_passed = eq.get("passed")

ok = local_ok and hotswap_ok and hotswap_fail == 0
print(f"{int(ok)}|proof={proof}|equiv={eq_status}|equiv_passed={bool(eq_passed)}")
sys.exit(0 if ok else 1)
