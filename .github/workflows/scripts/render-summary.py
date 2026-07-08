#!/usr/bin/env python3
"""Aggregate per-model HotSwap CI result JSONs into ONE consolidated markdown
summary and append it to $GITHUB_STEP_SUMMARY (falls back to stdout).

Per-model result JSONs (result-*/model-result.json) have the shape:
    {"lane","model","state","detail"}  with state in pass|diverged|fail|skip.

An optional manifest.json (uploaded as the result-manifest artifact) records the
exact versions under test and is rendered as a "Versions under test" table at the
top.

State -> emoji:
    pass -> :white_check_mark:  diverged -> :warning:  fail -> :x:  skip -> :fast_forward:

Usage: render-summary.py <dir-with-json-files>
"""
import glob
import json
import os
import sys

EMOJI = {
    "pass":     ":white_check_mark:",
    "diverged": ":warning:",
    "fail":     ":x:",
    "skip":     ":fast_forward:",
}
LANE_ORDER = ["SGLang E2E gfx950", "SGLang E2E gfx942", "Pytorch E2E gfx950"]


def load_results(root):
    results = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        if os.path.basename(path) == "manifest.json":
            continue
        try:
            obj = json.load(open(path))
        except (OSError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        obj.setdefault("lane", "Unknown lane")
        obj.setdefault("model", os.path.basename(path))
        st = obj.get("state", "fail")
        obj["state"] = st if st in EMOJI else "fail"
        obj.setdefault("detail", "")
        results.append(obj)
    return results


def load_manifest(root):
    for path in sorted(glob.glob(os.path.join(root, "**", "manifest.json"), recursive=True)):
        try:
            return json.load(open(path))
        except (OSError, ValueError):
            return None
    return None


def lane_sort_key(lane):
    return (LANE_ORDER.index(lane) if lane in LANE_ORDER else len(LANE_ORDER), lane)


def render_manifest(m):
    def nz(x):
        return x if x else "unknown"
    pr = m.get("llvm_project_pr", {}) or {}
    ri = m.get("runner_image", {}) or {}
    bi = m.get("build_lit_image", {}) or {}
    pr_val = f"`{nz(pr.get('commit'))}`"
    if pr.get("ref"):
        pr_val += f" (`{pr['ref']}`)"
    rows = [
        ("llvm-project (PR under test)", pr_val),
        ("rocm-hotswap-testing (harness)", f"`{nz(m.get('rocm_hotswap_testing'))}`"),
        ("rocm-systems (ROCR runtime)", f"`{nz(m.get('rocm_systems'))}`"),
        ("baked llvm-acc (link tree)", f"`{nz(m.get('llvm_acc_baked'))}`"),
        ("ROCm", f"`{nz(m.get('rocm_version'))}`"),
        ("runner image", f"`{nz(ri.get('ref'))}`"),
        ("build+lit image", f"`{nz(bi.get('ref'))}`"),
    ]
    out = ["## Versions under test", "", "| Component | Version |", "| --- | --- |"]
    out += [f"| {k} | {v} |" for k, v in rows]
    out.append("")
    return out


def render(results, manifest):
    out = ["# HotSwap PR CI — Model Results", ""]
    if manifest:
        out += render_manifest(manifest)
    out.append(
        "Legend: :white_check_mark: pass &nbsp; :warning: ran but numerically "
        "diverged (not gated) &nbsp; :x: failed &nbsp; :fast_forward: skipped"
    )
    out.append("")
    if not results:
        out.append(
            "> No model result artifacts were found. The E2E lanes were likely "
            "skipped (e.g. `build + lit` failed) or produced no results."
        )
        return "\n".join(out) + "\n"

    lanes = {}
    for r in results:
        lanes.setdefault(r["lane"], []).append(r)

    for lane in sorted(lanes, key=lane_sort_key):
        rows = sorted(lanes[lane], key=lambda r: r["model"])
        counts = {k: 0 for k in EMOJI}
        for r in rows:
            counts[r["state"]] += 1
        tally = (
            f"{counts['pass']} pass / {counts['diverged']} diverged / "
            f"{counts['fail']} fail / {counts['skip']} skip"
        )
        out += [
            f"## {lane}", "",
            f"_{len(rows)} models — {tally}_", "",
            "| Model | Result | Equivalence / Detail |",
            "| --- | :---: | --- |",
        ]
        for r in rows:
            detail = str(r.get("detail", "")).replace("|", r"\|").replace("\n", " ")
            out.append(f"| `{r['model']}` | {EMOJI[r['state']]} {r['state']} | {detail} |")
        out.append("")
    return "\n".join(out) + "\n"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    md = render(load_results(root), load_manifest(root))
    dest = os.environ.get("GITHUB_STEP_SUMMARY")
    if dest:
        with open(dest, "a") as fh:
            fh.write(md)
    sys.stdout.write(md)


if __name__ == "__main__":
    main()
