#!/usr/bin/env python3
"""Item 7a — static validation of handler.NODE_MAP against the workflow JSONs.

Runs at Docker build time (and in CI, before the image build) so node-map drift fails
the BUILD rather than surfacing as a wrong-but-plausible video after a 40 GB cold boot.

For every (mode, semantic field) in NODE_MAP this asserts:
  1. no placeholder IDs remain      — the template has actually been flattened
  2. the node ID exists             — catches renumbering
  3. class_type matches             — catches renumbering onto a valid but wrong node
  4. the input key already exists   — catches schema drift, e.g. writing `cfg` to
                                      LTXVDualCFGGuider (which declares video_cfg/audio_cfg)

What this canNOT prove — and why check_node_registration.sh exists:
  * that the class actually REGISTERED in the built image (a custom node whose deps
    failed to install still leaves valid-looking class_type strings in the JSON)
  * that ComfyUI accepts the mutated prompt

Exit 0 = clean, 1 = drift.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# handler.py sits at / in the image and at ../handler.py in the repo.
for candidate in ("/", os.path.dirname(HERE)):
    if os.path.exists(os.path.join(candidate, "handler.py")):
        sys.path.insert(0, candidate)
        break

# handler.py imports runpod, which is absent in a bare CI runner. Stub it — we only
# need the module-level NODE_MAP, not the serverless runtime.
try:
    import runpod  # noqa: F401
except ImportError:  # pragma: no cover
    import types
    stub = types.ModuleType("runpod")
    stub.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules["runpod"] = stub

from handler import NODE_MAP, PLACEHOLDER_PREFIX  # noqa: E402

WORKFLOWS = {
    "generated_audio": ["/workflow_generated_audio.json",
                        os.path.join(os.path.dirname(HERE), "workflow_generated_audio.json")],
    "custom_audio": ["/workflow_custom_audio.json",
                     os.path.join(os.path.dirname(HERE), "workflow_custom_audio.json")],
}


def _load(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return p, json.load(f)
    return None, None


def main() -> int:
    failures = []
    placeholders = []

    for mode, refs in NODE_MAP.items():
        for field, ref in refs.items():
            if ref.node_id.startswith(PLACEHOLDER_PREFIX):
                placeholders.append(f"{mode}.{field} -> {ref.node_id}")

    if placeholders:
        print("✗ NODE_MAP still contains placeholder node IDs.\n")
        print("  The Comfy-Org template (workflow_templates v0.11.39 ->")
        print("  templates/video_ltx2_5_i2v.json) has not been flattened to API format yet,")
        print("  so the real node IDs are not known. Flatten it, then replace every TBD:*")
        print("  entry in handler.NODE_MAP with the post-flatten ID.\n")
        for p in sorted(placeholders):
            print(f"    {p}")
        print(f"\n  {len(placeholders)} placeholder(s) remaining.")
        return 1

    for mode, refs in NODE_MAP.items():
        path, graph = _load(WORKFLOWS[mode])
        if graph is None:
            failures.append(f"{mode}: workflow JSON not found (looked in {WORKFLOWS[mode]})")
            continue
        print(f"checking {mode} against {path} ({len(graph)} nodes)")
        for field, ref in refs.items():
            node = graph.get(ref.node_id)
            if node is None:
                failures.append(
                    f"{mode}.{field}: node {ref.node_id!r} not in workflow")
                continue
            actual = node.get("class_type")
            if actual != ref.class_type:
                failures.append(
                    f"{mode}.{field}: node {ref.node_id} is {actual!r}, "
                    f"expected {ref.class_type!r}")
                continue
            inputs = node.get("inputs", {})
            if ref.input_key not in inputs:
                failures.append(
                    f"{mode}.{field}: node {ref.node_id} ({actual}) has no input "
                    f"{ref.input_key!r} (declared: {sorted(inputs)})")

    if failures:
        print("\n✗ NODE MAP DRIFT:")
        for f in failures:
            print(f"    {f}")
        return 1

    total = sum(len(r) for r in NODE_MAP.values())
    print(f"✓ node map clean — {total} mapped inputs across {len(NODE_MAP)} modes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
