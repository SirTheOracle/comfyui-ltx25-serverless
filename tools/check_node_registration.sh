#!/bin/bash
# Item 7b — model-free ComfyUI node-registration check. Runs as a Dockerfile RUN step.
#
# WHY THIS EXISTS, separately from check_node_map.py:
# check_node_map.py validates the workflow JSON against handler.py's map. It proves
# node IDs, class_types and input keys agree ON PAPER. It cannot prove that a class
# actually REGISTERED in the built image.
#
# That gap is real: every custom-node `pip install` in the Dockerfile ends in `|| true`
# (deliberately — their requirements pin conflicting torch versions). So a custom node
# whose dependencies failed to install still yields a green build and fails at RUNTIME,
# after a 40 GB cold boot, with a confusing "node type not found" from ComfyUI.
#
# This script closes that gap: boot ComfyUI with NO models present, read /object_info,
# and assert every class_type used by either workflow is registered. Turns a silent
# runtime failure into a named build failure.
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-/comfyui}"
PORT="${PORT:-8188}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-180}"

echo "=========================================="
echo "Item 7b: model-free node registration check"
echo "=========================================="

cd "$COMFY_DIR"
python main.py --listen 127.0.0.1 --port "$PORT" --disable-auto-launch --cpu >/tmp/comfy_boot.log 2>&1 &
COMFY_PID=$!
# shellcheck disable=SC2064
trap "kill $COMFY_PID 2>/dev/null || true" EXIT

echo "Waiting for ComfyUI to boot (max ${BOOT_TIMEOUT}s)..."
deadline=$((SECONDS + BOOT_TIMEOUT))
until curl -sf "http://127.0.0.1:${PORT}/object_info" -o /tmp/object_info.json 2>/dev/null; do
    if ! kill -0 "$COMFY_PID" 2>/dev/null; then
        echo "✗ ComfyUI died during boot. Last 60 lines:"
        tail -60 /tmp/comfy_boot.log
        exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "✗ ComfyUI did not answer /object_info within ${BOOT_TIMEOUT}s. Last 60 lines:"
        tail -60 /tmp/comfy_boot.log
        exit 1
    fi
    sleep 2
done
echo "✓ ComfyUI booted; /object_info retrieved"

python - <<'PY'
import json, sys, pathlib

registered = set(json.load(open("/tmp/object_info.json")))
print(f"registered node classes: {len(registered)}")

required = {}
for wf in ("/workflow_generated_audio.json", "/workflow_custom_audio.json"):
    p = pathlib.Path(wf)
    if not p.exists():
        print(f"✗ missing workflow: {wf}")
        sys.exit(1)
    graph = json.loads(p.read_text())
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type")
        if ct:
            required.setdefault(ct, []).append(f"{p.name}:{node_id}")

missing = {c: locs for c, locs in required.items() if c not in registered}
print(f"classes used by workflows: {len(required)}")

if missing:
    print("\n✗ NODE CLASSES NOT REGISTERED IN THIS IMAGE:")
    for c, locs in sorted(missing.items()):
        print(f"    {c:<34} used at {', '.join(locs)}")
    print("\nThis almost always means a custom-node import failed (see the `|| true` on")
    print("the custom-node pip installs in the Dockerfile). Check ComfyUI's boot log for")
    print("the import traceback; grep /tmp/comfy_boot.log for 'Cannot import' / 'Traceback'.")
    sys.exit(1)

print("✓ every class_type used by both workflows is registered")
PY

echo "✓ Item 7b passed"
