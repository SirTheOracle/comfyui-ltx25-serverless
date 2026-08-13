#!/usr/bin/env python3
"""End-to-end test against a deployed LTX-2.5 RunPod serverless endpoint.

This is the test that answers the two questions research cannot: does the model set fit
in VRAM, and does audio-driven lip-sync actually work. It must run against a real
SERVERLESS endpoint, not a plain pod — only serverless exercises RunPod's init/health
kill window, which is the failure mode that crash-looped LTX-2.3.

    RUNPOD_API_KEY=... ENDPOINT_ID=... python3 tests/e2e_endpoint.py both

Modes: generated_audio | custom_audio | both

Asserts the response contract (which is backward-compatible with LTX-2.3, see README),
saves the decoded mp4, and for custom_audio verifies the decoded frame count matches the
ceil-to-grid rule — a floor would silently truncate the tail of the dialogue.
"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.ai/v2"
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.environ.get("FIXTURES", "/tmp/ltx25test")
OUTDIR = os.environ.get("OUTDIR", "/tmp/ltx25test")

REQUIRED_SUCCESS_KEYS = {"ok", "video", "seed", "mode", "parameters", "timings", "worker"}


def _key():
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        sys.exit("RUNPOD_API_KEY is required")
    return k


def _endpoint():
    e = os.environ.get("ENDPOINT_ID")
    if not e:
        sys.exit("ENDPOINT_ID is required")
    return e


def _post(path, body, key, timeout=60):
    req = urllib.request.Request(f"{API}/{_endpoint()}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def _get(path, key, timeout=60):
    req = urllib.request.Request(f"{API}/{_endpoint()}{path}",
                                 headers={"Authorization": f"Bearer {key}"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def probe(path):
    """(duration, frames, width, height, has_audio) via ffprobe."""
    def q(args):
        r = subprocess.run(["ffprobe", "-v", "error", *args, path],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    dur = q(["-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
    v = q(["-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=nb_read_frames,width,height",
           "-of", "default=nw=1:nk=1"]).splitlines()
    a = q(["-select_streams", "a:0", "-show_entries", "stream=codec_type",
           "-of", "default=nw=1:nk=1"])
    w, h, frames = (int(v[0]), int(v[1]), int(v[2])) if len(v) >= 3 else (0, 0, 0)
    return float(dur or 0), frames, w, h, bool(a)


def run(mode, key, poll=10, budget=1800):
    print(f"\n{'='*72}\n{mode}\n{'='*72}")
    payload = {
        "image": b64(os.path.join(FIXTURES, "subject.png")),
        "prompt": ("A calm figure under moonlight speaks softly to the camera, natural "
                   "head movement, eyes engaged, gentle night breeze, cinematic soft light."),
        "seed": 12345,
    }
    if mode == "custom_audio":
        payload["audio"] = b64(os.path.join(FIXTURES, "speech.mp3"))
        payload.update(width=720, height=1280, fps=24)
        expect_frames = 81          # 1 + ceil(3.25*24/8)*8 — NOT 73 (floor would truncate)
    else:
        payload.update(width=1280, height=720, fps=25, frame_count=121)
        expect_frames = 121

    t0 = time.time()
    sub = _post("/run", {"input": payload}, key)
    job = sub["id"]
    print(f"  submitted {job}")

    last = None
    while time.time() - t0 < budget:
        st = _get(f"/status/{job}", key)
        s = st.get("status")
        if s != last:
            h = _get("/health", key)["workers"]
            print(f"  {int(time.time()-t0):>4}s  {s:<12} workers "
                  f"init={h['initializing']} ready={h['ready']} run={h['running']} "
                  f"unhealthy={h['unhealthy']}")
            last = s
        if s == "COMPLETED":
            break
        if s in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(f"  ✗ {s}")
            print(json.dumps(st, indent=2)[:1800])
            return False
        time.sleep(poll)
    else:
        print("  ✗ budget exhausted"); return False

    out = st.get("output") or {}
    delay_s = st.get("delayTime", 0) / 1000
    exec_s = st.get("executionTime", 0) / 1000
    print(f"  delayTime={delay_s:.1f}s  executionTime={exec_s:.1f}s")

    ok = True
    missing = REQUIRED_SUCCESS_KEYS - set(out)
    if missing:
        print(f"  ✗ response missing contract keys: {sorted(missing)}"); ok = False
    if out.get("ok") is not True:
        print(f"  ✗ ok={out.get('ok')} error={out.get('error')}"); ok = False
    if out.get("mode") != mode:
        print(f"  ✗ mode={out.get('mode')!r}, expected {mode!r}"); ok = False
    if not out.get("video"):
        print("  ✗ no video payload"); return False

    p = out.get("parameters", {})
    print(f"  parameters: frame_count={p.get('frame_count')} {p.get('width')}x{p.get('height')} "
          f"@{p.get('fps')} cfg={p.get('cfg')} audio_cfg={p.get('audio_cfg')} steps={p.get('steps')}")
    if p.get("frame_count") != expect_frames:
        print(f"  ✗ echoed frame_count {p.get('frame_count')} != expected {expect_frames}"); ok = False

    if mode == "custom_audio":
        ad = out.get("audio_duration")
        print(f"  audio_duration={ad}")
        if ad is None:
            print("  ✗ audio_duration missing for custom_audio"); ok = False

    dest = os.path.join(OUTDIR, f"ltx25_{mode}.mp4")
    with open(dest, "wb") as f:
        f.write(base64.b64decode(out["video"]))
    dur, frames, w, h, has_audio = probe(dest)
    print(f"  saved {dest} ({os.path.getsize(dest):,} B)")
    print(f"  decoded: {frames} frames, {w}x{h}, {dur:.2f}s, audio={'yes' if has_audio else 'NO'}")

    if frames != expect_frames:
        print(f"  ✗ decoded {frames} frames, expected {expect_frames}"); ok = False
    if not has_audio:
        print("  ✗ output has no audio stream"); ok = False
    if mode == "custom_audio":
        print("  ⚠ LIP-SYNC REQUIRES A HUMAN. A frame-count assertion passes whether or")
        print("    not the mouth moves. Watch the file and confirm the final phoneme survives.")

    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    key = _key()
    modes = ["generated_audio", "custom_audio"] if mode == "both" else [mode]
    results = {m: run(m, key) for m in modes}
    print(f"\n{'='*72}")
    for m, r in results.items():
        print(f"  {m:<18} {'PASS' if r else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
