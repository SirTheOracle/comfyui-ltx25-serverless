"""
ComfyUI LTX-2.5 Video Serverless Handler for RunPod

Supports two modes, selected by the TRUTHINESS of `audio` in the job input
(NOT mere key presence — `audio: ""` and `audio: null` both select generated_audio):
  1. generated_audio — LTX-2.5 generates audio from the prompt (landscape default)
  2. custom_audio    — pre-generated audio (e.g. ElevenLabs) drives lip-sync (portrait default)

Forked from comfyui-ltx23-serverless. The external API contract is BACKWARD-COMPATIBLE
with LTX-2.3 — every 2.3-shaped request still works — but it is NOT identical. Deltas:

  D1  generated_audio prompts are no longer VLM-expanded; they are encoded verbatim.
      (LTX-2.5 has no abliterated Gemma LoRA; the stock enhancer is a separate 10.28 GB
      model whose refusal behaviour is unverified. Dropped deliberately — see README.)
  D2  Off-grid `frame_count` is snapped UP to LTX-2.5's 8k+1 temporal grid, logged, and
      echoed back as the snapped value in `parameters.frame_count`.
  D3  New OPTIONAL request field `audio_cfg` (defaults to `cfg`). Absent -> 2.3 behaviour.
  D4  Malformed numeric inputs return non-retryable INVALID_INPUT (was: retryable
      INTERNAL_ERROR, which could quarantine a healthy worker on a bad request).
  D5  Corrupt/undecodable audio returns INVALID_INPUT (was: silently assumed 4.0 s).
  D6  Node-map defects return non-retryable WORKFLOW_EXECUTION_ERROR (was: retryable
      INTERNAL_ERROR). Retrying a baked-in code defect cannot help and burns workers.

Workflow node wiring lives in NODE_MAP below — ONE structured source consumed by both
the runtime (modify_workflow_*) and the static checker (tools/check_node_map.py), so the
two cannot drift apart. If the workflow JSON is regenerated, update NODE_MAP in lockstep;
the Dockerfile runs the checker at build time and will fail the build on a mismatch.
"""

import runpod
import json
import urllib.request
import urllib.error
import base64
import math
import time
import os
import sys
import logging
import traceback
import subprocess
from typing import Optional, Dict, Any, Tuple, NamedTuple
import socket

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logging():
    logger = logging.getLogger("LTX25-Handler")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

DEFAULT_PARAMS_GENERATED = {
    "width": 1280,
    "height": 720,
    "frame_count": 121,          # 8*15+1 — already on the 8k+1 grid
    "fps": 25,
    "cfg": 1.0,
    "audio_cfg": None,           # None -> mirror cfg (reproduces 2.3 single-CFG behaviour)
    "seed": None,
    "timeout": 600,
    "img_compression": 18,
    "i2v_strength_first": 1.0,
    "i2v_strength_second": 0.7,
    "steps": None,
}

DEFAULT_PARAMS_CUSTOM_AUDIO = {
    "width": 720,
    "height": 1280,
    "fps": 24,
    "cfg": 1.0,
    "audio_cfg": None,
    "seed": None,
    "timeout": 600,
    "img_compression": 18,
    "i2v_strength_first": 1.0,
    "i2v_strength_second": 0.7,
    "steps": None,
}

DEFAULT_NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"

# ---------------------------------------------------------------------------
# Sigma schedules — CARRIED VERBATIM FROM LTX-2.3. DO NOT REORDER.
#
# The names are misnomers relative to graph EXECUTION order, but the PAIRING is
# correct and internally consistent, and LTX-2.5 uses the identical schedule:
#   stage 1 (half-res)          runs the 9-value / 8-step list  (_REFINE)
#   stage 2 (post-upscale)      runs the 4-value / 3-step list  (_FIRST)
# Verified against Comfy-Org's video_ltx2_5_i2v.json: node 397 holds the 9-value
# list (stage 1), node 396 holds the 4-value list (stage 2) — byte-identical to
# the constants below. Inverting the 3:8 ratio in _split_steps would introduce a
# regression, not fix one.
# ---------------------------------------------------------------------------
JSON_SIGMAS_FIRST = [0.85, 0.7250, 0.4219, 0.0]
JSON_SIGMAS_REFINE = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
JSON_STEPS_FIRST = len(JSON_SIGMAS_FIRST) - 1
JSON_STEPS_REFINE = len(JSON_SIGMAS_REFINE) - 1
JSON_STEPS_TOTAL = JSON_STEPS_FIRST + JSON_STEPS_REFINE


def _resample_sigmas(reference: list, new_step_count: int) -> str:
    n_sigmas = new_step_count + 1
    ref_len = len(reference)
    if n_sigmas == ref_len:
        return ", ".join(f"{s:g}" for s in reference)
    out = []
    for i in range(n_sigmas):
        pos = i * (ref_len - 1) / (n_sigmas - 1)
        lo = int(pos)
        hi = min(lo + 1, ref_len - 1)
        frac = pos - lo
        out.append(reference[lo] * (1 - frac) + reference[hi] * frac)
    return ", ".join(f"{v:g}" for v in out)


def _split_steps(total: int) -> tuple[int, int]:
    if total <= 0:
        raise ValueError(f"steps must be >= 1, got {total}")
    if total == JSON_STEPS_TOTAL:
        return (JSON_STEPS_FIRST, JSON_STEPS_REFINE)
    first = max(1, round(total * JSON_STEPS_FIRST / JSON_STEPS_TOTAL))
    refine = max(1, total - first)
    return (first, refine)


# ---------------------------------------------------------------------------
# LTX-2.5 temporal grid
# ---------------------------------------------------------------------------
def _snap_frames(n: int) -> int:
    """Snap a frame count UP to LTX-2.5's 8k+1 temporal grid.

    EmptyLTXVLatentVideo declares `length` with step=8 and computes
    latent_frames = ((length - 1) // 8) + 1, so an off-grid length is silently
    truncated by the VAE.

    Rounds UP, not down. LTXVConcatAVLatent.fit_audio resolves a video/audio length
    mismatch asymmetrically: audio LONGER than the video is narrowed (truncated),
    while audio SHORTER than the video is zero-padded with an UNMASKED tail that the
    model generates. Rounding down therefore silently drops 1-7 frames of externally
    supplied dialogue; rounding up costs at most 7 model-generated frames of tail,
    which downstream assembly already trims.
    """
    n = int(n)
    snapped = max(9, 1 + -(-(n - 1) // 8) * 8)   # ceil-div to the 8k+1 grid
    if snapped != n:
        logger.warning("frame_count %s snapped up to %s (LTX-2.5 8k+1 grid)", n, snapped)
    return snapped


def _frames_for_duration(duration_s: float, fps: int) -> int:
    """Frames for an audio clip of `duration_s`, snapped up to the 8k+1 grid.

    Mirrors the graph's ComfyMathExpression `1 + ceil(a*b/8)*8`. Both sites must agree;
    tests/test_frames.py asserts they do across a range of non-grid durations.
    """
    return max(9, 1 + math.ceil((duration_s * fps) / 8) * 8)


INFRA_FAILURE_WINDOW_SECONDS = int(os.getenv("INFRA_FAILURE_WINDOW_SECONDS", "600"))
INFRA_FAILURE_THRESHOLD = int(os.getenv("INFRA_FAILURE_THRESHOLD", "3"))
HARD_EXIT_ON_QUARANTINE = os.getenv("HARD_EXIT_ON_QUARANTINE", "false").lower() == "true"
INFRA_FAILURE_TIMESTAMPS: list[float] = []

INFRA_ERROR_CODES = {
    "COMFYUI_BOOT_TIMEOUT",
    "COMFYUI_UNREACHABLE",
    "WORKFLOW_QUEUE_FAILED",
    "WORKFLOW_TIMEOUT",
    "INTERNAL_ERROR",
}


# ===========================================================================
# STRUCTURED NODE MAP
# ===========================================================================
# ONE source of truth for graph wiring, consumed by:
#   - modify_workflow_generated_audio / modify_workflow_custom_audio (runtime)
#   - tools/check_node_map.py (build-time static validation, Item 7a)
#   - tools/check_node_registration.sh (build-time /object_info check, Item 7b)
#
# !!! NODE IDs ARE PLACEHOLDERS ("TBD:*") !!!
# The real IDs are only knowable after the Comfy-Org template
# (workflow_templates v0.11.39 -> templates/video_ltx2_5_i2v.json) is flattened
# from UI/subgraph format to flat API format, which renumbers every node.
# check_node_map.py FAILS while any TBD:* remains, so this cannot ship half-done.
#
# The `class_type` and `input_key` columns are NOT placeholders — they were read
# directly from the pinned template and are the diff target for the flatten.
# Pre-flatten template IDs are noted in comments for orientation.
# ===========================================================================

class NodeRef(NamedTuple):
    node_id: str      # flat API-format node ID (post-flatten)
    class_type: str   # expected class_type; guards against renumbering onto a wrong node
    input_key: str    # must ALREADY exist on the node — see _set_input


_COMMON = {
    #  semantic field          NodeRef(node_id,           class_type,                input_key)
    "image":                   NodeRef("TBD:load_image",   "LoadImage",              "image"),
    "prompt":                  NodeRef("TBD:prompt",       "PrimitiveStringMultiline", "value"),
    "negative_prompt":         NodeRef("TBD:negprompt",    "CLIPTextEncode",         "text"),      # template node 373
    "width":                   NodeRef("TBD:latent_video", "EmptyLTXVLatentVideo",   "width"),     # template node 356
    "height":                  NodeRef("TBD:latent_video", "EmptyLTXVLatentVideo",   "height"),
    "img_compression":         NodeRef("TBD:preprocess",   "LTXVPreprocess",         "img_compression"),  # node 350, default 18
    "i2v_strength_first":      NodeRef("TBD:i2v_first",    "LTXVImgToVideoInplace",  "strength"),  # node 349, default 1.0
    "i2v_strength_second":     NodeRef("TBD:i2v_second",   "LTXVImgToVideoInplace",  "strength"),  # node 357, default 0.7
    "seed":                    NodeRef("TBD:noise",        "RandomNoise",            "noise_seed"),
    # Dual CFG (Ruling 9): LTX-2.5 replaced CFGGuider with LTXVDualCFGGuider, which
    # declares video_cfg + audio_cfg instead of a single `cfg`. Writing "cfg" here would
    # create an undeclared key and the caller's cfg would be SILENTLY IGNORED.
    "cfg_stage1":              NodeRef("TBD:guider1",      "LTXVDualCFGGuider",      "video_cfg"), # node 388
    "audio_cfg_stage1":        NodeRef("TBD:guider1",      "LTXVDualCFGGuider",      "audio_cfg"),
    "cfg_stage2":              NodeRef("TBD:guider2",      "LTXVDualCFGGuider",      "video_cfg"), # node 391
    "audio_cfg_stage2":        NodeRef("TBD:guider2",      "LTXVDualCFGGuider",      "audio_cfg"),
    # Sigma nodes. stage1 == the 9-value/8-step list; stage2 == the 4-value/3-step list.
    "sigmas_stage1":           NodeRef("TBD:sigmas1",      "ManualSigmas",           "sigmas"),    # node 397 (8-step)
    "sigmas_stage2":           NodeRef("TBD:sigmas2",      "ManualSigmas",           "sigmas"),    # node 396 (3-step)
}

NODE_MAP: Dict[str, Dict[str, NodeRef]] = {
    "generated_audio": {
        **_COMMON,
        "frame_count":         NodeRef("TBD:latent_video", "EmptyLTXVLatentVideo",   "length"),
        "audio_frames":        NodeRef("TBD:latent_audio", "LTXVEmptyLatentAudio",   "frames_number"),  # node 366
        "fps":                 NodeRef("TBD:latent_audio", "LTXVEmptyLatentAudio",   "fps"),
    },
    "custom_audio": {
        **_COMMON,
        # custom_audio derives its length from the audio clip via the graph's math node,
        # exactly as LTX-2.3 does — the handler writes the DURATION, not the frame count.
        "audio_file":          NodeRef("TBD:load_audio",   "LoadAudio",              "audio"),
        "duration":            NodeRef("TBD:duration",     "PrimitiveFloat",         "value"),
        "trim_duration":       NodeRef("TBD:trim_audio",   "TrimAudioDuration",      "duration"),
        # SolidMask dimensions must track width/height so the zero mask covers the latent.
        "mask_width":          NodeRef("TBD:solidmask",    "SolidMask",              "width"),
        "mask_height":         NodeRef("TBD:solidmask",    "SolidMask",              "height"),
    },
}

# Sentinel prefix for un-flattened IDs. check_node_map.py hard-fails on any of these.
PLACEHOLDER_PREFIX = "TBD:"


class WorkflowNodeMissing(RuntimeError):
    """A mapped node ID, class_type, or input key does not match the workflow JSON.

    Deterministic deployment defect: the node map and the graph have drifted. Classified
    as non-retryable WORKFLOW_EXECUTION_ERROR — retrying cannot repair a baked-in map,
    and counting it toward the infra quarantine would disguise a deploy bug as flaky
    infrastructure.
    """


class InvalidInput(ValueError):
    """Caller-supplied value failed validation. Always non-retryable INVALID_INPUT."""


def _worker_metadata() -> Dict[str, Any]:
    worker_id = (
        os.getenv("RUNPOD_WORKER_ID")
        or os.getenv("RUNPOD_POD_ID")
        or os.getenv("RUNPOD_MACHINE_ID")
    )
    return {
        "id": worker_id,
        "hostname": socket.gethostname(),
        "boot_id": os.getenv("RUNPOD_BOOT_ID"),
    }


def _record_infra_failure() -> int:
    now = time.time()
    INFRA_FAILURE_TIMESTAMPS.append(now)
    cutoff = now - INFRA_FAILURE_WINDOW_SECONDS
    while INFRA_FAILURE_TIMESTAMPS and INFRA_FAILURE_TIMESTAMPS[0] < cutoff:
        INFRA_FAILURE_TIMESTAMPS.pop(0)
    return len(INFRA_FAILURE_TIMESTAMPS)


def _failure_response(
    *,
    error_code: str,
    error_message: str,
    retryable: bool,
    infra_error: bool,
    elapsed_s: float,
    refresh_worker: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "error": error_message,  # Backwards compatibility
        "error_code": error_code,
        "error_message": error_message,
        "retryable": retryable,
        "infra_error": infra_error,
        "worker": _worker_metadata(),
        "timings": {"elapsed_s": elapsed_s},
        "elapsed_time": elapsed_s,  # Backwards compatibility
    }
    if refresh_worker:
        payload["refresh_worker"] = True
    if extra:
        payload.update(extra)
    return payload


def _infra_failure_response(
    *,
    error_code: str,
    error_message: str,
    retryable: bool,
    elapsed_s: float,
    refresh_worker: bool = False,
) -> Dict[str, Any]:
    infra_count = _record_infra_failure()
    if infra_count >= INFRA_FAILURE_THRESHOLD:
        quarantine = _failure_response(
            error_code="WORKER_QUARANTINED",
            error_message=(
                f"Worker quarantined after {infra_count} infra failures in "
                f"{INFRA_FAILURE_WINDOW_SECONDS}s"
            ),
            retryable=True,
            infra_error=True,
            elapsed_s=elapsed_s,
            refresh_worker=True,
            extra={"infra_failure_count": infra_count},
        )
        logger.error(quarantine["error_message"])
        if HARD_EXIT_ON_QUARANTINE:
            logger.error("Exiting worker process due to quarantine threshold")
            os._exit(1)  # noqa: PLW1510
        return quarantine

    return _failure_response(
        error_code=error_code,
        error_message=error_message,
        retryable=retryable,
        infra_error=True,
        elapsed_s=elapsed_s,
        refresh_worker=refresh_worker,
        extra={"infra_failure_count": infra_count},
    )


def _classify_exception(exc: Exception) -> tuple[str, str, bool, bool, bool]:
    message = str(exc)
    lowered = message.lower()

    # D6 — a node-map defect is deterministic. Never retryable, never infra.
    if isinstance(exc, WorkflowNodeMissing):
        return ("WORKFLOW_EXECUTION_ERROR", message, False, False, False)

    # D4/D5 — caller-supplied garbage is the caller's problem, not the worker's.
    if isinstance(exc, InvalidInput):
        return ("INVALID_INPUT", message, False, False, False)

    if isinstance(exc, TimeoutError):
        return ("WORKFLOW_TIMEOUT", message, True, True, True)

    if isinstance(exc, urllib.error.URLError):
        return ("COMFYUI_UNREACHABLE", message, True, True, True)

    if "workflow error" in lowered:
        return ("WORKFLOW_EXECUTION_ERROR", message, False, False, False)

    if "http error" in lowered and ("prompt" in lowered or "/prompt" in lowered):
        return ("WORKFLOW_QUEUE_FAILED", message, True, True, True)

    if "incorrect padding" in lowered or "invalid base64" in lowered:
        return ("INVALID_INPUT", message, False, False, False)

    if "missing required field" in lowered:
        return ("INVALID_INPUT", message, False, False, False)

    return ("INTERNAL_ERROR", message, True, True, True)


def _success_response(*, video_data: str, seed: Optional[int], elapsed_s: float, extra: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": True,
        "video": video_data,
        "seed": seed,
        "worker": _worker_metadata(),
        "timings": {"elapsed_s": elapsed_s},
        "elapsed_time": elapsed_s,  # Backwards compatibility
    }
    payload.update(extra)
    return payload


def log_separator(char="-", length=80):
    logger.info(char * length)


def log_section(title: str):
    log_separator("=")
    logger.info(f"  {title}")
    log_separator("=")


# ---------------------------------------------------------------------------
# Input validation (D4)
# ---------------------------------------------------------------------------
def _require_int(name: str, value, *, minimum: int, maximum: Optional[int] = None,
                 multiple_of: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput(f"{name} must be a number, got {type(value).__name__}")
    if isinstance(value, float) and not value.is_integer():
        raise InvalidInput(f"{name} must be a whole number, got {value}")
    iv = int(value)
    if iv < minimum:
        raise InvalidInput(f"{name} must be >= {minimum}, got {iv}")
    if maximum is not None and iv > maximum:
        raise InvalidInput(f"{name} must be <= {maximum}, got {iv}")
    if multiple_of is not None and iv % multiple_of != 0:
        raise InvalidInput(f"{name} must be a multiple of {multiple_of}, got {iv}")
    return iv


def _require_finite(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput(f"{name} must be a number, got {type(value).__name__}")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise InvalidInput(f"{name} must be finite, got {value}")
    return fv


def wait_for_comfyui(timeout: int = 120) -> bool:
    logger.info(f"Waiting for ComfyUI server at {COMFYUI_URL}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5)
            if response.status == 200:
                logger.info("ComfyUI server is ready!")
                return True
        except Exception as e:
            logger.debug(f"Waiting... ({e})")
        time.sleep(2)
    logger.error(f"ComfyUI server did not start within {timeout} seconds")
    return False


def save_input_image(image_data: str, filename: str) -> str:
    """Write the input image. `filename` is JOB-SCOPED (D-6m) so two in-flight jobs
    on one worker cannot overwrite each other's inputs."""
    logger.info("Saving input image...")
    try:
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        image_bytes = base64.b64decode(image_data)
        filepath = f"/comfyui/input/{filename}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        file_size = os.path.getsize(filepath)
        logger.info(f"Saved input image to {filepath} ({file_size} bytes)")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save input image: {e}")
        raise


def _probe_duration(filepath: str) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ], capture_output=True, text=True, timeout=30)
    return float(result.stdout.strip())


def save_input_audio(audio_data: str, filename: str) -> Tuple[str, float]:
    """Write the driving audio and return (path, duration).

    D5 — a probe failure now raises InvalidInput. LTX-2.3 returned a hardcoded 4.0 s
    here, which meant a corrupt or non-audio payload produced a 4-second video against
    unknown audio and reported SUCCESS. Fail loudly instead.
    `filename` is job-scoped (D-6m).
    """
    logger.info("Saving input audio...")
    if "base64," in audio_data:
        audio_data = audio_data.split("base64,")[1]
    try:
        audio_bytes = base64.b64decode(audio_data)
    except Exception as e:
        raise InvalidInput(f"audio is not valid base64: {e}") from e
    if not audio_bytes:
        raise InvalidInput("audio decoded to zero bytes")

    temp_filepath = f"/comfyui/input/temp_{filename}"
    filepath = f"/comfyui/input/{filename}"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(temp_filepath, "wb") as f:
        f.write(audio_bytes)
    logger.info(f"Saved temp audio to {temp_filepath} ({os.path.getsize(temp_filepath)} bytes)")

    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels",
            "-of", "default=noprint_wrappers=1:nokey=1",
            temp_filepath
        ], capture_output=True, text=True, timeout=30)
        channels = int(result.stdout.strip())
        logger.info(f"Audio channels: {channels}")
    except Exception as e:
        raise InvalidInput(
            f"audio could not be probed (corrupt, empty, or not an audio stream): {e}"
        ) from e

    # LTX-2.5's audio VAE requires stereo, same as LTX-2. Keep the mono->stereo convert.
    if channels == 1:
        logger.info("Converting mono audio to stereo...")
        convert_result = subprocess.run([
            "ffmpeg", "-y", "-i", temp_filepath,
            "-ac", "2",
            "-ar", "44100",
            filepath
        ], capture_output=True, text=True, timeout=60)
        if convert_result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {convert_result.stderr}")
            os.rename(temp_filepath, filepath)
        else:
            logger.info("Converted to stereo successfully")
            os.remove(temp_filepath)
    else:
        os.rename(temp_filepath, filepath)

    try:
        duration = _probe_duration(filepath)
    except Exception as e:
        raise InvalidInput(f"could not determine audio duration: {e}") from e
    if duration <= 0:
        raise InvalidInput(f"audio duration must be > 0, got {duration}")
    logger.info(f"Audio duration: {duration:.2f} seconds")
    return filepath, duration


def load_workflow(workflow_name: str = "generated_audio") -> Dict:
    workflow_path = f"/workflow_{workflow_name}.json"
    if not os.path.exists(workflow_path):
        workflow_path = "/workflow.json"
    logger.info(f"Loading workflow from {workflow_path}...")
    with open(workflow_path, "r") as f:
        workflow = json.load(f)
    logger.info("Workflow loaded successfully")
    return workflow


def _set(workflow: Dict, ref: NodeRef, value) -> None:
    """Write one mapped input, failing loudly on any drift (Ruling 10).

    LTX-2.3's version wrote `if node_id in workflow` and silently no-opped otherwise.
    Against a fully renumbered graph that is the single highest-probability failure of
    this port: a stale ID yields a plausible-looking video at the wrong resolution,
    duration, or seed, with no error anywhere.

    Three guards, each closing a distinct drift mode:
      1. node exists          -> catches renumbering
      2. class_type matches   -> catches renumbering ONTO A VALID BUT WRONG NODE
      3. input key exists     -> catches schema change, e.g. writing `cfg` to
                                 LTXVDualCFGGuider which declares video_cfg/audio_cfg
    """
    if ref.node_id.startswith(PLACEHOLDER_PREFIX):
        raise WorkflowNodeMissing(
            f"node map still contains the placeholder {ref.node_id!r} — the workflow "
            "template has not been flattened yet. See NODE_MAP."
        )
    node = workflow.get(ref.node_id)
    if node is None:
        raise WorkflowNodeMissing(f"node {ref.node_id} not in workflow (stale node map?)")
    actual_class = node.get("class_type")
    if actual_class != ref.class_type:
        raise WorkflowNodeMissing(
            f"node {ref.node_id} is {actual_class!r}, expected {ref.class_type!r}"
        )
    inputs = node.setdefault("inputs", {})
    if ref.input_key not in inputs:
        raise WorkflowNodeMissing(
            f"node {ref.node_id} ({actual_class}) has no input {ref.input_key!r} "
            f"(declared inputs: {sorted(inputs)})"
        )
    inputs[ref.input_key] = value


def _apply_steps(workflow: Dict, params: Dict, m: Dict[str, NodeRef]) -> None:
    """Override the calibrated sigma schedules from a caller-supplied `steps`.

    Ruling 5: stage1 carries the 8-step (_REFINE) list and stage2 the 3-step (_FIRST)
    list. The pairing below is IDENTICAL to LTX-2.3's and must not be inverted.
    """
    steps = params.get("steps")
    if steps is None:
        return
    first, refine = _split_steps(int(steps))
    logger.warning(
        f"Overriding ManualSigmas via steps={steps} "
        f"(first={first}, refine={refine}; JSON default was {JSON_STEPS_FIRST}+{JSON_STEPS_REFINE}={JSON_STEPS_TOTAL}). "
        "Output may drift from the calibrated schedule."
    )
    _set(workflow, m["sigmas_stage2"], _resample_sigmas(JSON_SIGMAS_FIRST, first))
    _set(workflow, m["sigmas_stage1"], _resample_sigmas(JSON_SIGMAS_REFINE, refine))


def _apply_common(workflow: Dict, params: Dict, m: Dict[str, NodeRef], image_filename: str) -> None:
    _set(workflow, m["image"], image_filename)
    _set(workflow, m["prompt"], params["prompt"])
    _set(workflow, m["negative_prompt"], params["negative_prompt"])
    _set(workflow, m["width"], params["width"])
    _set(workflow, m["height"], params["height"])
    _set(workflow, m["img_compression"], params["img_compression"])
    _set(workflow, m["i2v_strength_first"], params["i2v_strength_first"])
    _set(workflow, m["i2v_strength_second"], params["i2v_strength_second"])
    _set(workflow, m["seed"], params["seed"])
    # Dual CFG on BOTH stages. audio_cfg defaults to cfg, which reproduces LTX-2.3's
    # single-CFG behaviour exactly (Guider_LTXAVDualCFG delegates to the single-CFG path
    # when the two values are close).
    for stage in ("stage1", "stage2"):
        _set(workflow, m[f"cfg_{stage}"], params["cfg"])
        _set(workflow, m[f"audio_cfg_{stage}"], params["audio_cfg"])
    _apply_steps(workflow, params, m)


def modify_workflow_generated_audio(workflow: Dict, params: Dict, image_filename: str) -> Dict:
    """Configure the generated-audio graph. Wiring lives in NODE_MAP['generated_audio'].

    NOTE (D1): there is deliberately NO prompt-expander write here. LTX-2.3 routed the
    prompt through TextGenerateLTX2Prompt (driven by an abliterated Gemma LoRA) before
    encoding. LTX-2.5 encodes the prompt verbatim — the enhancer nodes are removed from
    the graph entirely. See the module docstring and README.
    """
    m = NODE_MAP["generated_audio"]
    _apply_common(workflow, params, m, image_filename)
    _set(workflow, m["frame_count"], params["frame_count"])
    _set(workflow, m["audio_frames"], params["frame_count"])
    _set(workflow, m["fps"], params["fps"])
    return workflow


def modify_workflow_custom_audio(workflow: Dict, params: Dict, audio_duration: float,
                                 image_filename: str, audio_filename: str) -> Dict:
    """Configure the custom-audio (lip-sync) graph. Wiring in NODE_MAP['custom_audio'].

    As in LTX-2.3, the handler writes the DURATION and the graph's math node derives the
    frame count — but with the ceil-to-grid form, not `a*b+1` (see _snap_frames).
    """
    m = NODE_MAP["custom_audio"]
    _apply_common(workflow, params, m, image_filename)
    _set(workflow, m["audio_file"], audio_filename)
    _set(workflow, m["duration"], audio_duration)
    _set(workflow, m["trim_duration"], audio_duration)
    # The zero mask must cover the full latent, so it tracks the requested resolution.
    _set(workflow, m["mask_width"], params["width"])
    _set(workflow, m["mask_height"], params["height"])
    return workflow


def queue_prompt(workflow: Dict) -> str:
    logger.info("Queueing prompt to ComfyUI...")
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:2000]
        except Exception:
            pass
        logger.error(f"Failed to queue prompt: HTTP error {e.code} on /prompt: {body}")
        raise RuntimeError(f"HTTP error {e.code} queueing /prompt: {body}") from e
    result = json.loads(response.read().decode("utf-8"))
    prompt_id = result.get("prompt_id")
    logger.info(f"Prompt queued with ID: {prompt_id}")
    return prompt_id


def wait_for_completion(prompt_id: str, timeout: int = 600) -> Dict:
    logger.info(f"Waiting for completion (timeout: {timeout}s)...")
    start_time = time.time()
    last_progress = 0
    while time.time() - start_time < timeout:
        try:
            response = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            history = json.loads(response.read().decode("utf-8"))
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    error_msg = status.get("messages", [["Unknown error"]])[0]
                    logger.error(f"Workflow execution error: {error_msg}")
                    raise RuntimeError(f"Workflow error: {error_msg}")
                if outputs:
                    elapsed = time.time() - start_time
                    logger.info(f"Generation completed in {elapsed:.1f}s")
                    return outputs
            current_time = int(time.time() - start_time)
            if current_time % 10 == 0 and current_time != last_progress:
                logger.info(f"Progress: {current_time}s elapsed...")
                last_progress = current_time
        except urllib.error.URLError as e:
            logger.debug(f"Status check error: {e}")
        except RuntimeError:
            raise
        except Exception as e:
            logger.debug(f"Status check error: {e}")
        time.sleep(1)
    logger.error(f"Generation timed out after {timeout}s")
    raise TimeoutError(f"Generation timed out after {timeout} seconds")


def get_output_video(outputs: Dict) -> Optional[str]:
    """Extract the video produced by THIS job.

    D-6l: LTX-2.3 fell back to walking all of /comfyui/output when the history response
    yielded nothing recognized, and returned the newest video file found. On a worker with
    a previous job's output still on disk that returns THE WRONG VIDEO and reports ok:true
    — NO_OUTPUT_VIDEO silently became a false success. The fallback is removed; only
    filenames named in this prompt's own history response are eligible.
    """
    logger.info("Extracting output video...")
    for node_id, node_output in outputs.items():
        for key in ["gifs", "videos", "video", "images", "files"]:
            if key in node_output:
                items = node_output[key]
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    if isinstance(item, dict):
                        filename = item.get("filename")
                        subfolder = item.get("subfolder", "")
                    elif isinstance(item, str):
                        filename = item
                        subfolder = ""
                    else:
                        continue
                    if not filename:
                        continue
                    filepath = (f"/comfyui/output/{subfolder}/{filename}" if subfolder
                                else f"/comfyui/output/{filename}")
                    if os.path.exists(filepath):
                        file_size = os.path.getsize(filepath)
                        logger.info(f"Found output video: {filepath} ({file_size} bytes)")
                        with open(filepath, "rb") as f:
                            return base64.b64encode(f.read()).decode("utf-8")
                    else:
                        logger.warning(f"History named {filepath} but it is not on disk")
    logger.warning("No output video found in this job's history outputs")
    return None


def handler(job: Dict) -> Dict:
    job_id = job.get("id", "unknown")
    log_section(f"LTX-2.5 VIDEO GENERATION JOB: {job_id}")
    start_time = time.time()
    try:
        job_input = job.get("input", {})
        if "image" not in job_input:
            return _failure_response(
                error_code="INVALID_INPUT",
                error_message="Missing required field: image",
                retryable=False, infra_error=False,
                elapsed_s=round(time.time() - start_time, 3),
            )
        if "prompt" not in job_input:
            return _failure_response(
                error_code="INVALID_INPUT",
                error_message="Missing required field: prompt",
                retryable=False, infra_error=False,
                elapsed_s=round(time.time() - start_time, 3),
            )

        # Mode selection is TRUTHINESS, not key presence — carried unchanged from
        # LTX-2.3 so callers relying on `audio: ""` meaning "no audio" keep working.
        # An empty-but-present `audio` intended as lip-sync now fails validation below
        # rather than silently producing a generated-audio video.
        has_custom_audio = "audio" in job_input and job_input["audio"]
        mode = "custom_audio" if has_custom_audio else "generated_audio"
        logger.info(f"Mode: {mode.upper()}")

        defaults = DEFAULT_PARAMS_CUSTOM_AUDIO if has_custom_audio else DEFAULT_PARAMS_GENERATED

        # --- validation (D4) -------------------------------------------------
        width = _require_int("width", job_input.get("width", defaults["width"]),
                             minimum=32, maximum=4096, multiple_of=32)
        height = _require_int("height", job_input.get("height", defaults["height"]),
                              minimum=32, maximum=4096, multiple_of=32)
        fps = _require_int("fps", job_input.get("fps", defaults["fps"]), minimum=1, maximum=120)
        cfg = _require_finite("cfg", job_input.get("cfg", defaults["cfg"]))
        raw_audio_cfg = job_input.get("audio_cfg")
        # .get(key, default) does NOT fire on an explicit JSON null — handle it here or a
        # `null` propagates into the graph and fails as an opaque infra error.
        audio_cfg = cfg if raw_audio_cfg is None else _require_finite("audio_cfg", raw_audio_cfg)
        raw_steps = job_input.get("steps", defaults["steps"])
        steps = None if raw_steps is None else _require_int("steps", raw_steps, minimum=1, maximum=200)

        params = {
            "image": job_input["image"],
            "prompt": job_input["prompt"],
            "negative_prompt": job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            "width": width,
            "height": height,
            "fps": fps,
            "cfg": cfg,
            "audio_cfg": audio_cfg,
            "seed": job_input.get("seed", defaults["seed"]),
            "timeout": job_input.get("timeout", defaults["timeout"]),
            "img_compression": job_input.get("img_compression", defaults["img_compression"]),
            "i2v_strength_first": job_input.get("i2v_strength_first", defaults["i2v_strength_first"]),
            "i2v_strength_second": job_input.get("i2v_strength_second", defaults["i2v_strength_second"]),
            "steps": steps,
        }
        if not has_custom_audio:
            requested = _require_int(
                "frame_count",
                job_input.get("frame_count", DEFAULT_PARAMS_GENERATED["frame_count"]),
                minimum=9,
            )
            params["frame_count"] = _snap_frames(requested)   # D2

        logger.info("Input parameters:")
        logger.info(f"  Prompt: {params['prompt'][:100]}...")
        logger.info(f"  Resolution: {params['width']}x{params['height']}")
        logger.info(f"  Steps: {params['steps'] or 'JSON-default (3+8=11)'}, "
                    f"CFG: {params['cfg']} (audio_cfg={params['audio_cfg']}), FPS: {params['fps']}")
        logger.info(f"  Seed: {params['seed'] or 'random'}")
        logger.info(f"  I2V Strength: first={params['i2v_strength_first']}, second={params['i2v_strength_second']}")

        if not wait_for_comfyui():
            return _infra_failure_response(
                error_code="COMFYUI_BOOT_TIMEOUT",
                error_message="ComfyUI server not available",
                retryable=True,
                elapsed_s=round(time.time() - start_time, 3),
                refresh_worker=True,
            )

        # Job-scoped input filenames (D-6m) — two in-flight jobs on one worker cannot
        # clobber each other. Written into the graph rather than assumed by it.
        safe_id = "".join(c for c in str(job_id) if c.isalnum() or c in "-_")[:64] or "job"
        image_filename = f"{safe_id}_image.png"
        audio_filename = f"{safe_id}_audio.mp3"

        save_input_image(params["image"], image_filename)

        audio_duration = None
        if has_custom_audio:
            _, audio_duration = save_input_audio(job_input["audio"], audio_filename)
            workflow = load_workflow("custom_audio")
            workflow = modify_workflow_custom_audio(
                workflow, params, audio_duration, image_filename, audio_filename)
        else:
            workflow = load_workflow("generated_audio")
            workflow = modify_workflow_generated_audio(workflow, params, image_filename)

        prompt_id = queue_prompt(workflow)
        outputs = wait_for_completion(prompt_id, params["timeout"])
        video_data = get_output_video(outputs)

        if not video_data:
            return _failure_response(
                error_code="NO_OUTPUT_VIDEO",
                error_message="No video output generated",
                retryable=False, infra_error=False,
                elapsed_s=round(time.time() - start_time, 3),
            )

        elapsed = time.time() - start_time
        log_section("JOB COMPLETED SUCCESSFULLY")
        logger.info(f"Total time: {elapsed:.1f}s")

        result = _success_response(
            video_data=video_data,
            seed=params["seed"],
            elapsed_s=round(elapsed, 3),
            extra={
                "mode": mode,
                "parameters": {
                    "prompt": params["prompt"],
                    "width": params["width"],
                    "height": params["height"],
                    "steps": params["steps"],
                    "cfg": params["cfg"],
                    "audio_cfg": params["audio_cfg"],
                    "fps": params["fps"],
                    "img_compression": params["img_compression"],
                    "i2v_strength_first": params["i2v_strength_first"],
                    "i2v_strength_second": params["i2v_strength_second"],
                },
            },
        )
        # `parameters.frame_count` always echoes what ACTUALLY RAN, post-snap — callers
        # must treat it as authoritative rather than assuming their requested value held.
        if audio_duration:
            result["audio_duration"] = audio_duration
            result["parameters"]["frame_count"] = _frames_for_duration(audio_duration, params["fps"])
        else:
            result["parameters"]["frame_count"] = params["frame_count"]
        return result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Job failed after {elapsed:.1f}s: {str(e)}")
        trace = traceback.format_exc()
        logger.error(trace)

        error_code, error_message, retryable, infra_error, refresh_worker = _classify_exception(e)
        elapsed_s = round(elapsed, 3)

        if infra_error or error_code in INFRA_ERROR_CODES:
            return _infra_failure_response(
                error_code=error_code,
                error_message=error_message,
                retryable=retryable,
                elapsed_s=elapsed_s,
                refresh_worker=refresh_worker,
            )

        return _failure_response(
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
            infra_error=infra_error,
            elapsed_s=elapsed_s,
            extra={"traceback": trace},
        )


if __name__ == "__main__":
    log_section("LTX-2.5 VIDEO SERVERLESS HANDLER STARTING")
    logger.info(f"ComfyUI URL: {COMFYUI_URL}")
    logger.info("Supported modes: generated_audio, custom_audio")
    runpod.serverless.start({"handler": handler})
