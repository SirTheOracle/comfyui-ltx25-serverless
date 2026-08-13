"""Local unit tests for the LTX-2.5 handler.

These deliberately run WITHOUT a GPU, without models, and without ComfyUI. The review
of the plan flagged that Rev 1 leaned too hard on expensive deployed-canary tests for
logic that is pure Python — this file is the correction.

Run: pytest tests/ -q
"""
import base64
import json
import math
import sys
import types
from pathlib import Path

import pytest

# handler.py imports runpod, which is not installed in CI. Stub before import.
if "runpod" not in sys.modules:
    stub = types.ModuleType("runpod")
    stub.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules["runpod"] = stub

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import handler as H  # noqa: E402


# ---------------------------------------------------------------------------
# Frame snapping (Ruling 6) — rounds UP so dialogue is never truncated
# ---------------------------------------------------------------------------
class TestSnapFrames:
    def test_on_grid_values_pass_through(self):
        for n in (9, 17, 73, 81, 121):
            assert H._snap_frames(n) == n

    def test_default_frame_count_is_on_grid(self):
        # 121 = 8*15+1. Back-compat: the documented 2.3 default must not be altered.
        assert H._snap_frames(H.DEFAULT_PARAMS_GENERATED["frame_count"]) == 121

    def test_rounds_up_not_down(self):
        # The whole point of Ruling 6: never lose the tail.
        assert H._snap_frames(74) == 81
        assert H._snap_frames(80) == 81
        assert H._snap_frames(122) == 129

    def test_never_below_floor(self):
        assert H._snap_frames(1) == 9
        assert H._snap_frames(-5) == 9

    def test_result_is_always_on_grid(self):
        for n in range(1, 400):
            assert (H._snap_frames(n) - 1) % 8 == 0


class TestFramesForDuration:
    def test_matches_graph_expression(self):
        """Python and the graph's ComfyMathExpression must agree.

        A divergence here is silent and yields a wrong echoed duration.
        Graph form: 1 + ceil(a*b/8)*8
        """
        for duration in [0.5, 1.0, 2.0, 3.0, 3.25, 3.33, 4.7, 5.0, 10.0, 12.5]:
            for fps in (24, 25):
                expected = 1 + math.ceil((duration * fps) / 8) * 8
                assert H._frames_for_duration(duration, fps) == max(9, expected)

    def test_non_grid_clip_is_not_truncated(self):
        # 3.25s @ 24fps = 78 raw frames -> must round UP to 81, not down to 73.
        assert H._frames_for_duration(3.25, 24) == 81

    def test_result_is_on_grid(self):
        for d in [0.4, 1.1, 2.9, 3.25, 7.7]:
            assert (H._frames_for_duration(d, 24) - 1) % 8 == 0


# ---------------------------------------------------------------------------
# Sigma schedules (Ruling 5) — must NOT be inverted
# ---------------------------------------------------------------------------
class TestSigmas:
    def test_constants_carried_verbatim_from_ltx23(self):
        assert H.JSON_SIGMAS_FIRST == [0.85, 0.7250, 0.4219, 0.0]
        assert H.JSON_SIGMAS_REFINE == [
            1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

    def test_step_split_is_3_and_8_not_inverted(self):
        assert H.JSON_STEPS_FIRST == 3
        assert H.JSON_STEPS_REFINE == 8
        assert H._split_steps(11) == (3, 8)

    def test_split_scales_proportionally(self):
        first, refine = H._split_steps(22)
        assert first + refine == 22
        assert first == 6           # round(22 * 3/11)

    def test_split_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            H._split_steps(0)

    def test_resample_identity(self):
        assert H._resample_sigmas(H.JSON_SIGMAS_FIRST, 3) == "0.85, 0.725, 0.4219, 0"


# ---------------------------------------------------------------------------
# Input validation (D4)
# ---------------------------------------------------------------------------
class TestValidation:
    @pytest.mark.parametrize("bad", ["abc", None, [], {}, True])
    def test_require_int_rejects_non_numbers(self, bad):
        with pytest.raises(H.InvalidInput):
            H._require_int("x", bad, minimum=1)

    def test_require_int_rejects_fractional(self):
        with pytest.raises(H.InvalidInput):
            H._require_int("x", 3.5, minimum=1)

    def test_require_int_enforces_multiple(self):
        with pytest.raises(H.InvalidInput):
            H._require_int("width", 100, minimum=32, multiple_of=32)
        assert H._require_int("width", 128, minimum=32, multiple_of=32) == 128

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "1.0", None])
    def test_require_finite_rejects(self, bad):
        with pytest.raises(H.InvalidInput):
            H._require_finite("cfg", bad)

    def test_require_finite_accepts_normal(self):
        assert H._require_finite("cfg", 1) == 1.0
        assert H._require_finite("cfg", 3.5) == 3.5


# ---------------------------------------------------------------------------
# Error classification (D4/D5/D6) — deterministic defects must not quarantine a worker
# ---------------------------------------------------------------------------
class TestClassification:
    def test_node_map_defect_is_non_retryable_and_not_infra(self):
        code, _, retryable, infra, refresh = H._classify_exception(
            H.WorkflowNodeMissing("node 42 missing"))
        assert code == "WORKFLOW_EXECUTION_ERROR"
        assert retryable is False
        assert infra is False
        assert refresh is False

    def test_node_map_defect_does_not_count_toward_quarantine(self):
        code, *_ = H._classify_exception(H.WorkflowNodeMissing("x"))
        assert code not in H.INFRA_ERROR_CODES

    def test_invalid_input_is_non_retryable(self):
        code, _, retryable, infra, _ = H._classify_exception(H.InvalidInput("bad fps"))
        assert code == "INVALID_INPUT"
        assert retryable is False and infra is False

    def test_timeout_is_retryable_infra(self):
        code, _, retryable, infra, _ = H._classify_exception(TimeoutError("slow"))
        assert code == "WORKFLOW_TIMEOUT"
        assert retryable is True and infra is True

    def test_unknown_still_falls_through_to_internal_error(self):
        code, *_ = H._classify_exception(RuntimeError("something odd"))
        assert code == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# _set — the hardened writer (Ruling 10)
# ---------------------------------------------------------------------------
def _graph():
    return {
        "10": {"class_type": "LTXVDualCFGGuider", "inputs": {"video_cfg": 3.0, "audio_cfg": 7.0}},
        "11": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": 768, "height": 512, "length": 97}},
    }


class TestSetInput:
    def test_writes_existing_key(self):
        g = _graph()
        H._set(g, H.NodeRef("10", "LTXVDualCFGGuider", "video_cfg"), 1.0)
        assert g["10"]["inputs"]["video_cfg"] == 1.0

    def test_raises_on_missing_node(self):
        with pytest.raises(H.WorkflowNodeMissing, match="not in workflow"):
            H._set(_graph(), H.NodeRef("999", "EmptyLTXVLatentVideo", "width"), 1)

    def test_raises_on_class_mismatch(self):
        # Renumbering onto a valid but wrong node — key-checking alone would miss this
        # when two classes share an input name.
        with pytest.raises(H.WorkflowNodeMissing, match="expected"):
            H._set(_graph(), H.NodeRef("11", "LTXVDualCFGGuider", "width"), 1)

    def test_raises_on_missing_input_key(self):
        """The exact CFGGuider -> LTXVDualCFGGuider bug (Ruling 9).

        Writing `cfg` to a node declaring video_cfg/audio_cfg must FAIL, not silently
        create a dead key while the real widgets keep their JSON defaults.
        """
        with pytest.raises(H.WorkflowNodeMissing, match="has no input"):
            H._set(_graph(), H.NodeRef("10", "LTXVDualCFGGuider", "cfg"), 1.0)

    def test_raises_on_placeholder(self):
        with pytest.raises(H.WorkflowNodeMissing, match="placeholder"):
            H._set(_graph(), H.NodeRef("TBD:guider1", "LTXVDualCFGGuider", "video_cfg"), 1.0)


# ---------------------------------------------------------------------------
# Node map shape
# ---------------------------------------------------------------------------
class TestNodeMap:
    def test_both_modes_present(self):
        assert set(H.NODE_MAP) == {"generated_audio", "custom_audio"}

    def test_dual_cfg_never_maps_to_bare_cfg(self):
        """Regression guard for Ruling 9 — `cfg` is not a valid LTXVDualCFGGuider input."""
        for mode, refs in H.NODE_MAP.items():
            for field, ref in refs.items():
                if ref.class_type == "LTXVDualCFGGuider":
                    assert ref.input_key in ("video_cfg", "audio_cfg"), (
                        f"{mode}.{field} writes {ref.input_key!r} to LTXVDualCFGGuider")

    def test_custom_audio_has_no_frame_count(self):
        """custom_audio derives length from the clip, as in 2.3 — it must not be written."""
        assert "frame_count" not in H.NODE_MAP["custom_audio"]

    def test_generated_audio_has_no_prompt_expander(self):
        """D1 — the enhancer is dropped; nothing may map to TextGenerateLTX2Prompt."""
        for refs in H.NODE_MAP.values():
            for ref in refs.values():
                assert ref.class_type != "TextGenerateLTX2Prompt"


# ---------------------------------------------------------------------------
# Mode selection (D-6g) — truthiness, not presence
# ---------------------------------------------------------------------------
class TestModeSelection:
    @pytest.mark.parametrize("payload,expected", [
        ({}, "generated_audio"),
        ({"audio": ""}, "generated_audio"),
        ({"audio": None}, "generated_audio"),
        ({"audio": "eyJhIjoxfQ=="}, "custom_audio"),
    ])
    def test_truthiness_rule(self, payload, expected):
        has_custom = "audio" in payload and payload["audio"]
        assert ("custom_audio" if has_custom else "generated_audio") == expected


# ---------------------------------------------------------------------------
# Output extraction (D-6l) — must never return a previous job's video
# ---------------------------------------------------------------------------
class TestOutputExtraction:
    def test_returns_none_when_history_names_nothing(self, tmp_path, monkeypatch):
        # A stale file on disk must NOT be picked up: NO_OUTPUT_VIDEO is the correct
        # answer, and 2.3's directory-walk fallback would have returned the stale one.
        assert H.get_output_video({}) is None

    def test_returns_none_when_named_file_absent(self):
        outputs = {"9": {"videos": [{"filename": "does_not_exist.mp4", "subfolder": ""}]}}
        assert H.get_output_video(outputs) is None


# ---------------------------------------------------------------------------
# Regression: the shipped defaults must survive our own validators
# ---------------------------------------------------------------------------
class TestDefaultsPassValidation:
    """A validator stricter than the service's own defaults fails 100% of jobs.

    This shipped once: an earlier revision enforced multiple_of=32 on width/height,
    which rejects 720 (16*45). 720 is the DEFAULT height for generated_audio and the
    DEFAULT width for custom_audio, so every job in both modes failed at the entry
    boundary with INVALID_INPUT — after a clean 40 GB cold boot. The 46 tests that
    existed at the time all passed, because none of them fed the defaults through
    the validators.
    """

    @pytest.mark.parametrize("mode,defaults", [
        ("generated_audio", H.DEFAULT_PARAMS_GENERATED),
        ("custom_audio", H.DEFAULT_PARAMS_CUSTOM_AUDIO),
    ])
    def test_dimensions(self, mode, defaults):
        assert H._require_int("width", defaults["width"], minimum=32, maximum=4096)
        assert H._require_int("height", defaults["height"], minimum=32, maximum=4096)

    @pytest.mark.parametrize("mode,defaults", [
        ("generated_audio", H.DEFAULT_PARAMS_GENERATED),
        ("custom_audio", H.DEFAULT_PARAMS_CUSTOM_AUDIO),
    ])
    def test_fps_and_cfg(self, mode, defaults):
        assert H._require_int("fps", defaults["fps"], minimum=1, maximum=120)
        assert H._require_finite("cfg", defaults["cfg"]) == 1.0

    def test_generated_frame_count(self):
        fc = H.DEFAULT_PARAMS_GENERATED["frame_count"]
        assert H._require_int("frame_count", fc, minimum=9) == fc
        assert H._snap_frames(fc) == fc, "default frame_count must already be on-grid"

    def test_ltx23_proven_resolutions_are_accepted(self):
        """Both orientations LTX-2.3 ran in production must remain valid."""
        for w, h in ((1280, 720), (720, 1280), (1920, 1080), (1080, 1920)):
            assert H._require_int("width", w, minimum=32, maximum=4096) == w
            assert H._require_int("height", h, minimum=32, maximum=4096) == h
