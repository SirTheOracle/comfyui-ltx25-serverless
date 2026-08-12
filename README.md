# comfyui-ltx25-serverless

RunPod serverless worker for **LTX-2.5** image-to-video generation. Forked from
[`comfyui-ltx23-serverless`](https://github.com/SirTheOracle/comfyui-ltx23-serverless).

Plan of record: `headless_factory/.dev/proposals/ltx25-service-build/final-plan.md` (Rev 2).

> ## ⚠️ NOT SHIPPABLE YET
> Two build-blocking placeholders remain, both by design:
> 1. **`workflow_generated_audio.json` / `workflow_custom_audio.json` do not exist.** They
>    must be produced by flattening the Comfy-Org template (below). `handler.NODE_MAP` is
>    populated with `TBD:*` node IDs until then, and `tools/check_node_map.py` hard-fails
>    the build while any remain.
> 2. **`start.sh`'s `MODELS` table has `REPLACE_SIZE` / `REPLACE_SHA256` placeholders.**
>    Real values must be measured from the uploaded R2 objects.
>
> Everything else — Dockerfile, handler, checkers, CI, tests — is complete and tested.

---

## Two modes

Mode is selected by the **truthiness** of `audio` in the job input — *not* mere key
presence. `audio: ""` and `audio: null` both select `generated_audio`. This rule is
carried unchanged from LTX-2.3 so existing callers keep working.

| Mode | Trigger | Default resolution | Frame count |
|---|---|---|---|
| `generated_audio` | no/empty `audio` | 1280×720 @ 25fps | from `frame_count` (default 121) |
| `custom_audio` | non-empty `audio` | 720×1280 @ 24fps | derived from the clip's duration |

## Request

```jsonc
{
  "input": {
    "image": "<base64 png>",          // required
    "prompt": "...",                  // required
    "audio": "<base64 mp3>",          // optional; presence+truthiness selects custom_audio
    "negative_prompt": "...",         // default: "pc game, console game, video game, cartoon, childish, ugly"
    "width": 1280, "height": 720,     // must be multiples of 32
    "frame_count": 121,               // generated_audio only; snapped UP to the 8k+1 grid
    "fps": 25,
    "cfg": 1.0,
    "audio_cfg": 1.0,                 // NEW in 2.5; defaults to cfg
    "seed": null,                     // null -> random
    "img_compression": 18,
    "i2v_strength_first": 1.0,
    "i2v_strength_second": 0.7,
    "steps": null                     // null -> calibrated 3+8=11 schedule
  }
}
```

## Response

Success: `{ok, video, seed, mode, audio_duration?, parameters, timings, worker}`
Failure: `{ok:false, error, error_code, retryable, infra_error, refresh_worker?}`

`error_code` ∈ `INVALID_INPUT | COMFYUI_BOOT_TIMEOUT | COMFYUI_UNREACHABLE |
WORKFLOW_QUEUE_FAILED | WORKFLOW_TIMEOUT | WORKFLOW_EXECUTION_ERROR | NO_OUTPUT_VIDEO |
WORKER_QUARANTINED | INTERNAL_ERROR` — unchanged from 2.3.

**`parameters.frame_count` is authoritative.** It echoes what actually ran after grid
snapping, which may differ from what you requested. Do not assume your requested value held.

---

## Compatibility with LTX-2.3: backward-compatible, NOT identical

Every 2.3-shaped request still works. These are the deliberate deltas:

| # | Delta | Notes |
|---|---|---|
| **D1** | `generated_audio` prompts are **no longer VLM-expanded** | Same prompt → different output vs 2.3. See below. |
| **D2** | Off-grid `frame_count` snapped **up** to the 8k+1 grid | Logged; echoed as the snapped value |
| **D3** | New optional `audio_cfg` (defaults to `cfg`) | Absent → identical to 2.3 |
| **D4** | Malformed numeric inputs → non-retryable `INVALID_INPUT` | Was retryable `INTERNAL_ERROR`, which could quarantine a healthy worker |
| **D5** | Corrupt/undecodable audio → `INVALID_INPUT` | 2.3 silently assumed 4.0 s and reported success |
| **D6** | Node-map defects → non-retryable `WORKFLOW_EXECUTION_ERROR` | Retrying a baked-in defect cannot help |

D4–D6 only change classification for requests that were already failing. **D1 is the only
delta that alters output for a valid request.**

### D1 — dropped prompt auto-expansion

LTX-2.3 routed `generated_audio` prompts through `TextGenerateLTX2Prompt`, driven by an
**abliterated** Gemma-3 CLIP LoRA, before encoding. LTX-2.5 has no abliterated equivalent.
Its enhancer is a separate ungated 10.28 GB model
(`Comfy-Org/gemma-4` → `text_encoders/gemma4_e2b_it_bf16.safetensors`).

Dropped deliberately, in order of weight:
1. **+10.28 GB** on both cold-boot pull and VRAM budget, against the plan's tightest constraint.
2. Stock Gemma-4's refusal/sanitization behaviour is unverified for this content; the
   abliterated LoRA existed in 2.3 specifically to avoid that class of surprise.
3. Upstream prompts are already detailed, so the enhancer solves a problem this pipeline
   does not have.
4. It removes the `GemmaAPITextEncode` cloud-API branch and its LTX API-key dependency.

**Re-enabling is cheap:** one `MODELS` row + one `CLIPLoader` + one `TextGenerateLTX2Prompt`.
No architectural commitment has been foreclosed.

---

## Building the workflow JSONs (the remaining work)

**Base graph:** Comfy-Org `workflow_templates` **v0.11.39** →
`templates/video_ltx2_5_i2v.json`. That exact version is pinned into ComfyUI v0.32.0 via
`comfyui-workflow-templates==0.11.39`, so it also ships *inside* this image — flattening
can be done in-container for reproducibility.

Chosen over Lightricks' `example_workflows/2.5/` because it is the i2v mode specifically,
defaults to the int8-convrot files, wires both VAEs correctly, and references the 2.5
spatial upscaler. Lightricks' 2.5 examples carry stale widget defaults (31 residual
`ltx-2.3-22b-dev.safetensors` references; both `VAELoader` slots pointing at the *audio* VAE).

### 1. `workflow_generated_audio.json`
Flatten the `Image to Video (LTX-2.5)` subgraph to flat API format, then:
- Set all five loader values explicitly; do not trust widget defaults.
- **Delete** the enhancer chain: `CLIPLoader[gemma4_e2b_it_bf16]`, `TextGenerateLTX2Prompt`,
  `ComfySwitchNode`. Wire the positive `CLIPTextEncode` straight to the prompt primitive.
- Replace the frame-count node's `a * b + 1` with **`1 + ceil(a*b/8)*8`**. If
  `ComfyMathExpression` lacks `ceil`, use the equivalent `1 + floor((a*b + 7)/8)*8`.
- Keep `LTXVDualCFGGuider` on both stages, and `euler_ancestral` on both samplers.
- Verify sigma binding: stage 1 = the 9-value list, stage 2 = the 4-value list.

### 2. `workflow_custom_audio.json`
Start from the above and swap the audio branch. Replace `LTXVEmptyLatentAudio` with:

```
LoadAudio → TrimAudioDuration → LTXVAudioVAEEncode(audio_vae ← VAELoader[audio])
          → SetLatentNoiseMask(mask ← SolidMask(value=0))
          → LTXVConcatAVLatent(video_latent ← LTXVImgToVideoInplace)
```

All classes verified present in ComfyUI v0.32.0. This mirrors LTX-2.3's proven frozen-audio
mechanism; no published 2.5 lip-sync reference exists, so this is the one genuinely new
construction and **must** be validated by spike 0d.

### 3. Update `handler.NODE_MAP`
Replace every `TBD:*` with the post-flatten ID. `class_type` and `input_key` are already
correct — they were read from the pinned template. Then:

```bash
python tools/check_node_map.py     # must exit 0
pytest tests/ -q
```

### Lip-sync fallback ladder
If spike 0d shows weak sync, apply **in this order**:
1. **`LTXVModalityGuidance`** (`modality_scale=3.0`) ahead of `LTXVDualCFGGuider`. This is
   the real lever — it runs an extra forward pass with a/v cross-attention severed. **Measure
   its VRAM and time cost**; it is one extra pass *per step*.
2. Raise `audio_cfg` above `video_cfg` (free — widgets already exist).
3. `LTXVSetAudioRefTokens`, routing its `positive`/`negative` conditioning outputs.

⚠️ Do **not** reach for `LTXVSetAudioRefTokens.frozen_audio` alone. It emits
`noise_mask = torch.zeros(...)` — the same zero mask the primary path already applies —
so it changes essentially nothing.

---

## Model set (~40 GB cold boot, vs ~48 GB for 2.3)

| File | Dir | ≈ size |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `diffusion_models` | 21.50 GB |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `text_encoders` | 15.37 GB |
| `ltx-2.5-video-vae-bf16.safetensors` (DiffVAE) | `vae` | 1.47 GB |
| `ltx-2.5-audio-vae-bf16.safetensors` | `vae` | 0.36 GB |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `latent_upscale_models` | 1.00 GB |

**Deliberately excluded:** the temporal upscaler and duration head (zero references across
all 9 Lightricks 2.5 examples *and* all 3 Comfy-Org 2.5 templates); the distilled LoRA
(2.5's distilled model is a full transformer checkpoint, not a LoRA over dev); the
abliterated Gemma LoRA (no 2.5 equivalent, and superseded by D1).

**int8-convrot is confirmed loadable** by static proof: every quantized layer's
`comfy_quant` blob decodes to `{"format": "int8_tensorwise", "convrot": true,
"convrot_groupsize": 256}`, and ComfyUI v0.32.0 implements exactly that at
`comfy/ops.py:1170,1178`. The `Unknown quantization format` error in ComfyUI issue #14722
is unreachable for this file — that issue is a *feature request* about other architectures,
opened before LTX-2.5 existed.

**bf16 is not a fallback**: 42.02 GB transformer + 26.26 GB encoder = 68.28 GB, which does
not fit a 48 GB L40S under any offload strategy. NVFP4 (18.72 GB) requires Blackwell
(sm_100/120), so it is not an L40S option either. If int8-convrot ever fails, that is a
GPU-class decision, not an engineering workaround.

## Model delivery

Source ladder, in order: **R2-S3 presigned** (primary — the S3 data plane has no CDN
throttle) → **R2 custom domain** (fallback — the CDN proxy throttles sustained multi-GB
pulls to <1 MB/s on the free plan) → **HuggingFace** (last, and **only when `HF_TOKEN` is
set**).

`Lightricks/LTX-2.5` is a **gated repo** — every file returns 401 unauthenticated (2.3's
files return 206). The HF source is therefore appended to the ladder only when a token is
present; otherwise a token-less worker would take three guaranteed 401s per file. Without a
token this is a documented **two**-source ladder.

Integrity is enforced by `aria2c --checksum=sha-256`. **Size alone is never proof of
completion** — aria2 pre-allocates the destination to full size, so a stalled partial looks
complete by size. The worker exits non-zero before starting ComfyUI if any file is missing,
so RunPod replaces it rather than serving half-initialized.

### R2 (provisioned 2026-08-12)

| | |
|---|---|
| Bucket | `anim8-ltx25-models` (WNAM, Standard) |
| Custom domain | `models-ltx25.anim8e2e.ai`, minTLS 1.2, SSL active |

⚠️ **`models.anim8e2e.ai` cannot serve this bucket.** R2 custom domains bind to exactly one
bucket; that hostname is bound to `anim8-ltx23-models` and 404s on 2.5 filenames. Never
re-point it — that would break the live 2.3 service.

## Required endpoint env vars

| Var | Purpose |
|---|---|
| `R2_S3_ENDPOINT`, `R2_S3_ACCESS_KEY_ID`, `R2_S3_SECRET_ACCESS_KEY` | Primary source (Object-Read-only) |
| `R2_S3_BUCKET` | Defaults to `anim8-ltx25-models` |
| `R2_BASE` | Defaults to `https://models-ltx25.anim8e2e.ai` |
| `HF_TOKEN` | Optional. Enables the HF fallback rung. |

**All credentials are injected as RunPod endpoint env vars and are never baked into the
image.** `presign_s3()` in `start.sh` is preserved byte-for-byte from the 2.3 service; the
only sanctioned edits in that block are the `R2_S3_BUCKET` and `R2_BASE` defaults.

## Build-time validation

| Check | Catches |
|---|---|
| `tools/check_node_map.py` | Node ID drift, class-type mismatch, missing input keys |
| `tools/check_node_registration.sh` | Custom-node import failures (boots ComfyUI with no models, diffs `/object_info`) |
| torch assertion in `Dockerfile` | A cu128 wheel older than the 2.7 floor ComfyUI v0.32.0 requires |
| `pytest tests/` | Frame snapping, sigma pairing, validation, error classification, `_set` guards |

The registration check exists because every custom-node `pip install` ends in `|| true`
(deliberately — their requirements pin conflicting torch versions). Without it, a node whose
dependencies failed to install yields a green build and fails at runtime after a 40 GB cold
boot. It converts that into a named build failure.

## Deployment target

L40S 48 GB, 80 GB container disk, max 2 workers — matching the 2.3 endpoint, **pending spike
0c** (VRAM peak). If peak exceeds ~47 GB, swap `ltx-2.5-video-vae-bf16.safetensors` for
`ltx-2.5-video-vae-conv-bf16.safetensors` (1.45 GB, non-diffusion decoder, drop-in) and
re-measure.
