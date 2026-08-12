#!/bin/bash
# LTX-2.5 serverless worker startup.
#
# Models live in Cloudflare R2 and are pulled in parallel with aria2c + SHA-256 validation.
# PRIMARY source is the R2 S3 API endpoint (r2.cloudflarestorage.com, SigV4-presigned):
# the S3 data plane has no CDN throttle (~630 MB/s measured). The Cloudflare custom-domain
# mirror routes through the CDN proxy, which throttles sustained multi-GB pulls to <1 MB/s
# on the free plan and crash-loops the boot — so it is only a FALLBACK.
# ~40 GB per cold boot (vs ~48 GB for LTX-2.3).
#
# HUGGINGFACE IS GATED FOR LTX-2.5. Every Lightricks/LTX-2.5 file returns 401 without a
# token (measured 2026-08-12; LTX-2.3's files return 206 unauthenticated). The HF source is
# therefore appended ONLY when HF_TOKEN is set — otherwise a token-less worker would take
# three guaranteed 401s per file before failing over. Without a token this is a documented
# TWO-source ladder (R2-S3 -> R2-CDN), not a silent degradation.
#
# IMPORTANT: do NOT trust file size alone for "download complete". aria2 pre-allocates
# the destination to full size immediately, so a stalled partial looks complete by size.
# Integrity is enforced by aria2 --checksum=sha-256; size is only a cheap secondary guard.

echo "=========================================="
echo "Starting ComfyUI LTX-2.5 Video Worker"
echo "=========================================="

MODELS_DIR="${MODELS_DIR:-/comfyui/models}"
# 2.5 has its OWN custom domain. R2 custom domains are bound to a single bucket, so
# models.anim8e2e.ai (bound to anim8-ltx23-models) CANNOT serve 2.5 objects — verified:
# it 404s on a 2.5 filename while 206ing on 2.3 files. Provisioned 2026-08-12,
# SSL active, minTLS 1.2, mirroring the 2.3 binding.
R2_BASE="${R2_BASE:-https://models-ltx25.anim8e2e.ai}"   # CDN custom-domain mirror (fallback; CDN-throttled)
PER_FILE_BUDGET="${PER_FILE_BUDGET:-1200}"          # hard per-file wall-clock cap (seconds)
SPEED_FLOOR="${SPEED_FLOOR:-1M}"                     # abort+retry a source that trickles below this

# R2 S3 API endpoint — PRIMARY source (bypasses the Cloudflare CDN proxy throttle).
# Credentials are injected as RunPod endpoint env vars (Object-Read-only token), never baked
# into the image. If any are unset, presign_s3 returns nothing and we fall back to R2_BASE/HF.
R2_S3_ENDPOINT="${R2_S3_ENDPOINT:-}"                 # e.g. https://<account>.r2.cloudflarestorage.com
R2_S3_BUCKET="${R2_S3_BUCKET:-anim8-ltx25-models}"   # 2.5 bucket — NOT anim8-ltx23-models
R2_S3_ACCESS_KEY_ID="${R2_S3_ACCESS_KEY_ID:-}"
R2_S3_SECRET_ACCESS_KEY="${R2_S3_SECRET_ACCESS_KEY:-}"

# HuggingFace read-only token. Same credential discipline as R2: injected as a RunPod
# endpoint env var, NEVER baked into the image, never echoed.
HF_TOKEN="${HF_TOKEN:-}"

# Presign an R2 object as a 1-hour GET URL (SigV4, Python stdlib only — no boto3 dependency).
# Prints the URL on success; prints nothing and returns 1 if creds are absent.
# ---- PRESERVED BYTE-FOR-BYTE FROM THE LTX-2.3 SERVICE. DO NOT MODIFY. ----
presign_s3() {
    local key="$1"
    { [ -z "$R2_S3_ENDPOINT" ] || [ -z "$R2_S3_ACCESS_KEY_ID" ] || [ -z "$R2_S3_SECRET_ACCESS_KEY" ]; } && return 1
    KEY="$key" python3 - <<'PYEOF'
import os, hashlib, hmac, datetime, urllib.parse
endpoint = os.environ["R2_S3_ENDPOINT"].rstrip("/"); bucket = os.environ["R2_S3_BUCKET"]; key = os.environ["KEY"]
ak = os.environ["R2_S3_ACCESS_KEY_ID"]; sk = os.environ["R2_S3_SECRET_ACCESS_KEY"]
region, service = "auto", "s3"
host = urllib.parse.urlparse(endpoint).netloc
now = datetime.datetime.now(datetime.timezone.utc)
amzdate = now.strftime("%Y%m%dT%H%M%SZ"); datestamp = now.strftime("%Y%m%d")
scope = f"{datestamp}/{region}/{service}/aws4_request"
uri = "/" + urllib.parse.quote(f"{bucket}/{key}")
qp = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{ak}/{scope}",
      "X-Amz-Date": amzdate, "X-Amz-Expires": "3600", "X-Amz-SignedHeaders": "host"}
qs = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}" for k, v in sorted(qp.items()))
creq = f"GET\n{uri}\n{qs}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD"
def _s(k, m): return hmac.new(k, m.encode(), hashlib.sha256).digest()
sts = f"AWS4-HMAC-SHA256\n{amzdate}\n{scope}\n{hashlib.sha256(creq.encode()).hexdigest()}"
kx = _s(_s(_s(_s(("AWS4" + sk).encode(), datestamp), region), service), "aws4_request")
sig = hmac.new(kx, sts.encode(), hashlib.sha256).hexdigest()
print(f"{endpoint}{uri}?{qs}&X-Amz-Signature={sig}")
PYEOF
}
# ---- END PRESERVED BLOCK ----

# basename | dest-subdir | size(bytes) | sha256 | huggingface resolve path
#
# !!! SIZES AND SHA-256 BELOW ARE PLACEHOLDERS !!!
# They MUST be replaced with values measured from the actual uploaded R2 objects
# (`stat -c%s` + `sha256sum`) during the mirror seed. The sizes are HF-API
# approximations and the hashes are not yet known. The build must not ship until
# these are real — a wrong sha256 fails every download and hard-exits the worker.
#
# Dropped vs LTX-2.3 (deliberate):
#   - the abliterated Gemma CLIP LoRA: no 2.5 equivalent exists, and it was load-bearing
#     ONLY for generated_audio prompt auto-expansion, which this service drops.
#   - the distilled LoRA: 2.5's distilled model is a FULL transformer checkpoint, not a
#     LoRA over dev, so LoraLoaderModelOnly disappears from both graphs.
#   - the temporal upscaler and duration head: zero references across all 9 Lightricks
#     2.5 examples and all 3 Comfy-Org 2.5 templates.
read -r -d '' MODELS <<'EOF'
ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors|diffusion_models|REPLACE_SIZE|REPLACE_SHA256|Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors|text_encoders|REPLACE_SIZE|REPLACE_SHA256|Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
ltx-2.5-video-vae-bf16.safetensors|vae|REPLACE_SIZE|REPLACE_SHA256|Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors
ltx-2.5-audio-vae-bf16.safetensors|vae|REPLACE_SIZE|REPLACE_SHA256|Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors
ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors|latent_upscale_models|REPLACE_SIZE|REPLACE_SHA256|Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
EOF

if [ -z "$HF_TOKEN" ]; then
    echo "  ⓘ HF_TOKEN unset — HuggingFace fallback DISABLED (LTX-2.5 is a gated repo)."
    echo "    Source ladder for this boot: R2-S3 -> R2-CDN."
fi

download_model() {
    local base="$1" subdir="$2" size="$3" sha="$4" hf="$5"
    local dir="${MODELS_DIR}/${subdir}"
    local dest="${dir}/${base}"
    mkdir -p "$dir"

    # Already present at exact size? Fresh downloads are checksum-validated, so an
    # exact-size existing file is a safe skip (helps warm disk / re-entrant boots).
    if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" = "$size" ]; then
        echo "  ✓ present: $base"
        return 0
    fi
    rm -f "$dest" "$dest.aria2"

    local -a srcs=()
    local s3url
    s3url="$(presign_s3 "$base")"        # PRIMARY: R2 S3 data plane (no CDN throttle)
    [ -n "$s3url" ] && srcs+=("$s3url")
    [ -n "$R2_BASE" ] && srcs+=("${R2_BASE}/${base}")   # FALLBACK: R2 CDN custom domain
    # LAST: HuggingFace — appended only with a token, because LTX-2.5 is gated and an
    # unauthenticated request can only ever 401.
    if [ -n "$HF_TOKEN" ]; then
        srcs+=("https://huggingface.co/${hf}")
    fi

    local src label attempt
    local -a extra
    for src in "${srcs[@]}"; do
        extra=()
        case "$src" in
            *.r2.cloudflarestorage.com/*) label="R2-S3" ;;
            *huggingface.co/*)            label="HF"; extra+=(--header="Authorization: Bearer ${HF_TOKEN}") ;;
            *)                            label="R2-CDN" ;;
        esac
        attempt=1
        while [ "$attempt" -le 3 ]; do
            echo "  ↓ ${base} from ${label} (attempt ${attempt})"
            if timeout "$PER_FILE_BUDGET" aria2c \
                    -x16 -s16 -k1M -c \
                    --checksum="sha-256=${sha}" \
                    --lowest-speed-limit="$SPEED_FLOOR" \
                    --max-tries=2 --retry-wait=5 --timeout=60 \
                    --console-log-level=warn --summary-interval=0 \
                    "${extra[@]}" \
                    -d "$dir" -o "$base" "$src" \
               && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" = "$size" ]; then
                echo "  ✓ ${base} (${label}, sha256 verified)"
                return 0
            fi
            echo "  ⚠ ${base} failed/invalid from ${label} (attempt ${attempt}); cleaning + retrying"
            rm -f "$dest" "$dest.aria2"
            attempt=$((attempt + 1))
            sleep $((attempt * 5))
        done
    done
    echo "  ✗ FAILED all sources: ${base}"
    return 1
}

echo ""
echo "=========================================="
echo "Downloading models (R2 S3 primary → R2 CDN → HF[token-gated] fallback)..."
echo "=========================================="
fail=0
while IFS='|' read -r base subdir size sha hf; do
    [ -z "$base" ] && continue
    download_model "$base" "$subdir" "$size" "$sha" "$hf" || fail=1
done < <(printf '%s\n' "$MODELS")

echo ""
echo "=========================================="
echo "Verifying models (fail-hard before ComfyUI)..."
echo "=========================================="
missing=0
while IFS='|' read -r base subdir size sha hf; do
    [ -z "$base" ] && continue
    dest="${MODELS_DIR}/${subdir}/${base}"
    actual=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$actual" = "$size" ]; then
        echo "✓ $base"
    else
        echo "✗ MISSING/incomplete: $base ($actual/$size)"
        missing=1
    fi
done < <(printf '%s\n' "$MODELS")

if [ "$fail" = "1" ] || [ "$missing" = "1" ]; then
    echo "FATAL: model set incomplete — exiting so RunPod replaces this worker (never serve half-initialized)."
    exit 1
fi

echo ""
echo "=========================================="
echo "Starting ComfyUI..."
echo "=========================================="
cd /comfyui
python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch &

echo "Waiting for ComfyUI..."
sleep 15

for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "✓ ComfyUI is running!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

echo ""
echo "=========================================="
echo "Starting Handler..."
echo "=========================================="
cd /
python -u /handler.py
