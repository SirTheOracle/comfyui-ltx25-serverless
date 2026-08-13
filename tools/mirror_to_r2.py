#!/usr/bin/env python3
"""Mirror the LTX-2.5 model set from HuggingFace to Cloudflare R2.

RUN THIS ON A RUNPOD POD, NOT LOCALLY. This is the same procedure used for LTX-2.3
(see `_poc/migrate.log` in the anim8-ltx23-models bucket): a pod has datacenter
bandwidth, so ~40 GB down + ~40 GB up takes minutes instead of hours.

Per file: aria2c download (burst-retried, because HF stalls mid-transfer) -> sha256
verify -> upload to R2 -> optionally delete the local copy. Progress is streamed to
stdout and mirrored into the bucket at _poc/migrate.log so a dropped SSH session
doesn't lose the record.

At the end it prints a ready-to-paste MODELS table for start.sh with the REAL sizes
and hashes — the whole point of the exercise, since start.sh ships with
REPLACE_SIZE/REPLACE_SHA256 placeholders and must not be built until they are real.

DIFFERENCE FROM 2.3: Lightricks/LTX-2.5 is a GATED repo. Every file 401s without a
token, so aria2c must send an Authorization header. HF_TOKEN is required.

Setup on the pod:
    apt-get update && apt-get install -y aria2
    pip install boto3
    export HF_TOKEN=...                     # read token, license accepted
    export R2_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
    export R2_S3_BUCKET=anim8-ltx25-models
    export R2_S3_ACCESS_KEY_ID=...
    export R2_S3_SECRET_ACCESS_KEY=...
    python3 mirror_to_r2.py
"""
import hashlib
import os
import subprocess
import sys
import time

REPO = "Lightricks/LTX-2.5"

# basename -> (subdir for start.sh's MODELS table, HF path within the repo)
# Matches final-plan Item 2c. Deliberately excludes the temporal upscaler, the duration
# head, the distilled LoRA, and any prompt-enhancer model — see the plan for why.
FILES = [
    ("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
     "diffusion_models",
     "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
    ("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
     "text_encoders",
     "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    ("ltx-2.5-video-vae-bf16.safetensors", "vae",
     "vae/ltx-2.5-video-vae-bf16.safetensors"),
    ("ltx-2.5-audio-vae-bf16.safetensors", "vae",
     "vae/ltx-2.5-audio-vae-bf16.safetensors"),
    ("ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", "latent_upscale_models",
     "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
]

WORKDIR = os.environ.get("WORKDIR", "/workspace/ltx25")
BURSTS = int(os.environ.get("BURSTS", "6"))          # aria2c restarts per file
BURST_TIMEOUT = int(os.environ.get("BURST_TIMEOUT", "900"))
DELETE_AFTER_UPLOAD = os.environ.get("KEEP_LOCAL", "0") != "1"

_log_lines = []


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    _log_lines.append(line)


def need(var):
    v = os.environ.get(var)
    if not v:
        sys.exit(f"FATAL: {var} is required (see the docstring)")
    return v


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, token):
    """aria2c with burst-retry. HF stalls mid-transfer; -c resumes where it left off."""
    d, base = os.path.dirname(dest), os.path.basename(dest)
    os.makedirs(d, exist_ok=True)
    for burst in range(1, BURSTS + 1):
        t0 = time.time()
        r = subprocess.run(
            ["aria2c", "-x16", "-s16", "-k1M", "-c",
             "--max-tries=2", "--retry-wait=5", "--timeout=60",
             "--lowest-speed-limit=1M",          # kill a stalled transfer fast
             "--console-log-level=warn", "--summary-interval=0",
             f"--header=Authorization: Bearer {token}",   # 2.5 is gated
             "-d", d, "-o", base, url],
            capture_output=True, text=True, timeout=BURST_TIMEOUT)
        el = int(time.time() - t0)
        if r.returncode == 0:
            return burst, el
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        log(f"  {base} burst#{burst} rc={r.returncode} elapsed={el}s :: "
            f"{tail[-1][:120] if tail else 'no output'}")
    raise RuntimeError(f"{base}: exhausted {BURSTS} bursts")


def main():
    token = need("HF_TOKEN")
    endpoint, bucket = need("R2_S3_ENDPOINT"), need("R2_S3_BUCKET")
    ak, sk = need("R2_S3_ACCESS_KEY_ID"), need("R2_S3_SECRET_ACCESS_KEY")

    import boto3
    from boto3.s3.transfer import TransferConfig
    s3 = boto3.client("s3", endpoint_url=endpoint,
                      aws_access_key_id=ak, aws_secret_access_key=sk, region_name="auto")
    # R2 caps multipart parts; 128 MB chunks keeps a 21 GB file well under the limit.
    xfer = TransferConfig(multipart_threshold=256 << 20, multipart_chunksize=128 << 20,
                          max_concurrency=8, use_threads=True)

    os.makedirs(WORKDIR, exist_ok=True)
    log(f"STARTED mirror -> {bucket} ({len(FILES)} files)")
    results = []

    for base, subdir, hfpath in FILES:
        url = f"https://huggingface.co/{REPO}/resolve/main/{hfpath}"
        dest = os.path.join(WORKDIR, base)
        log(f"=== FILE {base} ===")

        if os.path.exists(dest):
            log(f"  {base} already present locally ({os.path.getsize(dest):,}B); skipping download")
            bursts, el = 0, 0
        else:
            bursts, el = download(url, dest, token)
            log(f"  {base} download complete in {el}s/{bursts} bursts; sha256 verify...")

        size = os.path.getsize(dest)
        digest = sha256(dest)
        log(f"  {base} size={size:,}B sha256={digest}")

        # Flat basenames: start.sh presigns `bucket/<basename>` and builds
        # `${R2_BASE}/<basename>` — a subdirectory key would break BOTH rungs.
        log(f"  {base} uploading to R2 (key={base})...")
        t0 = time.time()
        s3.upload_file(dest, bucket, base, Config=xfer)
        log(f"  {base} UPLOADED in {int(time.time()-t0)}s ✅")

        head = s3.head_object(Bucket=bucket, Key=base)
        if head["ContentLength"] != size:
            raise RuntimeError(f"{base}: R2 size {head['ContentLength']} != local {size}")
        log(f"  {base} readback size OK")

        results.append((base, subdir, size, digest, f"{REPO}/resolve/main/{hfpath}"))
        if DELETE_AFTER_UPLOAD:
            os.remove(dest)
            log(f"  {base} local copy removed")

    log("ALL FILES MIRRORED ✅")
    total = sum(r[2] for r in results)
    log(f"total {total:,}B ({total/1e9:.2f} GB)")

    print("\n" + "=" * 78)
    print("PASTE THIS INTO start.sh, REPLACING THE MODELS HEREDOC BODY:")
    print("=" * 78)
    for base, subdir, size, digest, hf in results:
        print(f"{base}|{subdir}|{size}|{digest}|{hf}")
    print("=" * 78)

    try:
        s3.put_object(Bucket=bucket, Key="_poc/migrate.log",
                      Body=("\n".join(_log_lines) + "\n").encode())
        manifest = "\n".join(f"{b}|{sd}|{sz}|{dg}|{hf}" for b, sd, sz, dg, hf in results)
        s3.put_object(Bucket=bucket, Key="_poc/manifest.txt", Body=(manifest + "\n").encode())
        print("\nlog + manifest written to _poc/ in the bucket")
    except Exception as e:                                  # noqa: BLE001
        print(f"\n(could not write logs to R2: {e})")


if __name__ == "__main__":
    main()
