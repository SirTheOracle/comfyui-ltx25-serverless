#!/usr/bin/env python3
"""Rewrite start.sh's MODELS table from the manifest the mirror produced.

start.sh ships with REPLACE_SIZE / REPLACE_SHA256 placeholders so it CANNOT be
deployed against unverified values — a wrong sha256 fails every download and hard-exits
the worker. This is the one sanctioned way to fill them in: straight from
_poc/manifest.txt, which mirror_to_r2.py writes using the sizes and hashes it measured
from the bytes it actually uploaded (not from HF metadata).

Reads the manifest from R2 (default) or a local file, validates it against the file
list start.sh already declares, and rewrites the heredoc in place.

    python3 tools/apply_manifest.py                 # pull from R2
    python3 tools/apply_manifest.py --file man.txt  # or from disk
"""
import argparse
import datetime
import hashlib
import hmac
import os
import re
import sys
import urllib.parse
import urllib.request

START_SH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "start.sh")
MANIFEST_KEY = "_poc/manifest.txt"


def presign_get(endpoint, ak, sk, bucket, key, expires=300):
    endpoint = endpoint.rstrip("/")
    host = urllib.parse.urlparse(endpoint).netloc
    now = datetime.datetime.now(datetime.timezone.utc)
    amz, ds = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{ds}/auto/s3/aws4_request"
    uri = "/" + urllib.parse.quote(f"{bucket}/{key}")
    qp = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{ak}/{scope}",
          "X-Amz-Date": amz, "X-Amz-Expires": str(expires), "X-Amz-SignedHeaders": "host"}
    qs = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
                  for k, v in sorted(qp.items()))
    creq = f"GET\n{uri}\n{qs}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD"
    def s(k, m): return hmac.new(k, m.encode(), hashlib.sha256).digest()
    sts = f"AWS4-HMAC-SHA256\n{amz}\n{scope}\n{hashlib.sha256(creq.encode()).hexdigest()}"
    kx = s(s(s(s(("AWS4" + sk).encode(), ds), "auto"), "s3"), "aws4_request")
    sig = hmac.new(kx, sts.encode(), hashlib.sha256).hexdigest()
    return f"{endpoint}{uri}?{qs}&X-Amz-Signature={sig}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="read the manifest from disk instead of R2")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.file:
        manifest = open(args.file).read()
    else:
        ep = os.environ.get("R2_S3_ENDPOINT")
        ak = os.environ.get("R2_S3_ACCESS_KEY_ID")
        sk = os.environ.get("R2_S3_SECRET_ACCESS_KEY")
        bucket = os.environ.get("R2_S3_BUCKET", "anim8-ltx25-models")
        if not all((ep, ak, sk)):
            sys.exit("need R2_S3_ENDPOINT / R2_S3_ACCESS_KEY_ID / R2_S3_SECRET_ACCESS_KEY, "
                     "or pass --file")
        with urllib.request.urlopen(presign_get(ep, ak, sk, bucket, MANIFEST_KEY), timeout=30) as r:
            manifest = r.read().decode()

    rows = [ln.strip() for ln in manifest.strip().splitlines() if ln.strip()]
    parsed = {}
    for ln in rows:
        parts = ln.split("|")
        if len(parts) != 5:
            sys.exit(f"malformed manifest row ({len(parts)} fields): {ln[:90]}")
        base, subdir, size, sha, hf = parts
        if not size.isdigit():
            sys.exit(f"{base}: size {size!r} is not numeric")
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            sys.exit(f"{base}: sha256 {sha!r} is not a 64-char hex digest")
        parsed[base] = ln

    src = open(START_SH).read()
    m = re.search(r"(read -r -d '' MODELS <<'EOF'\n)(.*?)(\nEOF\n)", src, re.S)
    if not m:
        sys.exit("could not locate the MODELS heredoc in start.sh")

    existing = [ln for ln in m.group(2).splitlines() if ln.strip()]
    expected = [ln.split("|")[0] for ln in existing]

    missing = [b for b in expected if b not in parsed]
    extra = [b for b in parsed if b not in expected]
    if missing:
        sys.exit(f"manifest is missing files start.sh declares: {missing}")
    if extra:
        print(f"WARNING: manifest has files start.sh does not declare (ignored): {extra}")

    # Preserve start.sh's ordering; the manifest only supplies values.
    new_body = "\n".join(parsed[b] for b in expected)
    out = src[:m.start(2)] + new_body + src[m.end(2):]

    if "REPLACE_SIZE" in new_body or "REPLACE_SHA256" in new_body:
        sys.exit("refusing to write: manifest still contains placeholders")

    total = sum(int(parsed[b].split("|")[2]) for b in expected)
    print(f"{len(expected)} files, total {total:,}B ({total/1e9:.2f} GB)")
    for b in expected:
        _, sd, size, sha, _ = parsed[b].split("|")
        print(f"  {b[:58]:<58} {int(size)/1e9:>6.2f} GB  {sha[:16]}…")

    if args.dry_run:
        print("\n(dry run — start.sh not modified)")
        return
    open(START_SH, "w").write(out)
    print(f"\n✓ wrote {START_SH}")
    if "REPLACE_SIZE" in out or "REPLACE_SHA256" in out:
        sys.exit("ERROR: placeholders remain elsewhere in start.sh")
    print("✓ no placeholders remain")


if __name__ == "__main__":
    main()
