"""Fetch only the JSON members (questions + schema) from BIRD's remote zips
without downloading the multi-GB sqlite payload.

BIRD's official train.zip is ~8.9 GB but the routing task needs only the
question list (train.json) and the per-DB schema (train_tables.json), a few tens
of MB. ZIP supports per-member HTTP range extraction: read the End-Of-Central-
Directory (ZIP64-aware) from the tail, parse the central directory, then
range-GET + inflate only the wanted members.

Run (from experiment/):
  .venv/bin/python scripts/fetch_bird_json_remote.py \
    --url https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip \
    --out ../dataset/sql_bird/train --want train/train.json train/train_tables.json
"""

from __future__ import annotations

import argparse
import struct
import urllib.request
import zlib
from pathlib import Path


def http_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def http_range(url: str, start: int, end: int) -> bytes:
    """Inclusive byte range [start, end]."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def find_central_dir(url: str, size: int) -> tuple[int, int]:
    """Return (cd_offset, cd_size), ZIP64-aware."""
    tail_len = min(size, 1 << 16)
    tail = http_range(url, size - tail_len, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("EOCD not found in tail")
    cd_size = struct.unpack_from("<I", tail, eocd + 12)[0]
    cd_off = struct.unpack_from("<I", tail, eocd + 16)[0]
    if cd_off != 0xFFFFFFFF and cd_size != 0xFFFFFFFF:
        return cd_off, cd_size
    # ZIP64 path: locate the EOCD64 locator just before the EOCD.
    loc = tail.rfind(b"PK\x06\x07")
    if loc < 0:
        raise RuntimeError("ZIP64 EOCD locator not found")
    eocd64_off = struct.unpack_from("<Q", tail, loc + 8)[0]
    rec = http_range(url, eocd64_off, eocd64_off + 56)
    if rec[:4] != b"PK\x06\x06":
        raise RuntimeError("ZIP64 EOCD record signature mismatch")
    cd_size = struct.unpack_from("<Q", rec, 40)[0]
    cd_off = struct.unpack_from("<Q", rec, 48)[0]
    return cd_off, cd_size


def parse_central_dir(cd: bytes) -> list[dict]:
    """Parse central-directory headers → member records (ZIP64-aware)."""
    members = []
    i = 0
    while i + 46 <= len(cd) and cd[i : i + 4] == b"PK\x01\x02":
        method = struct.unpack_from("<H", cd, i + 10)[0]
        comp_size = struct.unpack_from("<I", cd, i + 20)[0]
        uncomp_size = struct.unpack_from("<I", cd, i + 24)[0]
        name_len = struct.unpack_from("<H", cd, i + 28)[0]
        extra_len = struct.unpack_from("<H", cd, i + 30)[0]
        cmt_len = struct.unpack_from("<H", cd, i + 32)[0]
        lho = struct.unpack_from("<I", cd, i + 42)[0]
        name = cd[i + 46 : i + 46 + name_len].decode("utf-8", "replace")
        extra = cd[i + 46 + name_len : i + 46 + name_len + extra_len]
        # ZIP64 extended info for any 0xFFFFFFFF fields, in fixed order.
        if 0xFFFFFFFF in (comp_size, uncomp_size, lho):
            j = 0
            while j + 4 <= len(extra):
                tag, sz = struct.unpack_from("<HH", extra, j)
                if tag == 0x0001:
                    vals = extra[j + 4 : j + 4 + sz]
                    k = 0
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = struct.unpack_from("<Q", vals, k)[0]; k += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size = struct.unpack_from("<Q", vals, k)[0]; k += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack_from("<Q", vals, k)[0]; k += 8
                    break
                j += 4 + sz
        members.append({"name": name, "method": method, "comp_size": comp_size,
                        "uncomp_size": uncomp_size, "lho": lho})
        i += 46 + name_len + extra_len + cmt_len
    return members


def extract_member(url: str, m: dict) -> bytes:
    """Range-GET a member's local header + data, then inflate."""
    head = http_range(url, m["lho"], m["lho"] + 30)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"local header mismatch for {m['name']}")
    n = struct.unpack_from("<H", head, 26)[0]
    e = struct.unpack_from("<H", head, 28)[0]
    data_start = m["lho"] + 30 + n + e
    raw = http_range(url, data_start, data_start + m["comp_size"] - 1)
    if m["method"] == 0:
        return raw
    return zlib.decompress(raw, -15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--want", nargs="*", default=[], help="exact member names; if empty, list all *.json")
    args = ap.parse_args()

    size = http_size(args.url)
    print(f"archive size: {size/1e9:.2f} GB", flush=True)
    cd_off, cd_size = find_central_dir(args.url, size)
    print(f"central dir: offset={cd_off} size={cd_size}", flush=True)
    cd = http_range(args.url, cd_off, cd_off + cd_size - 1)
    members = parse_central_dir(cd)
    print(f"{len(members)} members", flush=True)
    jsons = [m for m in members if m["name"].lower().endswith(".json")]
    print("JSON members:")
    for m in jsons:
        print(f"  {m['name']}  ({m['uncomp_size']/1e6:.1f} MB)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.want) if args.want else {m["name"] for m in jsons}
    for m in members:
        if m["name"] in want:
            print(f"fetch {m['name']} ...", flush=True)
            data = extract_member(args.url, m)
            dest = out / Path(m["name"]).name
            dest.write_bytes(data)
            print(f"  wrote {dest} ({len(data)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
