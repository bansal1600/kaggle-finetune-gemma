#!/usr/bin/env python3
"""Stage 0 item 13: prove every safetensors shard of a model is complete, without downloading it.

Why: a Kaggle model copy can be silently truncated (gemma-4-12b-it-qat-q4_0-unquantized/2 stops at
3.10 of its 23.92 GB). Nothing flags it: the file listing simply reports the short size, and loading
only fails deep inside a GPU session. So the check reads what the file itself promises (its header)
and asks the server how many bytes it really has.

How a safetensors file is laid out:
    [8 bytes: N, little-endian u64] [N bytes: JSON header] [tensor data ...]
The header maps each tensor name to {dtype, shape, data_offsets: [begin, end]}, with offsets
relative to the start of the data block. So the file must be at least 8 + N + max(end) bytes.

Per shard, two HTTP range reads (bytes 0-7, then the header) fetch the promise. The server's
Content-Range ("bytes 0-7/TOTAL") gives the real object size, and one more 1-byte read at the
last required byte proves the data is actually served. A local directory (for example a Kaggle
mount, /kaggle/input/models/...) is checked the same way with plain file reads.

It also checks model.safetensors.index.json: every shard it names exists, every tensor it maps is
in that shard's header (and vice versa), and each tensor's byte span matches dtype x shape.

Usage:
    python scripts/check_model_shards.py google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2
    python scripts/check_model_shards.py HANDLE_A HANDLE_B --probe-mb 64 --json out.json
    python scripts/check_model_shards.py /kaggle/input/models/google/gemma-4/transformers/<instance>/2

Exit code 1 if any shard is truncated, missing or inconsistent with the index. Kaggle access uses
the kaggle CLI's credentials (read-only API calls only).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

INDEX = "model.safetensors.index.json"
MAX_HEADER = 100 * 2**20  # real headers are well under 1 MB; anything bigger means garbage bytes
# Bytes per element. F4 / F6 types are packed below one byte, so the span check skips them.
DTYPE_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2,
               "U16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


class LocalSource:
    """A model directory on disk, e.g. a Kaggle input mount."""

    def __init__(self, root: Path):
        self.root = root
        self.label = str(root)

    def files(self) -> dict[str, int]:
        """Top-level files only, like a Hugging Face model dir (subfolders are not shards)."""
        return {p.name: p.stat().st_size for p in self.root.iterdir() if p.is_file()}

    def read(self, name: str, start: int, end: int) -> tuple[bytes, int]:
        """Return bytes [start, end) and the real file size."""
        path = self.root / name
        with path.open("rb") as f:
            f.seek(start)
            return f.read(end - start), path.stat().st_size


class KaggleSource:
    """A Kaggle model version, read through signed GCS URLs with HTTP Range requests."""

    def __init__(self, handle: str):
        parts = handle.strip("/").split("/")
        if len(parts) != 5 or not parts[4].isdigit():
            raise SystemExit(f"handle must be owner/model/framework/instance/version, got {handle!r}")
        self.owner, self.model, self.framework, self.instance, version = parts
        self.version = int(version)
        self.label = handle
        self._urls: dict[str, str] = {}
        from kaggle.api.kaggle_api_extended import KaggleApi  # the kaggle CLI package; only needed here

        self.api = KaggleApi()
        self.api.authenticate()

    def _request(self, cls):
        from kagglesdk.models.types.model_enums import ModelFramework

        r = cls()
        r.owner_slug, r.model_slug, r.instance_slug, r.version_number = self.owner, self.model, self.instance, self.version
        r.framework = self.api.lookup_enum(ModelFramework, ModelFramework.MODEL_FRAMEWORK_API, self.framework)
        return r

    def files(self) -> dict[str, int]:
        """Sizes as the listing reports them (metadata, not proof the bytes exist)."""
        from kagglesdk.models.types.model_api_service import ApiListModelInstanceVersionFilesRequest

        out, token = {}, None
        with self.api.build_kaggle_client() as client:
            while True:
                req = self._request(ApiListModelInstanceVersionFilesRequest)
                req.page_size = 100
                if token:
                    req.page_token = token
                resp = client.models.model_api_client.list_model_instance_version_files(req)
                out.update({f.name: int(f.size) for f in resp.files})
                token = resp.next_page_token
                if not token:
                    return out

    def _url(self, name: str) -> str:
        """The download API answers with a 302 to a short-lived signed GCS URL; keep only the URL.

        The SDK opens the download with stream=True, so closing it right away transfers no data.
        """
        if name not in self._urls:
            from kagglesdk.models.types.model_api_service import ApiDownloadModelInstanceVersionRequest

            req = self._request(ApiDownloadModelInstanceVersionRequest)
            req.path = name
            with self.api.build_kaggle_client() as client:
                resp = client.models.model_api_client.download_model_instance_version(req)
                self._urls[name] = resp.url
                resp.close()
        return self._urls[name]

    def read(self, name: str, start: int, end: int) -> tuple[bytes, int]:
        """Return bytes [start, end) and the real object size (from Content-Range).

        A Range past the end of the object returns 416, which here means "file is shorter":
        return b"" and the size the server reports.
        """
        import requests

        for attempt in range(4):
            try:
                r = requests.get(self._url(name), headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
                continue
            content_range = r.headers.get("Content-Range", "")  # "bytes a-b/TOTAL" or "bytes */TOTAL"
            size = int(content_range.rsplit("/", 1)[1]) if "/" in content_range else -1
            if r.status_code == 206:
                return r.content, size
            if r.status_code == 416:
                return b"", size
            if r.status_code in (400, 401, 403):  # signed URL expired: get a fresh one
                self._urls.pop(name, None)
            if attempt == 3:
                r.raise_for_status()
                raise RuntimeError(f"{name}: unexpected HTTP {r.status_code} for a Range request")
            time.sleep(2 ** attempt)
        raise AssertionError("unreachable")


def header_problems(header: dict) -> tuple[int, list[str]]:
    """Return (end of the data block the header promises, list of inconsistencies).

    safetensors requires the tensors to tile the data block with no gaps or overlaps, and each
    span to be exactly elements x bytes-per-element.
    """
    problems, spans = [], []
    for name, t in header.items():
        if name == "__metadata__":
            continue
        begin, end = t["data_offsets"]
        spans.append((begin, end, name))
        width = DTYPE_BYTES.get(t["dtype"])
        if width is not None and end - begin != math.prod(t["shape"]) * width:
            problems.append(f"{name}: span {end - begin} B != {t['dtype']}{t['shape']}")
    spans.sort()
    cursor = 0
    for begin, end, name in spans:
        if begin != cursor:
            problems.append(f"{name}: starts at {begin}, expected {cursor} (gap or overlap)")
        cursor = max(cursor, end)
    return (spans[-1][1] if spans else 0), problems


def check_shard(src, name: str, listed: int | None) -> dict:
    row = {"shard": name, "listed": listed, "actual": None, "required": None, "tensors": 0, "ok": False, "notes": []}
    head, size = src.read(name, 0, 8)
    row["actual"] = size
    if len(head) < 8:
        row["notes"].append("shorter than 8 bytes")
        return row
    (n,) = struct.unpack("<Q", head)
    if n > MAX_HEADER or 8 + n > size:
        row["notes"].append(f"header length {n} is impossible for a {size} B file")
        return row
    try:
        header = json.loads(src.read(name, 8, 8 + n)[0])
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        row["notes"].append(f"header is not JSON: {e}")
        return row
    data_end, problems = header_problems(header)
    row["tensors"] = len([k for k in header if k != "__metadata__"])
    row["header_bytes"], row["data_bytes"] = n, data_end
    row["required"] = 8 + n + data_end
    row["header"] = header
    row["notes"] += problems[:5] + ([f"... {len(problems) - 5} more"] if len(problems) > 5 else [])
    if size < row["required"]:
        row["notes"].append(f"TRUNCATED: has {size} of {row['required']} B ({100 * size / row['required']:.1f}%)")
    elif len(src.read(name, row["required"] - 1, row["required"])[0]) != 1:
        row["notes"].append("last required byte is not served")
    else:
        row["ok"] = not problems
        if size > row["required"]:
            row["notes"].append(f"{size - row['required']} trailing bytes (harmless)")
    if listed is not None and listed != size:
        row["notes"].append(f"listing says {listed} B, server has {size} B")
    return row


def check_index(src, files: dict[str, int], rows: dict[str, dict]) -> tuple[dict, list[str]]:
    """Cross-check model.safetensors.index.json against the files and the shard headers."""
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    if INDEX not in files:
        # A single-file model has no index; more than one shard without one cannot be loaded.
        return {"present": False}, ([] if len(shards) == 1 else [f"{INDEX} missing for {len(shards)} shards"])
    raw, _ = src.read(INDEX, 0, files[INDEX])
    index = json.loads(raw)
    weight_map: dict[str, str] = index.get("weight_map", {})
    named = sorted(set(weight_map.values()))
    problems = [f"index names {s}, which is not in the model files" for s in named if s not in files]
    problems += [f"{s} is in the files but not in the index" for s in shards if s not in named]
    in_headers = {t: s for s, r in rows.items() for t in r.get("header", {}) if t != "__metadata__"}
    wrong = [t for t, s in weight_map.items() if s in rows and "header" in rows[s] and in_headers.get(t) != s]
    unmapped = [t for t in in_headers if t not in weight_map]
    if wrong:
        problems.append(f"{len(wrong)} indexed tensors not in their shard's header, e.g. {wrong[0]}")
    if unmapped:
        problems.append(f"{len(unmapped)} header tensors missing from the index, e.g. {unmapped[0]}")
    info = {"present": True, "shards_named": len(named), "tensors_indexed": len(weight_map),
            "tensors_in_headers": len(in_headers), "metadata_total_size": index.get("metadata", {}).get("total_size"),
            "header_data_bytes": sum(r.get("data_bytes", 0) for r in rows.values())}
    return info, problems


def probe_speed(src, name: str, size: int, mb: int) -> float:
    """MB/s (1 MB = 1e6 B) for one sequential read of `mb` MB from the middle of `name`.

    It measures the link from wherever this script runs: a laptop for a Kaggle handle, or the
    model mount itself when run inside a Kaggle notebook on /kaggle/input/... (a fresh mount is
    lazily fetched from GCS, so a second run of the same bytes can be much faster: page cache).
    """
    start = max(0, size // 2 - mb * 500_000)
    t0 = time.monotonic()
    data, _ = src.read(name, start, start + mb * 1_000_000)
    return len(data) / 1e6 / (time.monotonic() - t0)


def check_model(target: str, probe_mb: int) -> dict:
    src = LocalSource(Path(target)) if Path(target).is_dir() else KaggleSource(target)
    files = src.files()
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    rows = {s: check_shard(src, s, files[s]) for s in shards}
    index, index_problems = check_index(src, files, rows)
    actual = sum(r["actual"] or 0 for r in rows.values())
    result = {"model": src.label, "shards": [{k: v for k, v in r.items() if k != "header"} for r in rows.values()],
              "index": index, "index_problems": index_problems, "shard_bytes": actual,
              "all_files_bytes": sum(files.values()),
              "ok": bool(rows) and all(r["ok"] for r in rows.values()) and not index_problems}
    if probe_mb and shards:
        biggest = max(shards, key=lambda s: rows[s]["actual"] or 0)
        result["probe_mb_per_s"] = round(probe_speed(src, biggest, rows[biggest]["actual"], probe_mb), 1)
    return result


def gb(v: int | None) -> str:
    return f"{v / 1e9:.3f}" if v is not None else "-"


def report(r: dict) -> None:
    print(f"\n== {r['model']}")
    print(f"{'shard':36} {'tensors':>7} {'required GB':>12} {'actual GB':>10} {'listed GB':>10}  status")
    for s in r["shards"]:
        status = "OK" if s["ok"] else "FAIL"
        print(f"{s['shard']:36} {s['tensors']:>7} {gb(s['required']):>12} {gb(s['actual']):>10} {gb(s['listed']):>10}  "
              f"{status}{'  ' + '; '.join(s['notes']) if s['notes'] else ''}")
    if not r["shards"]:
        print("no .safetensors files found")
    idx = r["index"]
    if idx.get("present"):
        total = idx["metadata_total_size"]
        print(f"index: {idx['shards_named']} shards named, {idx['tensors_indexed']} tensors indexed, "
              f"{idx['tensors_in_headers']} in shard headers; metadata total_size {total} B, "
              f"header data blocks {idx['header_data_bytes']} B")
    for p in r["index_problems"]:
        print(f"index FAIL: {p}")
    mb = r["shard_bytes"] / 1e6
    print(f"shards total {mb / 1e3:.2f} GB (all files {r['all_files_bytes'] / 1e9:.2f} GB)")
    rates = [("measured", r["probe_mb_per_s"])] if r.get("probe_mb_per_s") else []
    rates += [("assumed", 100), ("assumed", 500)]
    print("read time: " + ", ".join(f"{mb / v / 60:.1f} min at {v} MB/s ({k})" for k, v in rates))
    print("RESULT:", "COMPLETE" if r["ok"] else "INCOMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="Kaggle handle owner/model/framework/instance/version, or a local dir")
    parser.add_argument("--probe-mb", type=int, default=0, help="time a read of this many MB to estimate read speed")
    parser.add_argument("--json", type=Path, help="write the full results here")
    args = parser.parse_args()

    results = [check_model(m, args.probe_mb) for m in args.models]
    for r in results:
        report(r)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
