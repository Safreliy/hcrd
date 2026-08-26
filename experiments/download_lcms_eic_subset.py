#!/usr/bin/env python3
"""Download only the frozen E1 files from the 4 GB Zenodo ZIP archive.

Zenodo supports HTTP byte ranges.  Reading the ZIP central directory and the
four relevant DEFLATE members avoids downloading the unrelated raw mzML and
Excel files.  No dataset content is redistributed by this repository.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


URL = (
    "https://zenodo.org/records/3756211/files/"
    "A_dataset_for_evaluation_of_peak_detection_methods_v3.zip?download=1"
)
ARCHIVE_MD5 = "d86faf581d85b640384ebb27d11c2085"
RECORD_DOI = "10.5281/zenodo.3756211"
MEMBERS = (
    "1_Processed_data/EIC_data/EIC_data.RData",
    "1_Processed_data/Selected_subset_MZmine_table/"
    "Table_2_Holtemme_Stichtag2015_water_ESIpos_MRP2.7_gap-filled_clean.csv",
    "2_Classified_Data/Classification/Classification_before_cleanup.csv",
    "2_Classified_Data/Classification/Classification_after_cleanup.csv",
)


@dataclass(frozen=True)
class ZipMember:
    name: str
    method: int
    crc32: int
    compressed_size: int
    size: int
    local_header_offset: int


def _request_range(url: str, start: int | None, end: int | None) -> bytes:
    if start is None:
        value = f"bytes=-{end}"
    else:
        value = f"bytes={start}-{'' if end is None else end}"
    request = urllib.request.Request(url, headers={"Range": value})
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 206:
            raise RuntimeError(f"server ignored Range {value}: HTTP {response.status}")
        return response.read()


def parse_central_directory(tail: bytes) -> dict[str, ZipMember]:
    eocd_index = tail.rfind(b"PK\x05\x06")
    if eocd_index < 0:
        raise ValueError("ZIP end-of-central-directory record not found")
    _, _, _, count_disk, count, directory_size, _, comment_size = struct.unpack_from(
        "<4s4H2LH", tail, eocd_index
    )
    if count_disk != count or comment_size != len(tail) - eocd_index - 22:
        raise ValueError("multi-disk or malformed ZIP archive")
    position = eocd_index - directory_size
    members: dict[str, ZipMember] = {}
    for _ in range(count):
        if tail[position : position + 4] != b"PK\x01\x02":
            raise ValueError("malformed central-directory entry")
        header = struct.unpack_from("<4s6H3L5H2L", tail, position)
        name_length, extra_length, comment_length = header[10:13]
        name_start = position + 46
        name = tail[name_start : name_start + name_length].decode("utf-8")
        members[name] = ZipMember(
            name=name,
            method=header[4],
            crc32=header[7],
            compressed_size=header[8],
            size=header[9],
            local_header_offset=header[16],
        )
        position += 46 + name_length + extra_length + comment_length
    return members


def _member_data_start(url: str, member: ZipMember) -> int:
    header = _request_range(
        url, member.local_header_offset, member.local_header_offset + 65535
    )
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"invalid local header for {member.name}")
    fixed = struct.unpack_from("<4s5H3L2H", header, 0)
    name_length, extra_length = fixed[-2:]
    embedded_name = header[30 : 30 + name_length].decode("utf-8")
    if embedded_name != member.name:
        raise ValueError(f"local/central name mismatch for {member.name}")
    return member.local_header_offset + 30 + name_length + extra_length


def download_member(url: str, member: ZipMember, output: Path) -> dict[str, object]:
    if member.method != 8:
        raise ValueError(f"unsupported ZIP compression method {member.method}")
    start = _member_data_start(url, member)
    compressed = _request_range(url, start, start + member.compressed_size - 1)
    if len(compressed) != member.compressed_size:
        raise RuntimeError(f"truncated compressed member {member.name}")
    decompressed = zlib.decompress(compressed, -zlib.MAX_WBITS)
    if len(decompressed) != member.size:
        raise RuntimeError(f"size mismatch for {member.name}")
    crc32 = binascii.crc32(decompressed) & 0xFFFFFFFF
    if crc32 != member.crc32:
        raise RuntimeError(f"CRC mismatch for {member.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(decompressed)
    return {
        "archive_member": member.name,
        "size": member.size,
        "crc32": f"{member.crc32:08x}",
        "sha256": hashlib.sha256(decompressed).hexdigest(),
        "local_file": output.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tail = _request_range(args.url, None, 131072)
    directory = parse_central_directory(tail)
    missing = sorted(set(MEMBERS) - set(directory))
    if missing:
        raise RuntimeError(f"members absent from archive: {missing}")
    files = []
    for name in MEMBERS:
        output = args.output_dir / Path(name).name
        files.append(download_member(args.url, directory[name], output))
        print(f"verified {name} -> {output}")
    manifest = {
        "record_doi": RECORD_DOI,
        "archive_url": args.url,
        "archive_md5_published": ARCHIVE_MD5,
        "download_mode": "verified HTTP Range extraction from ZIP v3",
        "files": files,
    }
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
