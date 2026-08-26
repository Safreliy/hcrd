from __future__ import annotations

import struct

from experiments.download_lcms_eic_subset import parse_central_directory


def test_parse_single_member_central_directory() -> None:
    name = b"data/example.csv"
    central = struct.pack(
        "<4s6H3L5H2L",
        b"PK\x01\x02",
        20,
        20,
        0,
        8,
        0,
        0,
        0x12345678,
        17,
        29,
        len(name),
        0,
        0,
        0,
        0,
        0,
        101,
    ) + name
    eocd = struct.pack(
        "<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, len(central), 500, 0
    )
    parsed = parse_central_directory(central + eocd)
    member = parsed["data/example.csv"]
    assert member.compressed_size == 17
    assert member.size == 29
    assert member.crc32 == 0x12345678
    assert member.local_header_offset == 101
