"""
tfrecord_io.py
──────────────────────────────────────────────────────────────────────────────
Pure-Python TFRecord reader / writer.  No TensorFlow dependency.
Produces files that are fully compatible with tf.data.TFRecordDataset.

Schema written/expected:
  "image"    – BytesList   (raw uint8 bytes, row-major)
  "label"    – Int64List   (single int)
  "height"   – Int64List   (single int)
  "width"    – Int64List   (single int)
  "channels" – Int64List   (single int)
"""

from __future__ import annotations

import struct

# ── CRC32C (Castagnoli, poly = 0x1EDC6F41 reflected = 0x82F63B78) ────────────

_CRC_TABLE: list[int] = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x82F63B78 if _c & 1 else _c >> 1
    _CRC_TABLE.append(_c)
del _i, _c


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    tbl = _CRC_TABLE
    for b in data:
        crc = (crc >> 8) ^ tbl[(crc ^ b) & 0xFF]
    return crc ^ 0xFFFFFFFF


def _masked(crc: int) -> int:
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


# ── Minimal protobuf encoder ──────────────────────────────────────────────────

def _varint(n: int) -> bytes:
    out: list[int] = []
    while n > 0x7F:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n & 0x7F)
    return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _len_delim(field: int, payload: bytes) -> bytes:
    """Wire type 2 (length-delimited) field."""
    return _tag(field, 2) + _varint(len(payload)) + payload


def _int64_field(field: int, value: int) -> bytes:
    """Wire type 0 (varint) field for int64."""
    if value < 0:
        value += 1 << 64
    return _tag(field, 0) + _varint(value)


def _encode_example(image_bytes: bytes, label: int,
                    height: int, width: int, channels: int) -> bytes:
    """Encode one sample as a serialised tf.train.Example protobuf."""

    # Feature{bytes_list{value: [image_bytes]}}
    bytes_list    = _len_delim(1, image_bytes)         # BytesList.value[0]
    image_feature = _len_delim(1, bytes_list)          # Feature.bytes_list

    def _int64_feature(n: int) -> bytes:
        int64_list = _int64_field(1, n)                # Int64List.value[0]
        return _len_delim(3, int64_list)               # Feature.int64_list

    def _map_entry(key: str, feature_bytes: bytes) -> bytes:
        # Protobuf map<string,Feature> entry: field 1=key, field 2=value
        entry = _len_delim(1, key.encode()) + _len_delim(2, feature_bytes)
        return _len_delim(1, entry)                    # Features.feature entry

    features_msg = (
        _map_entry("image",    image_feature)      +
        _map_entry("label",    _int64_feature(label))    +
        _map_entry("height",   _int64_feature(height))   +
        _map_entry("width",    _int64_feature(width))    +
        _map_entry("channels", _int64_feature(channels)) +
        b""
    )
    return _len_delim(1, features_msg)               # Example.features


# ── Minimal protobuf decoder ──────────────────────────────────────────────────

def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _read_field(buf: bytes, pos: int) -> tuple[int, int, object, int]:
    tag, pos = _read_varint(buf, pos)
    field, wire = tag >> 3, tag & 7
    if wire == 0:
        val, pos = _read_varint(buf, pos)
        return field, wire, val, pos
    if wire == 2:
        length, pos = _read_varint(buf, pos)
        val = buf[pos: pos + length]; pos += length
        return field, wire, val, pos
    raise ValueError(f"Unsupported protobuf wire type {wire} in TFRecord")


def _iter_fields(buf: bytes):
    pos = 0
    while pos < len(buf):
        field, wire, val, pos = _read_field(buf, pos)
        yield field, wire, val


def _decode_example(data: bytes) -> tuple[bytes, int, int, int, int]:
    """Return (image_bytes, label, height, width, channels)."""
    image_bytes = b""
    label = height = width = channels = 0

    for f1, w1, v1 in _iter_fields(data):
        if f1 != 1 or w1 != 2:
            continue
        # v1 = Features message
        for f2, w2, v2 in _iter_fields(v1):
            if f2 != 1 or w2 != 2:
                continue
            # v2 = map entry (key + Feature)
            key: str = ""
            feature: bytes = b""
            for fe, we, ve in _iter_fields(v2):
                if fe == 1 and we == 2: key = ve.decode()
                if fe == 2 and we == 2: feature = ve

            if not key:
                continue

            # Decode Feature
            for ff, wf, vf in _iter_fields(feature):
                if ff == 1 and wf == 2:          # BytesList
                    for fbl, wbl, vbl in _iter_fields(vf):
                        if fbl == 1 and wbl == 2:
                            if key == "image":
                                image_bytes = vbl
                if ff == 3 and wf == 2:          # Int64List
                    for fil, wil, vil in _iter_fields(vf):
                        if fil == 1 and wil == 0:
                            n = vil if vil < (1 << 63) else vil - (1 << 64)
                            if key == "label":    label    = int(n)
                            if key == "height":   height   = int(n)
                            if key == "width":    width    = int(n)
                            if key == "channels": channels = int(n)

    return image_bytes, label, height, width, channels


# ── TFRecord frame I/O ────────────────────────────────────────────────────────

def write_tfrecord(path: str, records: list[bytes]) -> None:
    """Write a sequence of serialised protobuf bytes as a TFRecord file."""
    with open(path, "wb") as fh:
        for data in records:
            lb = struct.pack("<Q", len(data))
            fh.write(lb)
            fh.write(struct.pack("<I", _masked(_crc32c(lb))))
            fh.write(data)
            fh.write(struct.pack("<I", _masked(_crc32c(data))))


def read_tfrecord(path: str):
    """Yield raw serialised protobuf bytes from a TFRecord file."""
    with open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if not header:
                break
            if len(header) < 8:
                raise IOError("Truncated TFRecord: incomplete length field")
            length = struct.unpack("<Q", header)[0]
            fh.read(4)                         # skip masked_crc32(length)
            data = fh.read(length)
            if len(data) < length:
                raise IOError("Truncated TFRecord: incomplete data")
            fh.read(4)                         # skip masked_crc32(data)
            yield data


# ── High-level save / load (mirrors universal_pipeline API) ───────────────────

import json
import os
import numpy as np


def save(ds, path: str) -> None:
    """Save a Dataset to a TFRecord file (pure Python, no TF needed)."""
    records = [
        _encode_example(img.tobytes(), int(label), *img.shape)
        for img, label in zip(ds.X, ds.Y)
    ]
    write_tfrecord(path, records)

    with open(path + ".meta.json", "w") as fh:
        json.dump({
            "class_names": ds.class_names,
            "img_shape":   list(ds.img_shape),
            "num_samples": int(len(ds.X)),
        }, fh)


def load(path: str):
    """Load a TFRecord file into a Dataset (pure Python, no TF needed)."""
    # import here to avoid circular imports if used from universal_pipeline
    from universal_pipeline import Dataset

    meta_path = path + ".meta.json"
    class_names: list = []
    img_shape: tuple  = (0, 0, 3)
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        class_names = meta.get("class_names", [])
        img_shape   = tuple(meta.get("img_shape", [0, 0, 3]))

    Xs: list[np.ndarray] = []
    Ys: list[int]        = []
    for raw in read_tfrecord(path):
        img_bytes, label, h, w, c = _decode_example(raw)
        arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, c).copy()
        Xs.append(arr)
        Ys.append(label)

    X = np.array(Xs, dtype=np.uint8) if Xs else np.empty((0,) + img_shape, dtype=np.uint8)
    Y = np.array(Ys, dtype=np.int32) if Ys else np.array([], dtype=np.int32)
    return Dataset(X, Y, tuple(X.shape[1:]) if len(X) else img_shape, class_names)
