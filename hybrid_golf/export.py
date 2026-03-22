from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


DTYPE_CODES = {
    "torch.int8": 1,
    "torch.float16": 2,
    "torch.bfloat16": 3,
    "torch.float32": 4,
    "torch.int32": 5,
    "torch.int64": 6,
    "torch.uint8": 7,
    "torch.bool": 8,
}
INV_DTYPE_CODES = {value: key for key, value in DTYPE_CODES.items()}


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), 0.9999984, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), 0.9999984).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor], keep_float_max_numel: int) -> tuple[dict[str, Any], dict[str, int]]:
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, Any]] = {}
    stats = {"param_count": 0, "num_tensors": 0, "baseline_tensor_bytes": 0, "payload_bytes": 0}
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            passthrough[name] = t
            stats["payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= keep_float_max_numel:
            kept = t.to(dtype=torch.float16 if t.dtype in {torch.float32, torch.bfloat16} else t.dtype).contiguous()
            passthrough[name] = kept
            if kept.dtype != t.dtype:
                passthrough_orig_dtypes[name] = str(t.dtype)
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue
        q, s = quantize_float_tensor(t)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, Any] = {
        "__quant_format__": "hybrid_int8_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if qmeta:
        obj["qmeta"] = qmeta
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, Any]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    qmeta = obj.get("qmeta", {})
    for name, q in obj["quantized"].items():
        scale = obj["scales"][name]
        dtype = getattr(torch, obj["dtypes"][name])
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            scale = scale.to(dtype=torch.float32)
            out[name] = (q.float() * scale.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(scale.item())).to(dtype=dtype).contiguous()
    for name, tensor in obj["passthrough"].items():
        restored = tensor.detach().to("cpu").contiguous()
        if name in passthrough_orig_dtypes:
            dtype_name = passthrough_orig_dtypes[name].removeprefix("torch.")
            restored = restored.to(dtype=getattr(torch, dtype_name))
        out[name] = restored
    return out


def _dtype_code(dtype_str: str) -> int:
    if not dtype_str.startswith("torch."):
        dtype_str = "torch." + dtype_str
    return DTYPE_CODES[dtype_str]


def _read_tensor(dtype_str: str, shape: tuple[int, ...], payload: bytes) -> Tensor:
    if dtype_str == "torch.bfloat16":
        return torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).clone().view(shape)
    if dtype_str == "torch.bool":
        return torch.frombuffer(bytearray(payload), dtype=torch.bool).clone().view(shape)
    np_dtype = getattr(np, dtype_str.removeprefix("torch."))
    return torch.from_numpy(np.frombuffer(payload, dtype=np_dtype).copy()).view(shape)


def pack_quantized(quant_obj: dict[str, Any]) -> bytes:
    buf = bytearray()
    buf.extend(b"HGPQ")
    buf.extend(struct.pack("<H", 1))
    names = sorted(set(quant_obj["quantized"].keys()) | set(quant_obj["passthrough"].keys()))
    buf.extend(struct.pack("<H", len(names)))
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name in names:
        name_bytes = name.encode("utf-8")
        buf.append(len(name_bytes))
        buf.extend(name_bytes)
        if name in quant_obj["quantized"]:
            q = quant_obj["quantized"][name]
            s = quant_obj["scales"][name]
            scheme_code = 2 if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0 else 1
            orig_dtype = quant_obj["dtypes"][name]
            buf.append(scheme_code)
            buf.append(_dtype_code(orig_dtype))
            buf.append(_dtype_code(str(q.dtype)))
            buf.append(q.ndim)
            for dim in q.shape:
                buf.extend(struct.pack("<I", dim))
            if scheme_code == 2:
                buf.append(_dtype_code(str(s.dtype)))
                scale_bytes = s.cpu().numpy().tobytes()
                buf.extend(struct.pack("<I", len(scale_bytes)))
                buf.extend(scale_bytes)
            else:
                buf.extend(struct.pack("<f", float(s.item())))
            q_bytes = q.cpu().numpy().tobytes()
            buf.extend(struct.pack("<I", len(q_bytes)))
            buf.extend(q_bytes)
        else:
            tensor = quant_obj["passthrough"][name]
            buf.append(10 if tensor.is_floating_point() else 11)
            buf.append(_dtype_code(passthrough_orig_dtypes.get(name, str(tensor.dtype))))
            buf.append(_dtype_code(str(tensor.dtype)))
            buf.append(tensor.ndim)
            for dim in tensor.shape:
                buf.extend(struct.pack("<I", dim))
            data = tensor.cpu().numpy().tobytes()
            buf.extend(struct.pack("<I", len(data)))
            buf.extend(data)
    return bytes(buf)


def unpack_quantized(data: bytes) -> dict[str, Any]:
    if data[:4] != b"HGPQ":
        raise ValueError("Unsupported quantized payload header")
    version, = struct.unpack_from("<H", data, 4)
    if version != 1:
        raise ValueError(f"Unsupported quantized payload version: {version}")
    offset = 6
    tensor_count, = struct.unpack_from("<H", data, offset)
    offset += 2
    out: dict[str, Any] = {
        "__quant_format__": "hybrid_int8_per_row_v1",
        "quantized": {},
        "scales": {},
        "dtypes": {},
        "passthrough": {},
        "passthrough_orig_dtypes": {},
        "qmeta": {},
    }
    for _ in range(tensor_count):
        name_len = data[offset]
        offset += 1
        name = data[offset : offset + name_len].decode("utf-8")
        offset += name_len
        scheme_code = data[offset]
        offset += 1
        orig_dtype = INV_DTYPE_CODES[data[offset]]
        offset += 1
        payload_dtype = INV_DTYPE_CODES[data[offset]]
        offset += 1
        ndim = data[offset]
        offset += 1
        shape = []
        for _idx in range(ndim):
            dim, = struct.unpack_from("<I", data, offset)
            offset += 4
            shape.append(dim)
        shape_tuple = tuple(shape)
        if scheme_code in {1, 2}:
            if scheme_code == 2:
                scale_dtype = INV_DTYPE_CODES[data[offset]]
                offset += 1
                scale_len, = struct.unpack_from("<I", data, offset)
                offset += 4
                if len(shape_tuple) < 1:
                    raise ValueError("Per-row quantization requires at least 1 dimension")
                scale = _read_tensor(scale_dtype, (shape_tuple[0],), data[offset : offset + scale_len])
                offset += scale_len
                out["qmeta"][name] = {"scheme": "per_row", "axis": 0}
            else:
                scale_value, = struct.unpack_from("<f", data, offset)
                offset += 4
                scale = torch.tensor(scale_value, dtype=torch.float32)
            q_len, = struct.unpack_from("<I", data, offset)
            offset += 4
            q = _read_tensor(payload_dtype, shape_tuple, data[offset : offset + q_len])
            offset += q_len
            out["quantized"][name] = q
            out["scales"][name] = scale
            out["dtypes"][name] = orig_dtype.removeprefix("torch.")
        elif scheme_code in {10, 11}:
            payload_len, = struct.unpack_from("<I", data, offset)
            offset += 4
            tensor = _read_tensor(payload_dtype, shape_tuple, data[offset : offset + payload_len])
            offset += payload_len
            out["passthrough"][name] = tensor
            if orig_dtype != payload_dtype:
                out["passthrough_orig_dtypes"][name] = orig_dtype
        else:
            raise ValueError(f"Unsupported scheme code: {scheme_code}")
    return out


def write_quantized_artifact(
    state_dict: dict[str, Tensor],
    artifact_path: str | Path,
    keep_float_max_numel: int,
    zlib_level: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    quant_obj, stats = quantize_state_dict_int8(state_dict, keep_float_max_numel=keep_float_max_numel)
    payload = pack_quantized(quant_obj)
    compressed = zlib.compress(payload, level=zlib_level)
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    stats["artifact_bytes"] = path.stat().st_size
    stats["raw_bytes"] = len(payload)
    return quant_obj, stats


def load_quantized_artifact(path: str | Path) -> dict[str, Tensor]:
    payload = zlib.decompress(Path(path).read_bytes())
    quant_obj = unpack_quantized(payload)
    return dequantize_state_dict_int8(quant_obj)
