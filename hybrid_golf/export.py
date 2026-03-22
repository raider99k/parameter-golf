from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


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
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "q_gain",
    "gate",
    "pass_",
    "attn_mod_",
    "mlp_mod_",
    "delta",
    "log_lr",
    "logit_decay",
    "norm.weight",
)


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def name_matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in name for pattern in patterns)


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


def quantize_intn_per_row(t: Tensor, clip_range: int) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / clip_range).clamp_min(1e-12).to(torch.float16).contiguous()
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -(clip_range + 1), clip_range).to(torch.int8)
        return q.contiguous(), scale
    clip_abs = float(t32.abs().max().item()) if t32.numel() else 0.0
    scale = torch.tensor(max(clip_abs / clip_range, 1e-12), dtype=torch.float32)
    q = torch.clamp(torch.round(t32 / scale), -(clip_range + 1), clip_range).to(torch.int8).contiguous()
    return q, scale


def pack_ternary(ternary_tensor: Tensor) -> Tensor:
    shifted = ternary_tensor.flatten().to(torch.int16) + 1
    pad_len = (5 - (shifted.numel() % 5)) % 5
    if pad_len > 0:
        shifted = torch.cat([shifted, torch.zeros(pad_len, device=shifted.device, dtype=shifted.dtype)])
    shifted = shifted.view(-1, 5).to(torch.uint8)
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.uint8, device=shifted.device)
    packed = (shifted * powers).sum(dim=1, dtype=torch.uint8)
    return packed.cpu().contiguous()


def unpack_ternary(packed: Tensor, original_shape: tuple[int, ...], device: torch.device | str) -> Tensor:
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=device)
    unpacked = (packed.to(device).unsqueeze(1).to(torch.int32) // powers) % 3
    unpacked = unpacked.flatten() - 1
    numel = math.prod(original_shape)
    return unpacked[:numel].view(original_shape).to(torch.int8)


def quantize_ternary_latent_tensor(t: Tensor, group_size: int = 0) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim != 2:
        raise ValueError("Ternary latent quantization expects a rank-2 tensor")
    if group_size <= 0 or group_size >= t32.shape[1]:
        scale = t32.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        ternary = torch.round(t32 / scale).clamp(-1, 1).to(torch.int8)
        return ternary, scale.to(dtype=torch.float16).contiguous()

    ternary_chunks: list[Tensor] = []
    scale_chunks: list[Tensor] = []
    for start in range(0, t32.shape[1], group_size):
        end = min(start + group_size, t32.shape[1])
        chunk = t32[:, start:end]
        scale = chunk.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        ternary_chunks.append(torch.round(chunk / scale).clamp(-1, 1).to(torch.int8))
        scale_chunks.append(scale.to(dtype=torch.float16))
    return torch.cat(ternary_chunks, dim=1).contiguous(), torch.cat(scale_chunks, dim=1).contiguous()


def apply_groupwise_row_scales(t: Tensor, scales: Tensor, group_size: int) -> Tensor:
    if scales.ndim == 1:
        return t * scales.to(dtype=t.dtype)[:, None]
    if scales.ndim != 2 or scales.shape[1] == 1 or group_size <= 0:
        return t * scales.to(dtype=t.dtype)
    out = torch.empty_like(t)
    group_idx = 0
    for start in range(0, t.shape[1], group_size):
        end = min(start + group_size, t.shape[1])
        out[:, start:end] = t[:, start:end] * scales[:, group_idx : group_idx + 1].to(dtype=t.dtype)
        group_idx += 1
    return out


def dequantize_tensor(q: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    if scale.ndim > 0:
        viewed = scale.view(q.shape[0], *([1] * (q.ndim - 1))).float()
        return (q.float() * viewed).to(dtype=dtype).contiguous()
    return (q.float() * float(scale.item())).to(dtype=dtype).contiguous()


def _classify_param(name: str) -> str:
    if name.startswith("attn_bank."):
        return "attn"
    if name.startswith("mlp_bank."):
        return "mlp"
    if name.startswith("tok_emb.") or name.startswith("emb_proj.") or name.startswith("lm_head."):
        return "embed"
    if name.startswith("fast_blocks.") or name.startswith("out_fast."):
        return "adapter"
    return "other"


def should_keep_float_tensor(name: str, t: Tensor, keep_float_max_numel: int, keep_float_policy: str) -> bool:
    policy = str(keep_float_policy).strip().lower()
    is_small = t.numel() <= keep_float_max_numel
    is_control = t.ndim < 2 or name_matches_any(name, CONTROL_TENSOR_NAME_PATTERNS)
    if policy == "none":
        return False
    if policy == "small_only":
        return is_small
    if policy == "control_only":
        return is_control
    if policy == "small_and_control":
        return is_small or is_control
    raise ValueError(f"Unsupported keep_float_policy: {keep_float_policy!r}")


def select_quant_bits(name: str, t: Tensor, quant_scheme: str) -> int:
    scheme = str(quant_scheme).strip().lower()
    if scheme == "int8_v1":
        return 8
    if scheme not in {"mixed_v1", "mixed_v2"}:
        raise ValueError(f"Unsupported quant_scheme: {quant_scheme!r}")
    if t.ndim < 2:
        return 8
    category = _classify_param(name)
    if category == "attn":
        return 6
    if category == "mlp":
        return 5
    return 8


def quantize_named_float_tensor(
    name: str,
    t: Tensor,
    quant_scheme: str,
    bitlinear_group_size: int = 0,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    scheme = str(quant_scheme).strip().lower()
    if scheme == "mixed_v2" and name.endswith(".weight_latent"):
        group_size = max(int(bitlinear_group_size), 0)
        qmeta = {
            "scheme": "ternary_packed",
            "shape": list(t.shape),
            "group_size": group_size,
        }
        ternary, scale = quantize_ternary_latent_tensor(t, group_size=group_size)
        packed = pack_ternary(ternary)
        if scale.ndim == 2 and scale.shape[1] > 1:
            qmeta["scheme"] = "ternary_packed_group"
            qmeta["group_size"] = max(int(t.shape[1] // scale.shape[1]), 1)
        return packed, scale, qmeta
    quant_bits = select_quant_bits(name, t, quant_scheme)
    if quant_bits == 8:
        q, scale = quantize_float_tensor(t)
    elif quant_bits == 6:
        q, scale = quantize_intn_per_row(t, clip_range=31)
    elif quant_bits == 5:
        q, scale = quantize_intn_per_row(t, clip_range=15)
    else:
        raise ValueError(f"Unsupported quant_bits: {quant_bits}")
    meta = {
        "scheme": "per_row" if scale.ndim > 0 else "per_tensor",
        "quant_bits": quant_bits,
    }
    if scale.ndim > 0:
        meta["axis"] = 0
    return q, scale, meta


def dequantize_named_tensor(q: Tensor, scale: Tensor, dtype: torch.dtype, meta: dict[str, Any]) -> Tensor:
    scheme = str(meta.get("scheme", "per_tensor")).strip().lower()
    if scheme in {"ternary_packed", "ternary_packed_group"}:
        shape = tuple(int(dim) for dim in meta["shape"])
        group_size = int(meta.get("group_size", 0))
        ternary = unpack_ternary(q, shape, "cpu").float()
        restored = apply_groupwise_row_scales(ternary, scale.float(), group_size)
        return restored.to(dtype=dtype).contiguous()
    return dequantize_tensor(q, scale, dtype=dtype)


def quantize_state_dict(
    state_dict: dict[str, Tensor],
    keep_float_max_numel: int,
    quant_scheme: str,
    keep_float_policy: str,
    bitlinear_group_size: int = 0,
) -> tuple[dict[str, Any], dict[str, int]]:
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, Any]] = {}
    stats = {
        "param_count": 0,
        "num_tensors": 0,
        "num_quantized_tensors": 0,
        "baseline_tensor_bytes": 0,
        "payload_bytes": 0,
    }
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            passthrough[name] = t
            stats["payload_bytes"] += tensor_nbytes(t)
            continue
        if should_keep_float_tensor(name, t, keep_float_max_numel, keep_float_policy):
            kept = t.to(dtype=torch.float16 if t.dtype in {torch.float32, torch.bfloat16} else t.dtype).contiguous()
            passthrough[name] = kept
            if kept.dtype != t.dtype:
                passthrough_orig_dtypes[name] = str(t.dtype)
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue
        q, scale, meta = quantize_named_float_tensor(name, t, quant_scheme, bitlinear_group_size=bitlinear_group_size)
        quantized[name] = q
        scales[name] = scale
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        qmeta[name] = meta
        stats["num_quantized_tensors"] += 1
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(scale)
    obj: dict[str, Any] = {
        "__quant_format__": "hybrid_mixed_v2",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict(obj: dict[str, Any]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    qmeta = obj.get("qmeta", {})
    for name, q in obj["quantized"].items():
        scale = obj["scales"][name]
        dtype = getattr(torch, obj["dtypes"][name])
        out[name] = dequantize_named_tensor(q, scale, dtype=dtype, meta=qmeta.get(name, {}))
    for name, tensor in obj["passthrough"].items():
        restored = tensor.detach().to("cpu").contiguous()
        if name in passthrough_orig_dtypes:
            restored = restored.to(dtype=getattr(torch, passthrough_orig_dtypes[name].removeprefix("torch.")))
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
    buf.extend(struct.pack("<H", 2))
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
            scale = quant_obj["scales"][name]
            meta = qmeta.get(name, {})
            scheme_name = str(meta.get("scheme", "per_tensor")).strip().lower()
            if scheme_name == "per_tensor":
                scheme_code = 1
            elif scheme_name == "per_row":
                scheme_code = 2
            elif scheme_name == "ternary_packed":
                scheme_code = 3
            elif scheme_name == "ternary_packed_group":
                scheme_code = 4
            else:
                raise ValueError(f"Unsupported quantized tensor scheme: {scheme_name!r}")
            buf.append(scheme_code)
            buf.append(_dtype_code(quant_obj["dtypes"][name]))
            buf.append(_dtype_code(str(q.dtype)))
            tensor_shape = tuple(meta.get("shape", list(q.shape))) if scheme_code in {3, 4} else q.shape
            buf.append(len(tensor_shape))
            for dim in tensor_shape:
                buf.extend(struct.pack("<I", dim))
            buf.append(int(meta.get("quant_bits", 3 if scheme_code in {3, 4} else 8)))
            if scheme_code in {2, 3, 4}:
                buf.append(_dtype_code(str(scale.dtype)))
                if scheme_code in {3, 4}:
                    buf.extend(struct.pack("<I", int(meta.get("group_size", 0))))
                    buf.append(scale.ndim)
                    for dim in scale.shape:
                        buf.extend(struct.pack("<I", int(dim)))
                scale_bytes = scale.cpu().numpy().tobytes()
                buf.extend(struct.pack("<I", len(scale_bytes)))
                buf.extend(scale_bytes)
            else:
                buf.extend(struct.pack("<f", float(scale.item())))
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


def _unpack_quantized_v1(data: bytes) -> dict[str, Any]:
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
                scale = _read_tensor(scale_dtype, (shape_tuple[0],), data[offset : offset + scale_len])
                offset += scale_len
                out["qmeta"][name] = {"scheme": "per_row", "axis": 0, "quant_bits": 8}
            else:
                scale_value, = struct.unpack_from("<f", data, offset)
                offset += 4
                scale = torch.tensor(scale_value, dtype=torch.float32)
                out["qmeta"][name] = {"scheme": "per_tensor", "quant_bits": 8}
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


def unpack_quantized(data: bytes) -> dict[str, Any]:
    if data[:4] != b"HGPQ":
        raise ValueError("Unsupported quantized payload header")
    version, = struct.unpack_from("<H", data, 4)
    if version == 1:
        return _unpack_quantized_v1(data)
    if version != 2:
        raise ValueError(f"Unsupported quantized payload version: {version}")
    offset = 6
    tensor_count, = struct.unpack_from("<H", data, offset)
    offset += 2
    out: dict[str, Any] = {
        "__quant_format__": "hybrid_mixed_v2",
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
            quant_bits = int(data[offset])
            offset += 1
            if scheme_code == 2:
                scale_dtype = INV_DTYPE_CODES[data[offset]]
                offset += 1
                scale_len, = struct.unpack_from("<I", data, offset)
                offset += 4
                scale = _read_tensor(scale_dtype, (shape_tuple[0],), data[offset : offset + scale_len])
                offset += scale_len
                out["qmeta"][name] = {"scheme": "per_row", "axis": 0, "quant_bits": quant_bits}
            else:
                scale_value, = struct.unpack_from("<f", data, offset)
                offset += 4
                scale = torch.tensor(scale_value, dtype=torch.float32)
                out["qmeta"][name] = {"scheme": "per_tensor", "quant_bits": quant_bits}
            q_len, = struct.unpack_from("<I", data, offset)
            offset += 4
            q = _read_tensor(payload_dtype, shape_tuple, data[offset : offset + q_len])
            offset += q_len
            out["quantized"][name] = q
            out["scales"][name] = scale
            out["dtypes"][name] = orig_dtype.removeprefix("torch.")
        elif scheme_code in {3, 4}:
            quant_bits = int(data[offset])
            offset += 1
            scale_dtype = INV_DTYPE_CODES[data[offset]]
            offset += 1
            group_size, = struct.unpack_from("<I", data, offset)
            offset += 4
            scale_ndim = data[offset]
            offset += 1
            scale_shape = []
            for _idx in range(scale_ndim):
                dim, = struct.unpack_from("<I", data, offset)
                offset += 4
                scale_shape.append(dim)
            scale_len, = struct.unpack_from("<I", data, offset)
            offset += 4
            scale = _read_tensor(scale_dtype, tuple(scale_shape), data[offset : offset + scale_len])
            offset += scale_len
            q_len, = struct.unpack_from("<I", data, offset)
            offset += 4
            q = _read_tensor(payload_dtype, (q_len,), data[offset : offset + q_len])
            offset += q_len
            out["quantized"][name] = q
            out["scales"][name] = scale
            out["dtypes"][name] = orig_dtype.removeprefix("torch.")
            out["qmeta"][name] = {
                "scheme": "ternary_packed_group" if scheme_code == 4 else "ternary_packed",
                "shape": list(shape_tuple),
                "group_size": int(group_size),
                "quant_bits": quant_bits,
            }
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
    quant_scheme: str = "mixed_v1",
    keep_float_policy: str = "small_and_control",
    bitlinear_group_size: int = 0,
) -> tuple[dict[str, Any], dict[str, int]]:
    quant_obj, stats = quantize_state_dict(
        state_dict,
        keep_float_max_numel=keep_float_max_numel,
        quant_scheme=quant_scheme,
        keep_float_policy=keep_float_policy,
        bitlinear_group_size=bitlinear_group_size,
    )
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
    return dequantize_state_dict(quant_obj)


def get_export_project_params(
    model: nn.Module,
    *,
    quant_scheme: str,
    keep_float_max_numel: int,
    keep_float_policy: str,
    bitlinear_group_size: int = 0,
) -> tuple[tuple[str, Tensor], ...]:
    params: list[tuple[str, Tensor]] = []
    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        if should_keep_float_tensor(name, param.detach(), keep_float_max_numel, keep_float_policy):
            continue
        params.append((name, param))
    return tuple(params)


@torch.no_grad()
def project_model_to_export_grid(
    model: nn.Module,
    *,
    quant_scheme: str,
    keep_float_max_numel: int,
    keep_float_policy: str,
    bitlinear_group_size: int = 0,
) -> None:
    for name, param in get_export_project_params(
        model,
        quant_scheme=quant_scheme,
        keep_float_max_numel=keep_float_max_numel,
        keep_float_policy=keep_float_policy,
        bitlinear_group_size=bitlinear_group_size,
    ):
        q, scale, _meta = quantize_named_float_tensor(
            name,
            param.data,
            quant_scheme,
            bitlinear_group_size=bitlinear_group_size,
        )
        param.data.copy_(dequantize_named_tensor(q, scale, dtype=param.dtype, meta=_meta))


def compute_export_grid_regularizer(
    model: nn.Module,
    *,
    quant_scheme: str,
    keep_float_max_numel: int,
    keep_float_policy: str,
    bitlinear_group_size: int = 0,
) -> Tensor:
    penalties: list[Tensor] = []
    for name, param in get_export_project_params(
        model,
        quant_scheme=quant_scheme,
        keep_float_max_numel=keep_float_max_numel,
        keep_float_policy=keep_float_policy,
        bitlinear_group_size=bitlinear_group_size,
    ):
        q, scale, _meta = quantize_named_float_tensor(
            name,
            param.detach(),
            quant_scheme,
            bitlinear_group_size=bitlinear_group_size,
        )
        target = dequantize_named_tensor(q, scale, dtype=param.dtype, meta=_meta)
        penalties.append((param.float() - target.float()).square().mean())
    if not penalties:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(penalties).mean()
