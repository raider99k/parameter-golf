from __future__ import annotations

import zlib

import torch

from hybrid_golf.config import DEFAULT_CONFIG, deep_merge
from hybrid_golf.export import load_quantized_artifact, unpack_quantized, write_quantized_artifact
from hybrid_golf.model import build_model
from hybrid_golf.runtime import build_submission_size_metrics


def build_export_config(**overrides):
    return deep_merge(
        DEFAULT_CONFIG,
        deep_merge(
            {
            "run": {"device": "cpu"},
            "model": {
                "vocab_size": 260,
                "num_layers": 3,
                "model_dim": 32,
                "num_heads": 4,
                "num_kv_heads": 2,
                "mlp_mult": 2,
                "tie_embeddings": True,
                "use_factor_embed": True,
                "embed_dim": 16,
                "num_unique_attn": 2,
                "num_unique_mlp": 2,
                "normformer_lite": True,
                "depth_aware_init": True,
                "writable_rank": 2,
                "writable_blocks": 1,
            },
            "export": {
                "keep_float_max_numel": 4,
                "quant_scheme": "mixed_v1",
                "keep_float_policy": "small_and_control",
            },
            },
            overrides,
        ),
    )


def test_export_roundtrip_preserves_logits_shape_and_uses_mixed_bits(tmp_path):
    config = build_export_config()
    model = build_model(config).to("cpu")
    x = torch.tensor([[1, 10, 20, 30]], dtype=torch.int64)
    logits_before = model.forward_logits(x)
    artifact_path = tmp_path / "model.int8.ptz"
    _quant_obj, stats = write_quantized_artifact(
        model.state_dict(),
        artifact_path,
        keep_float_max_numel=int(config["export"]["keep_float_max_numel"]),
        zlib_level=9,
        quant_scheme=str(config["export"]["quant_scheme"]),
        keep_float_policy=str(config["export"]["keep_float_policy"]),
        bitlinear_group_size=int(config["model"]["bitlinear_group_size"]),
    )
    unpacked = unpack_quantized(zlib.decompress(artifact_path.read_bytes()))
    quant_bits = {meta["quant_bits"] for meta in unpacked["qmeta"].values()}
    roundtrip_model = build_model(config).to("cpu")
    roundtrip_model.load_state_dict(load_quantized_artifact(artifact_path), strict=True)
    logits_after = roundtrip_model.forward_logits(x)
    assert stats["artifact_bytes"] > 0
    assert 5 in quant_bits
    assert 6 in quant_bits
    assert logits_before.shape == logits_after.shape


def test_mixed_v2_roundtrip_preserves_bitlinear_latents(tmp_path):
    config = build_export_config(
        model={
            "linear_impl": "bitlinear",
            "bitlinear_targets": "mlp_only",
            "bitlinear_group_size": 64,
        },
        export={"quant_scheme": "mixed_v2"},
    )
    model = build_model(config).to("cpu")
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    logits_before = model.forward_logits(x)
    artifact_path = tmp_path / "model.mixed_v2.ptz"
    _quant_obj, _stats = write_quantized_artifact(
        model.state_dict(),
        artifact_path,
        keep_float_max_numel=int(config["export"]["keep_float_max_numel"]),
        zlib_level=9,
        quant_scheme=str(config["export"]["quant_scheme"]),
        keep_float_policy=str(config["export"]["keep_float_policy"]),
        bitlinear_group_size=int(config["model"]["bitlinear_group_size"]),
    )
    unpacked = unpack_quantized(zlib.decompress(artifact_path.read_bytes()))
    ternary_names = [name for name, meta in unpacked["qmeta"].items() if meta["scheme"].startswith("ternary_packed")]
    roundtrip_model = build_model(config).to("cpu")
    roundtrip_model.load_state_dict(load_quantized_artifact(artifact_path), strict=True)
    logits_after = roundtrip_model.forward_logits(x)
    assert ternary_names
    assert logits_after.shape == (1, 4, 260)
    assert (logits_before - logits_after).abs().max().item() < 0.05


def test_mixed_v2_preserves_group_size_for_non_divisible_bitlinear_widths(tmp_path):
    config = build_export_config(
        model={
            "model_dim": 72,
            "num_heads": 4,
            "num_kv_heads": 2,
            "linear_impl": "bitlinear",
            "bitlinear_targets": "mlp_only",
            "bitlinear_group_size": 64,
        },
        export={"quant_scheme": "mixed_v2"},
    )
    model = build_model(config).to("cpu")
    artifact_path = tmp_path / "model_nondivisible_group.mixed_v2.ptz"
    _quant_obj, _stats = write_quantized_artifact(
        model.state_dict(),
        artifact_path,
        keep_float_max_numel=int(config["export"]["keep_float_max_numel"]),
        zlib_level=9,
        quant_scheme=str(config["export"]["quant_scheme"]),
        keep_float_policy=str(config["export"]["keep_float_policy"]),
        bitlinear_group_size=int(config["model"]["bitlinear_group_size"]),
    )
    unpacked = unpack_quantized(zlib.decompress(artifact_path.read_bytes()))
    ternary_group_sizes = {
        meta["group_size"]
        for meta in unpacked["qmeta"].values()
        if meta["scheme"] == "ternary_packed_group"
    }
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    logits_before = model.forward_logits(x)
    roundtrip_model = build_model(config).to("cpu")
    roundtrip_model.load_state_dict(load_quantized_artifact(artifact_path), strict=True)
    logits_after = roundtrip_model.forward_logits(x)
    assert ternary_group_sizes == {64}
    assert (logits_before - logits_after).abs().max().item() < 0.05


def test_submission_total_budget_metrics_include_code_bytes():
    metrics = build_submission_size_metrics(artifact_bytes=1024, budget_bytes=1024, budget_mode="artifact_only")
    assert metrics["model_artifact_bytes"] == 1024
    assert metrics["submission_total_bytes"] >= 1024
    tight = build_submission_size_metrics(
        artifact_bytes=1024,
        budget_bytes=1023 + metrics["counted_code_bytes"],
        budget_mode="submission_total",
    )
    assert tight["counted_code_bytes"] > 0
    assert tight["submission_total_bytes"] == 1024 + tight["counted_code_bytes"]
    assert tight["over_budget"] is True
