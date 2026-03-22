from __future__ import annotations

import torch

from hybrid_golf.config import DEFAULT_CONFIG, deep_merge
from hybrid_golf.export import load_quantized_artifact, write_quantized_artifact
from hybrid_golf.model import build_model


def test_export_roundtrip_preserves_logits_shape(tmp_path):
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "run": {"device": "cpu"},
            "model": {
                "vocab_size": 260,
                "num_layers": 2,
                "model_dim": 32,
                "num_heads": 4,
                "num_kv_heads": 2,
                "mlp_mult": 2,
                "tie_embeddings": True,
                "use_factor_embed": False,
                "embed_dim": 32,
                "writable_rank": 2,
                "writable_blocks": 1,
            },
        },
    )
    model = build_model(config).to("cpu")
    x = torch.tensor([[1, 10, 20, 30]], dtype=torch.int64)
    logits_before = model.forward_logits(x)
    artifact_path = tmp_path / "model.int8.ptz"
    _quant_obj, stats = write_quantized_artifact(model.state_dict(), artifact_path, keep_float_max_numel=128, zlib_level=9)
    roundtrip_model = build_model(config).to("cpu")
    roundtrip_model.load_state_dict(load_quantized_artifact(artifact_path), strict=True)
    logits_after = roundtrip_model.forward_logits(x)
    assert stats["artifact_bytes"] > 0
    assert logits_before.shape == logits_after.shape
