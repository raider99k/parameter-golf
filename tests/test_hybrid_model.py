from __future__ import annotations

import torch
import torch.nn.functional as F

from hybrid_golf.config import DEFAULT_CONFIG, deep_merge
from hybrid_golf.model import BitLinear, CastedLinear, build_model, depth_aware_branch_scale


def build_test_config(**model_overrides):
    return deep_merge(
        DEFAULT_CONFIG,
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
                "writable_rank": 2,
                "writable_blocks": 1,
                **model_overrides,
            },
        },
    )


def test_shared_banks_avoid_duplicate_block_parameters():
    config = build_test_config(num_unique_attn=1, num_unique_mlp=2, normformer_lite=True, depth_aware_init=True)
    model = build_model(config)
    state_keys = list(model.state_dict().keys())
    assert len(model.attn_bank) == 1
    assert len(model.mlp_bank) == 2
    assert all(".attn." not in key for key in state_keys if key.startswith("blocks."))
    assert all(".mlp." not in key for key in state_keys if key.startswith("blocks."))
    assert torch.allclose(model.blocks[0].attn_scale, torch.full_like(model.blocks[0].attn_scale, depth_aware_branch_scale(3)))


def test_unique_bank_counts_match_layer_count_and_factorized_embed_runs():
    config = build_test_config(num_unique_attn=3, num_unique_mlp=3)
    model = build_model(config).to("cpu")
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    logits = model.forward_logits(x)
    assert len(model.attn_bank) == 3
    assert len(model.mlp_bank) == 3
    assert model.tok_emb.weight.shape == (260, 16)
    assert model.emb_proj is not None
    assert logits.shape == (1, 4, 260)


def test_recurrent_single_pass_matches_internal_stack_path():
    torch.manual_seed(0)
    config = build_test_config(recurrent_passes=1, pass_modulation=False, pass_q_gain=False, low_rank_deltas=False)
    model = build_model(config).to("cpu")
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    hidden = model.forward_hidden(x)
    embedded = model.tok_emb(x)
    if model.emb_proj is not None:
        embedded = model.emb_proj(embedded)
    embedded = F.rms_norm(embedded, (embedded.size(-1),))
    internal = model._run_transformer_stack(embedded, pass_idx=0)
    assert torch.allclose(hidden, internal)


def test_recurrent_passes_and_bitlinear_targets_wire_correct_modules():
    recurrent = build_model(build_test_config(recurrent_passes=2, recurrent_gates=True)).to("cpu")
    bitlinear_mlp = build_model(
        build_test_config(linear_impl="bitlinear", bitlinear_targets="mlp_only", bitlinear_group_size=64)
    ).to("cpu")
    bitlinear_full = build_model(
        build_test_config(linear_impl="bitlinear", bitlinear_targets="attn_and_mlp", bitlinear_group_size=64)
    ).to("cpu")
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    assert recurrent.pass_gates.shape == (2, recurrent.model_dim)
    assert recurrent.forward_logits(x).shape == (1, 4, 260)
    assert isinstance(bitlinear_mlp.mlp_bank[0].fc, BitLinear)
    assert isinstance(bitlinear_mlp.attn_bank[0].c_q, CastedLinear)
    assert isinstance(bitlinear_full.mlp_bank[0].fc, BitLinear)
    assert isinstance(bitlinear_full.attn_bank[0].c_q, BitLinear)
