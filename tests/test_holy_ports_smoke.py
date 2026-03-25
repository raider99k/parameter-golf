import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from train_gpt_holy import (
    GPT,
    get_int6_export_params,
    compute_export_grid_regularizer_int6,
    project_model_to_int6_grid,
    build_parameter_alias_groups,
)

def test_wave_1_compression_aware_training():
    model = GPT(
        vocab_size=32, num_layers=3, model_dim=32,
        num_heads=2, num_kv_heads=2, mlp_mult=2,
        tie_embeddings=True, tied_embed_init_std=0.02,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.0,
        interface_layers=1, tail_layers=1,
        num_shared_cores=1, shared_depth=1,
    )
    
    canonical_by_name, aliases_by_canonical, reuse_count_by_canonical = build_parameter_alias_groups(model)
    export_fp16_base_names = {
        "tok_emb.weight",
        "bigram.embed.weight",
        "ve_shared.embed.weight",
    }
    
    # Check that it finds params
    params = get_int6_export_params(model, {"mlp", "attn"}, export_fp16_base_names, canonical_by_name)
    # The toy model might have numel <= 65536, so the default filter drops them.
    # We will temporarily mock the size filter inside the list comprehension if we wanted, 
    # but the simplest test is just checking that the functions don't crash with empty or real params.
    
    reg_val = compute_export_grid_regularizer_int6(model, {"mlp", "attn"}, export_fp16_base_names, canonical_by_name)
    assert isinstance(reg_val, torch.Tensor), "Regularizer should return a tensor"
    assert reg_val.ndim == 0, "Regularizer should be a scalar"
    
    # Should run without error
    project_model_to_int6_grid(model, {"mlp", "attn"}, export_fp16_base_names, canonical_by_name)
    print("Wave 1 tests passed (No exceptions)")

def test_wave_2_optimization_stability():
    model = GPT(
        vocab_size=32, num_layers=3, model_dim=32,
        num_heads=2, num_kv_heads=2, mlp_mult=2,
        tie_embeddings=True, tied_embed_init_std=0.02,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.0,
        interface_layers=1, tail_layers=1,
        num_shared_cores=1, shared_depth=1,
        normformer_lite=True,
        depth_aware_init=True,
    )
    
    # 1. Check Normformer-lite
    first_block = model.interface_blocks[0]
    assert isinstance(first_block.attn_post_norm, nn.LayerNorm) or "RMSNorm" in type(first_block.attn_post_norm).__name__
    assert isinstance(first_block.mlp_post_norm, nn.LayerNorm) or "RMSNorm" in type(first_block.mlp_post_norm).__name__
    
    # 2. Check Depth-aware init
    # Interface layer 0 (virtual depth 1)
    expected_scale_1 = (2.0 * max(1, 1)) ** -0.5
    assert torch.allclose(first_block.attn_scale, torch.full_like(first_block.attn_scale, expected_scale_1))
    
    # Core bank (virtual depth 2, since interface=1, mid shared_depth=1) -> center = 1 + (1//2) + 1 = 2
    core_block = model.core_bank[0]
    expected_scale_2 = (2.0 * max(2, 1)) ** -0.5
    assert torch.allclose(core_block.attn_scale, torch.full_like(core_block.attn_scale, expected_scale_2))
    
    # Tail block (virtual depth 3) -> interface(1) + shared(1) + 1 = 3
    tail_block = model.tail_blocks[0]
    expected_scale_3 = (2.0 * max(3, 1)) ** -0.5
    assert torch.allclose(tail_block.attn_scale, torch.full_like(tail_block.attn_scale, expected_scale_3))

    # 3. Forward pass should work
    x = torch.randint(0, 32, (2, 8))
    loss = model(x, x)
    assert loss.ndim == 0
    loss.backward()
    
    print("Wave 2 tests passed")

if __name__ == "__main__":
    test_wave_1_compression_aware_training()
    test_wave_2_optimization_stability()
    print("All smoke tests passed!")
