from __future__ import annotations

import json

import torch

from hybrid_golf.config import DEFAULT_CONFIG, deep_merge
from hybrid_golf.evaluate import evaluate_tokens, repo_style_strict_eval
from hybrid_golf.model import build_model
from hybrid_golf.policies import StrictCausalPolicy
from hybrid_golf.tokenizers import load_tokenizer


def write_pure_byte_tokenizer(path):
    payload = {
        "tokenizer_type": "pure_byte",
        "config": {
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "byte_offset": 4,
            "byte_count": 256,
        },
        "vocab_size": 260,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_test_config(tokenizer_path):
    return deep_merge(
        DEFAULT_CONFIG,
        {
            "run": {"device": "cpu"},
            "tokenizer": {"path": str(tokenizer_path), "kind": "pure_byte", "bos_id": 1, "eos_id": 2},
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
            "experts": {"enable_ngram": False, "enable_pointer": False, "enable_doc_bias": False},
            "adaptation": {"strict_commit_steps": 0},
            "eval": {"score_tokens": 4, "adapt_tokens": 8, "documentwise": True},
        },
    )


def test_strict_matches_repo_style_when_experts_and_adaptation_are_off(tmp_path):
    tokenizer_path = tmp_path / "pure_byte.json"
    write_pure_byte_tokenizer(tokenizer_path)
    config = build_test_config(tokenizer_path)
    model = build_model(config).to("cpu")
    tokens = torch.tensor([1, 10, 20, 30, 40, 1, 10, 20, 30, 40], dtype=torch.int64)
    tokenizer = load_tokenizer(tokenizer_path, kind="pure_byte")
    strict_result = evaluate_tokens(model, tokens, tokenizer, config, "strict_causal")
    repo_result = repo_style_strict_eval(model, tokens, tokenizer, config)
    assert abs(strict_result["val_bpb"] - repo_result["val_bpb"]) < 1e-8


def test_strict_first_block_ignores_future_tokens(tmp_path):
    tokenizer_path = tmp_path / "pure_byte.json"
    write_pure_byte_tokenizer(tokenizer_path)
    config = build_test_config(tokenizer_path)
    model = build_model(config).to("cpu")
    policy = StrictCausalPolicy(config, torch.device("cpu"))
    policy.reset_document(model)
    tokens_a = torch.tensor([1, 10, 11, 12, 13, 14, 99, 98], dtype=torch.int64)
    tokens_b = torch.tensor([1, 10, 11, 12, 13, 14, 55, 54], dtype=torch.int64)
    context_a = policy.prepare_window(tokens_a, 0, 0, 4)
    result_a = policy.score_block(model, context_a, experts=[])
    policy = StrictCausalPolicy(config, torch.device("cpu"))
    policy.reset_document(model)
    context_b = policy.prepare_window(tokens_b, 0, 0, 4)
    result_b = policy.score_block(model, context_b, experts=[])
    assert torch.allclose(result_a.logprobs, result_b.logprobs)


def test_exploratory_is_labeled_non_legal(tmp_path):
    tokenizer_path = tmp_path / "pure_byte.json"
    write_pure_byte_tokenizer(tokenizer_path)
    config = build_test_config(tokenizer_path)
    model = build_model(config).to("cpu")
    tokens = torch.tensor([1, 10, 20, 30, 40, 1, 10, 20, 30, 40], dtype=torch.int64)
    tokenizer = load_tokenizer(tokenizer_path, kind="pure_byte")
    result = evaluate_tokens(model, tokens, tokenizer, config, "exploratory_ttt")
    assert result["policy"]["legal_for_submission"] is False
