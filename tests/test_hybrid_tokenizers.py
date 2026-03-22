from __future__ import annotations

import json

import pytest
import torch

from hybrid_golf.tokenizers import PureByteTokenizer, SentencePieceTokenizer


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


def test_pure_byte_tokenizer_counts_bytes(tmp_path):
    path = tmp_path / "pure_byte.json"
    write_pure_byte_tokenizer(path)
    tok = PureByteTokenizer.from_json(path)
    accounting = tok.build_byte_accounting(torch.device("cpu"))
    prev = torch.tensor([tok.bos_id, 70], dtype=torch.int64)
    tgt = torch.tensor([tok.byte_offset + ord("A"), tok.byte_offset + ord("B")], dtype=torch.int64)
    bytes_out = accounting.token_bytes(prev, tgt)
    assert bytes_out.tolist() == [1, 1]
    assert tok.encode("AB") == [tok.byte_offset + 65, tok.byte_offset + 66]


def test_sentencepiece_tokenizer_loads_and_builds_accounting(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nhello again\n", encoding="utf-8")
    prefix = tmp_path / "mini_sp"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=32,
        character_coverage=1.0,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="nmt_nfkc",
        add_dummy_prefix=False,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
    )
    tok = SentencePieceTokenizer(prefix.with_suffix(".model"))
    accounting = tok.build_byte_accounting(torch.device("cpu"))
    encoded = tok.encode("hello world")
    assert len(encoded) > 0
    assert accounting.base_bytes.shape[0] == tok.vocab_size
