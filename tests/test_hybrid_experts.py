from __future__ import annotations

import torch

from hybrid_golf.experts import DocumentBiasExpert, HashedNGramExpert, RecentPointerExpert


def test_ngram_expert_prefers_repeated_successor():
    expert = HashedNGramExpert(order=2, vocab_size=32, num_buckets=128, alpha=0.1)
    expert.observe(torch.tensor([10, 20, 10, 20, 10, 20], dtype=torch.int64))
    delta = expert.logprob_delta(torch.tensor([10], dtype=torch.int64), score_offset=1, score_len=1, vocab_size=32)
    assert int(delta[0].argmax().item()) == 20


def test_pointer_expert_prefers_recent_token():
    expert = RecentPointerExpert(vocab_size=64, window=8, alpha=0.1)
    expert.observe(torch.tensor([30, 40, 50, 30], dtype=torch.int64))
    delta = expert.logprob_delta(torch.tensor([], dtype=torch.int64), score_offset=0, score_len=1, vocab_size=64)
    assert int(delta[0].argmax().item()) == 30


def test_doc_bias_expert_prefers_frequent_token():
    expert = DocumentBiasExpert(vocab_size=64, alpha=0.1)
    expert.observe(torch.tensor([7, 7, 7, 8], dtype=torch.int64))
    delta = expert.logprob_delta(torch.tensor([], dtype=torch.int64), score_offset=0, score_len=1, vocab_size=64)
    assert int(delta[0].argmax().item()) == 7
