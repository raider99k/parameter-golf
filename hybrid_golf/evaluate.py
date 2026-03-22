from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .blend import BlendDiagnostics
from .data import find_documents_by_bos, load_validation_tokens
from .experts import build_experts
from .export import load_quantized_artifact, write_quantized_artifact
from .model import HybridGPT, build_model
from .policies import build_policy
from .runtime import RunLogger, build_run_dir, resolve_autocast_dtype, resolve_device
from .tokenizers import ByteAccounting, load_tokenizer


def _loss_and_bytes(logprobs: Tensor, targets: Tensor, current_inputs: Tensor, accounting: ByteAccounting) -> tuple[float, float, int]:
    gathered = logprobs.gather(-1, targets[:, None]).squeeze(-1)
    loss_sum = float((-gathered).sum().item())
    byte_count = float(accounting.token_bytes(current_inputs, targets).to(dtype=torch.float64).sum().item())
    token_count = int(targets.numel())
    return loss_sum, byte_count, token_count


def evaluate_tokens(
    model: HybridGPT,
    tokens: Tensor,
    tokenizer: Any,
    config: dict[str, Any],
    policy_name: str,
    logger: RunLogger | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    accounting = tokenizer.build_byte_accounting(device)
    experts = build_experts(config, model.vocab_size)
    policy = build_policy(policy_name, config, device)
    bos_id = int(config["tokenizer"].get("bos_id", getattr(tokenizer, "bos_id", 1)))
    docs = find_documents_by_bos(tokens, bos_id) if bool(config["eval"]["documentwise"]) else [(0, int(tokens.numel()))]
    if not docs:
        docs = [(0, int(tokens.numel()))]

    total_loss_sum = 0.0
    total_byte_count = 0.0
    total_token_count = 0
    gate_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.eval()

    for doc_idx, (doc_offset, doc_len) in enumerate(docs):
        doc_tokens = tokens[doc_offset : doc_offset + doc_len]
        if doc_tokens.numel() <= 1:
            continue
        for expert in experts:
            expert.reset()
        policy.reset_document(model)
        pred_len = int(doc_tokens.numel() - 1)
        score_tokens = int(config["eval"]["score_tokens"])
        for block_start in range(0, pred_len, score_tokens):
            block_end = min(block_start + score_tokens, pred_len)
            context = policy.prepare_window(doc_tokens, doc_offset, block_start, block_end)
            if bool(config["eval"]["adaptive_extra_pass"]) and bool(config["model"]["recurrent_top_block"]):
                with torch.no_grad():
                    probe_logits = model.forward_logits(
                        context.x_full,
                        writable_state=policy.writable_state,
                        extra_passes=0,
                    )
                probe_logprobs = torch.log_softmax(
                    probe_logits[0, context.score_offset : context.score_offset + context.score_len],
                    dim=-1,
                )
                entropy = float((-(probe_logprobs.exp() * probe_logprobs).sum(dim=-1)).mean().item())
                disagreement = 0.0
                if experts:
                    base_top = probe_logprobs.argmax(dim=-1)
                    diff_terms = []
                    for expert in experts:
                        preview = expert.logprob_delta(context.x_full[0], context.score_offset, context.score_len, model.vocab_size).to(device)
                        diff_terms.append((preview.argmax(dim=-1) != base_top).float())
                    disagreement = float(torch.stack(diff_terms).mean().item()) if diff_terms else 0.0
                if (
                    entropy >= float(config["eval"]["adaptive_entropy_threshold"])
                    or disagreement >= float(config["eval"]["adaptive_disagreement_threshold"])
                ):
                    context.extra_passes = 1
            result = policy.score_block(model, context, experts)
            loss_sum, byte_count, token_count = _loss_and_bytes(
                result.logprobs,
                result.targets,
                result.current_inputs,
                accounting,
            )
            total_loss_sum += loss_sum
            total_byte_count += byte_count
            total_token_count += token_count
            gate_rows.append(
                {
                    "doc_index": doc_idx,
                    "score_start": block_start,
                    "score_end": block_end,
                    "mean_entropy": result.diagnostics.mean_entropy,
                    "disagreement": result.diagnostics.disagreement,
                    "mean_weights": result.diagnostics.mean_weights,
                }
            )
            if logger and bool(config["eval"]["write_block_logs"]):
                logger.event("eval_block", **gate_rows[-1])
            policy.commit_scored_tokens(model, result, experts)

    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    val_loss = total_loss_sum / max(total_token_count, 1)
    bits_per_token = val_loss / torch.log(torch.tensor(2.0)).item()
    tokens_per_byte = total_token_count / max(total_byte_count, 1.0)
    val_bpb = bits_per_token * tokens_per_byte
    summary_weights: dict[str, float] = {}
    if gate_rows:
        keys = sorted({key for row in gate_rows for key in row["mean_weights"]})
        for key in keys:
            values = [row["mean_weights"].get(key, 0.0) for row in gate_rows]
            summary_weights[key] = float(sum(values) / max(len(values), 1))
    return {
        "policy": policy.metadata(),
        "val_loss": float(val_loss),
        "val_bpb": float(val_bpb),
        "eval_time_ms": float(elapsed_ms),
        "token_count": int(total_token_count),
        "byte_count": float(total_byte_count),
        "gate_summary": summary_weights,
        "blocks": gate_rows if bool(config["eval"]["write_block_logs"]) else [],
    }


def repo_style_strict_eval(model: HybridGPT, tokens: Tensor, tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    vanilla_config = {
        **config,
        "experts": {
            **config["experts"],
            "enable_ngram": False,
            "enable_pointer": False,
            "enable_doc_bias": False,
        },
        "adaptation": {
            **config["adaptation"],
            "strict_commit_steps": 0,
        },
    }
    return evaluate_tokens(model, tokens, tokenizer, vanilla_config, policy_name="strict_causal")


def evaluate_model(model: HybridGPT, config: dict[str, Any], policy_name: str | None = None, logger: RunLogger | None = None) -> dict[str, Any]:
    tokenizer = load_tokenizer(config["tokenizer"]["path"], kind=str(config["tokenizer"]["kind"]))
    val_tokens = load_validation_tokens(config["data"]["val_glob"], limit=int(config["data"]["val_tokens_limit"]))
    return evaluate_tokens(model, val_tokens.to(device="cpu"), tokenizer, config, policy_name or str(config["eval"]["policy"]), logger=logger)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    policy_name: str,
    export_roundtrip: bool = False,
) -> dict[str, Any]:
    device = resolve_device(str(config["run"]["device"]))
    run_dir = build_run_dir(config)
    logger = RunLogger(run_dir)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    result = evaluate_model(model, config, policy_name=policy_name, logger=logger)
    logger.write_json("eval_result.json", result)
    if export_roundtrip:
        artifact_path = run_dir / str(config["export"]["artifact_name"])
        _quant_obj, export_stats = write_quantized_artifact(
            model.state_dict(),
            artifact_path,
            keep_float_max_numel=int(config["export"]["keep_float_max_numel"]),
            zlib_level=int(config["export"]["zlib_level"]),
        )
        roundtrip_model = build_model(config).to(device)
        roundtrip_model.load_state_dict(load_quantized_artifact(artifact_path), strict=True)
        roundtrip_result = evaluate_model(roundtrip_model, config, policy_name=policy_name, logger=logger)
        roundtrip_result["artifact_bytes"] = int(export_stats["artifact_bytes"])
        roundtrip_result["artifact_raw_bytes"] = int(export_stats["raw_bytes"])
        logger.write_json("eval_result_roundtrip.json", roundtrip_result)
        return {"base": result, "roundtrip": roundtrip_result}
    return result
