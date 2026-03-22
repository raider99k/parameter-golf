from __future__ import annotations

import time

import torch

from .adaptation import adapt_writable_state, init_writable_state
from .data import TokenBatcher
from .evaluate import evaluate_model
from .export import compute_export_grid_regularizer, project_model_to_export_grid, write_quantized_artifact
from .model import build_model
from .optim import Muon
from .runtime import (
    RunLogger,
    build_run_dir,
    build_submission_size_metrics,
    resolve_autocast_dtype,
    resolve_device,
    seed_everything,
)


def _lr_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min((step + 1) / warmup_steps, 1.0)


def _progress_frac(step: int, total_steps: int) -> float:
    return min((step + 1) / max(total_steps, 1), 1.0)


def _should_write_artifact_after_train(config: dict[str, object]) -> bool:
    if bool(config["export"]["write_after_train"]):
        return True
    return str(config["run"].get("stage", "smoke")).strip().lower() != "smoke"


def _zero_grad_all(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def _step_all(optimizers: list[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.step()


def _set_lr_scale(optimizers: list[torch.optim.Optimizer], scale: float) -> None:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * scale


def _build_optimizers(model: torch.nn.Module, config: dict[str, object]) -> tuple[list[torch.optim.Optimizer], Muon | None]:
    train_cfg = config["train"]
    optimizer_name = str(train_cfg["optimizer"]).strip().lower()
    weight_decay = float(train_cfg["weight_decay"])
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            [{"params": list(model.parameters()), "lr": float(train_cfg["lr"]), "base_lr": float(train_cfg["lr"])}],
            lr=float(train_cfg["lr"]),
            weight_decay=weight_decay,
        )
        return [optimizer], None
    if optimizer_name not in {"muon", "muon_split"}:
        raise ValueError(f"Unsupported optimizer setting: {train_cfg['optimizer']!r}")

    embed_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    matrix_params: list[torch.nn.Parameter] = []
    scalar_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("tok_emb.") or name.startswith("emb_proj."):
            embed_params.append(param)
        elif name.startswith("lm_head."):
            head_params.append(param)
        elif name.endswith(".weight_latent"):
            matrix_params.append(param)
        elif param.ndim == 2 and (name.startswith("attn_bank.") or name.startswith("mlp_bank.")):
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    optimizers: list[torch.optim.Optimizer] = []
    if embed_params:
        optimizers.append(
            torch.optim.AdamW(
                [{"params": embed_params, "lr": float(train_cfg["embed_lr"]), "base_lr": float(train_cfg["embed_lr"])}],
                lr=float(train_cfg["embed_lr"]),
                weight_decay=weight_decay,
            )
        )
    if head_params:
        optimizers.append(
            torch.optim.AdamW(
                [{"params": head_params, "lr": float(train_cfg["head_lr"]), "base_lr": float(train_cfg["head_lr"])}],
                lr=float(train_cfg["head_lr"]),
                weight_decay=weight_decay,
            )
        )
    muon_optimizer: Muon | None = None
    if matrix_params:
        muon_optimizer = Muon(
            [{"params": matrix_params, "lr": float(train_cfg["matrix_lr"]), "base_lr": float(train_cfg["matrix_lr"])}],
            lr=float(train_cfg["matrix_lr"]),
            momentum=float(train_cfg["muon_momentum"]),
            backend_steps=int(train_cfg["muon_backend_steps"]),
            weight_decay=weight_decay,
        )
        optimizers.append(muon_optimizer)
    if scalar_params:
        optimizers.append(
            torch.optim.AdamW(
                [{"params": scalar_params, "lr": float(train_cfg["scalar_lr"]), "base_lr": float(train_cfg["scalar_lr"])}],
                lr=float(train_cfg["scalar_lr"]),
                weight_decay=weight_decay,
            )
        )
    return optimizers, muon_optimizer


def _update_muon_momentum(muon_optimizer: Muon | None, step: int, config: dict[str, object]) -> None:
    if muon_optimizer is None:
        return
    train_cfg = config["train"]
    warmup_steps = int(train_cfg["muon_momentum_warmup_steps"])
    frac = min(step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
    target = float(train_cfg["muon_momentum"])
    start = float(train_cfg["muon_momentum_warmup_start"])
    momentum = (1.0 - frac) * start + frac * target
    for group in muon_optimizer.param_groups:
        group["momentum"] = momentum


def run_training(config: dict[str, object]) -> dict[str, object]:
    run_dir = build_run_dir(config)
    logger = RunLogger(run_dir)
    seed_everything(int(config["run"]["seed"]))
    logger.write_json("resolved_config.json", config)
    device = resolve_device(str(config["run"]["device"]))
    autocast_dtype = resolve_autocast_dtype(device, str(config["run"]["dtype"]))
    model = build_model(config).to(device)
    batcher = TokenBatcher(config["data"]["train_glob"], device=device)
    optimizers, muon_optimizer = _build_optimizers(model, config)
    best_val: float | None = None
    best_path = run_dir / "best.pt"
    checkpoint_path = run_dir / "last.pt"
    max_wallclock_seconds = float(config["train"]["max_wallclock_seconds"])
    started = time.perf_counter()
    swa_state: dict[str, torch.Tensor] | None = None
    swa_count = 0
    model.train()
    final_step = 0

    for step in range(int(config["train"]["iterations"])):
        elapsed = time.perf_counter() - started
        if max_wallclock_seconds > 0 and elapsed >= max_wallclock_seconds:
            logger.log(f"wallclock_stop step={step}")
            break
        progress = _progress_frac(step, int(config["train"]["iterations"]))
        lr_scale = _lr_scale(step, int(config["train"]["warmup_steps"]))
        qat_active = bool(config["train"]["projection_qat_enabled"]) and progress >= float(config["train"]["qat_start_frac"])
        if qat_active:
            lr_scale *= float(config["train"]["qat_lr_scale"])
        _set_lr_scale(optimizers, lr_scale)
        _update_muon_momentum(muon_optimizer, step, config)
        _zero_grad_all(optimizers)
        meta_enabled = bool(config["adaptation"]["meta_enabled"])
        meta_every = int(config["adaptation"]["meta_every"])
        use_meta_step = meta_enabled and meta_every > 0 and (step % meta_every == 0)
        compression_reg_active = (
            float(config["train"]["compression_reg_weight"]) > 0.0
            and progress >= float(config["train"]["compression_reg_start_frac"])
            and step % max(int(config["train"]["compression_reg_every_steps"]), 1) == 0
        )
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            if use_meta_step:
                adapt_tokens = int(config["adaptation"]["meta_adapt_tokens"])
                query_tokens = int(config["adaptation"]["meta_query_tokens"])
                local = batcher.next_local_span(adapt_tokens + query_tokens + 1, same_file=False)
                x_a = local[:adapt_tokens].reshape(1, adapt_tokens)
                y_a = local[1 : adapt_tokens + 1].reshape(1, adapt_tokens)
                x_b = local[adapt_tokens : adapt_tokens + query_tokens].reshape(1, query_tokens)
                y_b = local[adapt_tokens + 1 : adapt_tokens + query_tokens + 1].reshape(1, query_tokens)
                zero_state = init_writable_state(model, device)
                loss_a = model(x_a, y_a, writable_state=zero_state)
                adapted = adapt_writable_state(
                    model,
                    zero_state,
                    x_a,
                    y_a,
                    steps=int(config["adaptation"]["meta_steps"]),
                    clip=float(config["adaptation"]["inner_clip"]),
                    doc_bias_lr=float(config["adaptation"]["doc_bias_lr"]),
                    doc_bias_decay=float(config["adaptation"]["doc_bias_decay"]),
                    create_graph=False,
                    detach_result=False,
                )
                loss_b = model(x_b, y_b, writable_state=adapted)
                loss = (
                    float(config["adaptation"]["meta_loss_a_weight"]) * loss_a
                    + float(config["adaptation"]["meta_loss_b_weight"]) * loss_b
                )
            else:
                x, y = batcher.next_batch(int(config["train"]["batch_tokens"]), int(config["train"]["seq_len"]))
                loss = model(x, y)
            if compression_reg_active:
                reg = compute_export_grid_regularizer(
                    model,
                    quant_scheme=str(config["export"]["quant_scheme"]),
                    keep_float_max_numel=int(config["export"]["keep_float_max_numel"]),
                    keep_float_policy=str(config["export"]["keep_float_policy"]),
                    bitlinear_group_size=int(config["model"]["bitlinear_group_size"]),
                )
                loss = loss + float(config["train"]["compression_reg_weight"]) * reg.to(dtype=loss.dtype)
        loss.backward()
        clip_norm = float(config["train"]["grad_clip_norm"])
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        _step_all(optimizers)
        if qat_active and (step + 1) % max(int(config["train"]["qat_every_steps"]), 1) == 0:
            project_model_to_export_grid(
                model,
                quant_scheme=str(config["export"]["quant_scheme"]),
                keep_float_max_numel=int(config["export"]["keep_float_max_numel"]),
                keep_float_policy=str(config["export"]["keep_float_policy"]),
                bitlinear_group_size=int(config["model"]["bitlinear_group_size"]),
            )
        final_step = step + 1

        if step == 0 or (step + 1) % int(config["train"]["train_log_every"]) == 0 or step + 1 == int(config["train"]["iterations"]):
            elapsed_ms = 1000.0 * (time.perf_counter() - started)
            logger.log(
                f"step:{step + 1}/{config['train']['iterations']} loss:{float(loss.item()):.4f} "
                f"meta:{use_meta_step} train_time:{elapsed_ms:.0f}ms"
            )
            logger.event(
                "train_step",
                step=step + 1,
                loss=float(loss.item()),
                meta=bool(use_meta_step),
                train_time_ms=float(elapsed_ms),
            )

        if bool(config["train"]["swa_enabled"]) and progress >= float(config["train"]["swa_start_frac"]) and (step + 1) % max(int(config["train"]["swa_every"]), 1) == 0:
            current_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            if swa_state is None:
                swa_state = current_state
            else:
                for name, tensor in current_state.items():
                    swa_state[name] += tensor
            swa_count += 1

        val_every = int(config["train"]["val_every"])
        if val_every > 0 and ((step + 1) % val_every == 0):
            val_result = evaluate_model(model, config, policy_name=str(config["eval"]["policy"]), logger=logger)
            logger.log(
                f"val step:{step + 1} val_loss:{val_result['val_loss']:.4f} val_bpb:{val_result['val_bpb']:.4f} "
                f"policy:{val_result['policy']['policy']}"
            )
            logger.write_json("latest_val.json", val_result)
            if best_val is None or float(val_result["val_bpb"]) < best_val:
                best_val = float(val_result["val_bpb"])
                torch.save({"model_state": model.state_dict(), "config": config, "step": step + 1}, best_path)

    if bool(config["train"]["restore_best_val_checkpoint"]) and best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best_checkpoint["model_state"], strict=True)
    elif bool(config["train"]["swa_enabled"]) and swa_state is not None and swa_count > 1:
        avg_state = {
            name: (tensor / swa_count).to(dtype=model.state_dict()[name].dtype)
            for name, tensor in swa_state.items()
        }
        model.load_state_dict(avg_state, strict=True)

    state_dict = model.state_dict()
    torch.save({"model_state": state_dict, "config": config, "step": final_step}, checkpoint_path)
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "run_stage": str(config["run"].get("stage", "smoke")),
        "train_time_ms": float(elapsed_ms),
        "stopped_at_step": int(final_step),
    }
    if best_path.exists():
        result["best_checkpoint"] = str(best_path)
    if _should_write_artifact_after_train(config):
        artifact_path = run_dir / str(config["export"]["artifact_name"])
        _quant_obj, export_stats = write_quantized_artifact(
            state_dict,
            artifact_path,
            keep_float_max_numel=int(config["export"]["keep_float_max_numel"]),
            zlib_level=int(config["export"]["zlib_level"]),
            quant_scheme=str(config["export"]["quant_scheme"]),
            keep_float_policy=str(config["export"]["keep_float_policy"]),
            bitlinear_group_size=int(config["model"]["bitlinear_group_size"]),
        )
        budget_metrics = build_submission_size_metrics(
            artifact_bytes=int(export_stats["artifact_bytes"]),
            budget_bytes=int(config["export"]["artifact_budget_bytes"]),
            budget_mode=str(config["export"]["budget_mode"]),
        )
        export_result = {
            "artifact_path": str(artifact_path),
            "artifact_bytes": int(export_stats["artifact_bytes"]),
            "model_artifact_bytes": int(export_stats["artifact_bytes"]),
            "artifact_raw_bytes": int(export_stats["raw_bytes"]),
            "payload_bytes": int(export_stats["payload_bytes"]),
            "baseline_tensor_bytes": int(export_stats["baseline_tensor_bytes"]),
            "param_count": int(export_stats["param_count"]),
            "num_quantized_tensors": int(export_stats["num_quantized_tensors"]),
            "quant_scheme": str(config["export"]["quant_scheme"]),
            "keep_float_policy": str(config["export"]["keep_float_policy"]),
            **budget_metrics,
        }
        logger.write_json("export_stats.json", export_result)
        logger.log(
            "artifact "
            f"model_bytes:{export_result['model_artifact_bytes']} code_bytes:{export_result['counted_code_bytes']} "
            f"submission_total:{export_result['submission_total_bytes']} budget:{export_result['budget_bytes']} "
            f"within_budget:{export_result['within_budget']}"
        )
        logger.event("artifact_export", **export_result)
        result["export"] = export_result
        if export_result["over_budget"] and bool(config["export"]["fail_if_over_budget"]):
            logger.write_json("train_result.json", result)
            raise RuntimeError(
                f"Submission exceeds budget: {export_result['budgeted_bytes']} > {export_result['budget_bytes']} bytes"
            )
    logger.write_json("train_result.json", result)
    return result
