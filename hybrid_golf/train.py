from __future__ import annotations

import time

import torch

from .adaptation import adapt_writable_state, init_writable_state
from .data import TokenBatcher
from .evaluate import evaluate_model
from .model import build_model
from .runtime import RunLogger, build_run_dir, resolve_autocast_dtype, resolve_device, seed_everything


def _lr_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min((step + 1) / warmup_steps, 1.0)


def run_training(config: dict[str, object]) -> dict[str, str]:
    run_dir = build_run_dir(config)
    logger = RunLogger(run_dir)
    seed_everything(int(config["run"]["seed"]))
    logger.write_json("resolved_config.json", config)
    device = resolve_device(str(config["run"]["device"]))
    autocast_dtype = resolve_autocast_dtype(device, str(config["run"]["dtype"]))
    model = build_model(config).to(device)
    batcher = TokenBatcher(config["data"]["train_glob"], device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    best_val: float | None = None
    best_path = run_dir / "best.pt"
    checkpoint_path = run_dir / "last.pt"
    max_wallclock_seconds = float(config["train"]["max_wallclock_seconds"])
    started = time.perf_counter()
    model.train()

    for step in range(int(config["train"]["iterations"])):
        if max_wallclock_seconds > 0 and (time.perf_counter() - started) >= max_wallclock_seconds:
            logger.log(f"wallclock_stop step={step}")
            break
        lr_scale = _lr_scale(step, int(config["train"]["warmup_steps"]))
        for group in optimizer.param_groups:
            group["lr"] = float(config["train"]["lr"]) * lr_scale
        optimizer.zero_grad(set_to_none=True)
        meta_enabled = bool(config["adaptation"]["meta_enabled"])
        meta_every = int(config["adaptation"]["meta_every"])
        use_meta_step = meta_enabled and meta_every > 0 and (step % meta_every == 0)
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
        loss.backward()
        clip_norm = float(config["train"]["grad_clip_norm"])
        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

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

        val_every = int(config["train"]["val_every"])
        should_validate = val_every > 0 and ((step + 1) % val_every == 0)
        if should_validate:
            val_result = evaluate_model(model, config, policy_name=str(config["eval"]["policy"]), logger=logger)
            logger.log(
                f"val step:{step + 1} val_loss:{val_result['val_loss']:.4f} val_bpb:{val_result['val_bpb']:.4f} "
                f"policy:{val_result['policy']['policy']}"
            )
            logger.write_json("latest_val.json", val_result)
            if best_val is None or float(val_result["val_bpb"]) < best_val:
                best_val = float(val_result["val_bpb"])
                torch.save({"model_state": model.state_dict(), "config": config, "step": step + 1}, best_path)

    torch.save({"model_state": model.state_dict(), "config": config}, checkpoint_path)
    result = {"checkpoint": str(checkpoint_path), "run_dir": str(run_dir)}
    if best_path.exists():
        result["best_checkpoint"] = str(best_path)
    logger.write_json("train_result.json", result)
    return result
