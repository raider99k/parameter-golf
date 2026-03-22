from __future__ import annotations

from hybrid_golf.config import DEFAULT_CONFIG, apply_overrides, load_config_file


def test_apply_overrides_nested_types(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = load_config_file(config_path, overrides=["train.iterations=7", "experts.enable_ngram=false", "run.id=\"x\""])
    assert config["train"]["iterations"] == 7
    assert config["experts"]["enable_ngram"] is False
    assert config["run"]["id"] == "x"


def test_apply_overrides_creates_new_nested_keys():
    config = apply_overrides(DEFAULT_CONFIG, ["eval.extra.debug=true"])
    assert config["eval"]["extra"]["debug"] is True


def test_default_config_exposes_mainline_submission_fields():
    assert DEFAULT_CONFIG["model"]["num_unique_attn"] == 0
    assert DEFAULT_CONFIG["model"]["normformer_lite"] is False
    assert DEFAULT_CONFIG["model"]["recurrent_passes"] == 1
    assert DEFAULT_CONFIG["model"]["linear_impl"] == "dense"
    assert DEFAULT_CONFIG["model"]["bitlinear_targets"] == "none"
    assert DEFAULT_CONFIG["train"]["optimizer"] == "adamw"
    assert DEFAULT_CONFIG["export"]["artifact_budget_bytes"] == 16_000_000
    assert DEFAULT_CONFIG["export"]["budget_mode"] == "submission_total"


def test_load_config_file_supports_relative_extends(tmp_path):
    base_path = tmp_path / "base.json"
    child_path = tmp_path / "child.json"
    base_path.write_text(
        '{"run":{"id":"base"},"model":{"num_layers":7,"normformer_lite":true},"export":{"quant_scheme":"mixed_v2"}}',
        encoding="utf-8",
    )
    child_path.write_text(
        '{"extends":"base.json","run":{"id":"child"},"model":{"recurrent_passes":2}}',
        encoding="utf-8",
    )
    config = load_config_file(child_path)
    assert config["run"]["id"] == "child"
    assert config["model"]["num_layers"] == 7
    assert config["model"]["normformer_lite"] is True
    assert config["model"]["recurrent_passes"] == 2
    assert config["export"]["quant_scheme"] == "mixed_v2"
