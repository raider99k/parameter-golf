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
