from src.train.sft import run_sft_training, validate_sft_exports


def test_sft_training_saves_adapter_and_tokenizer_before_merged_export(tmp_path, monkeypatch):
    events = []

    class Model:
        def save_pretrained(self, path):
            events.append(("adapter", path))

        def save_pretrained_merged(self, path, tokenizer, save_method):
            events.append(("merged", path, tokenizer, save_method))

    class Tokenizer:
        def save_pretrained(self, path):
            events.append(("tokenizer", path))

    class Trainer:
        model = Model()
        processing_class = Tokenizer()

        def train(self):
            events.append(("train",))
            return {"global_step": 60}

    monkeypatch.setattr("src.train.sft.build_sft_trainer", lambda config: Trainer())
    config = {
        "output_dir": str(tmp_path / "merged"),
        "adapter_dir": str(tmp_path / "adapter"),
    }

    result = run_sft_training(config)

    assert result == {"global_step": 60}
    assert events == [
        ("train",),
        ("adapter", str(tmp_path / "adapter")),
        ("tokenizer", str(tmp_path / "adapter")),
        ("merged", str(tmp_path / "merged"), Trainer.processing_class, "merged_16bit"),
    ]
    assert (tmp_path / "adapter").is_dir()
    assert (tmp_path / "merged").is_dir()


def test_validate_sft_exports_loads_merged_and_adapter_from_expected_locations(monkeypatch):
    calls = []
    merged_model = object()
    base_model = object()
    adapter_model = object()
    monkeypatch.setattr(
        "src.train.sft._load_sft_tokenizer",
        lambda path: calls.append(("tokenizer", path)) or object(),
    )
    monkeypatch.setattr(
        "src.train.sft._load_sft_causal_lm",
        lambda path: calls.append(("model", path)) or merged_model,
    )
    monkeypatch.setattr(
        "src.train.sft._load_sft_base_causal_lm",
        lambda model_id: calls.append(("base_model", model_id)) or base_model,
    )
    monkeypatch.setattr(
        "src.train.sft._load_sft_adapter",
        lambda base, path: calls.append(("adapter", base, path)) or adapter_model,
    )
    config = {
        "output_dir": "models/sft_model",
        "adapter_dir": "models/sft_model/adapter",
        "model_id": "Qwen/Qwen3.5-0.8B",
    }

    result = validate_sft_exports(config)

    assert calls == [
        ("tokenizer", "models/sft_model"),
        ("model", "models/sft_model"),
        ("base_model", "Qwen/Qwen3.5-0.8B"),
        ("adapter", base_model, "models/sft_model/adapter"),
    ]
    assert result == {"merged_model": merged_model, "adapter_model": adapter_model}
