from src.config import load_config


def test_load_config_reads_model_and_task(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model_id: Qwen/Qwen3.5-0.8B\ntask_id: click-checkboxes\n")
    cfg = load_config(path)
    assert cfg["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert cfg["task_id"] == "click-checkboxes"
