from src.data.storage import read_jsonl, write_jsonl


def test_write_and_read_jsonl_roundtrip(tmp_path):
    path = tmp_path / "traj.jsonl"
    rows = [{"instruction": "Select Apple", "action": "CLICK ref=1"}]
    write_jsonl(path, rows)
    loaded = read_jsonl(path)
    assert loaded == rows
