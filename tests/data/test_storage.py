from src.data.storage import read_jsonl, write_jsonl, write_jsonl_atomic


def test_write_jsonl_compatibility_wrapper_roundtrips_rows(tmp_path):
    path = tmp_path / "traj.jsonl"
    rows = [{"instruction": "Select Apple", "action": "CLICK ref=1"}]

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows


def test_write_jsonl_atomic_replaces_only_valid_complete_file(tmp_path):
    path = tmp_path / "successful_trajectories.jsonl"

    write_jsonl_atomic(path, [{"episode_id": "id-1"}])

    assert read_jsonl(path) == [{"episode_id": "id-1"}]
    assert not list(tmp_path.glob("*.tmp"))
