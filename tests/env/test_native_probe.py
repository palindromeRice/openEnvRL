from src.env.native_miniwob_probe import summarize_probe_result


def test_summarize_probe_result_extracts_required_keys():
    raw = {
        "utterance": "Select Apple",
        "fields": {"target 0": "Apple"},
        "dom_elements": [{"ref": 1, "text": "Apple"}],
    }
    summary = summarize_probe_result(raw)
    assert summary["instruction"] == "Select Apple"
    assert "fields" in summary
    assert "dom_refs" in summary
