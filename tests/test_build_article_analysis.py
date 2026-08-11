from pathlib import Path

import pytest

from scripts.build_article_analysis import run_analysis


def test_cli_refuses_nonempty_output_directory(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        run_analysis(
            output_dir=output,
            sft_root=tmp_path / "sft",
            grpo_root=tmp_path / "grpo",
            base_report_path=tmp_path / "base.md",
        )
