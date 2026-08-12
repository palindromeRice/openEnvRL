from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ArtifactValidationError(ValueError):
    """Raised when an authoritative analysis artifact is unusable."""


@dataclass(frozen=True)
class StageMetric:
    stage: str
    success_rate: float
    average_reward: float
    average_steps: float
    source_path: Path
    evidence_type: str


@dataclass(frozen=True)
class EpisodeStep:
    instruction: str
    observation: str
    action: str
    reward: float
    done: bool
    raw_completion: str | None
    parsed_action: str | None


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    steps: tuple[EpisodeStep, ...]
    success: bool
    total_reward: float
    source_line: int


@dataclass(frozen=True)
class TraceView:
    episode_id: str
    instruction: str
    initial_observation: str
    raw_completions: tuple[str, ...]
    parsed_actions: tuple[str, ...]
    executed_actions: tuple[str, ...]
    rewards: tuple[float, ...]
    done_flags: tuple[bool, ...]
    success: bool
    queue_is_derived: bool


@dataclass(frozen=True)
class ValidationIssue:
    source_line: int | None
    episode_id: str | None
    field_path: str
    error_code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    source_path: Path
    record_count: int
    required_fields: tuple[str, ...]
    invalid_records: tuple[ValidationIssue, ...]
    aggregate_reconciled: bool | None = None


@dataclass(frozen=True)
class FailureClassification:
    category: str
    episode_id: str
    clicked_labels: tuple[str, ...]
    requested_labels: tuple[str, ...]


REQUIRED_STEP_FIELDS = (
    "steps[].instruction",
    "steps[].observation",
    "steps[].action",
    "steps[].reward",
    "steps[].done",
    "steps[].policy_diagnostics.raw_completion",
    "steps[].policy_diagnostics.parsed_action",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"missing artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"expected JSON object in {path}")
    return value


def _metric_from_json(path: Path, stage: str, evidence_type: str) -> StageMetric:
    value = _read_json(path)
    required = ("success_rate", "average_reward", "average_steps")
    for field in required:
        if field not in value or isinstance(value[field], bool):
            raise ArtifactValidationError(f"missing/non-numeric {field} in {path}")
        try:
            value[field] = float(value[field])
        except (TypeError, ValueError) as exc:
            raise ArtifactValidationError(f"missing/non-numeric {field} in {path}") from exc
    return StageMetric(
        stage=stage,
        success_rate=value["success_rate"],
        average_reward=value["average_reward"],
        average_steps=value["average_steps"],
        source_path=path,
        evidence_type=evidence_type,
    )


def load_base_metric_from_report(path: Path) -> StageMetric:
    if not path.is_file():
        raise ArtifactValidationError(f"missing artifact: {path}")
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"\|\s*Base(?: Qwen model)?\s*\|\s*\d+\s*\|\s*(\d+(?:\.\d+)?)%\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|",
        text,
    )
    if match is None:
        raise ArtifactValidationError(f"Base comparison row missing in {path}")
    success, reward, steps = (float(item) for item in match.groups())
    return StageMetric(
        stage="base",
        success_rate=success / 100.0,
        average_reward=reward,
        average_steps=steps,
        source_path=path,
        evidence_type="documented_base_comparison",
    )


def load_stage_metrics(
    base_report_path: Path,
    sft_metrics_path: Path,
    grpo_metrics_path: Path,
) -> dict[str, StageMetric]:
    return {
        "base": load_base_metric_from_report(base_report_path),
        "sft": _metric_from_json(sft_metrics_path, "sft", "full_sft_aggregate"),
        "grpo": _metric_from_json(
            grpo_metrics_path, "grpo", "corrected_final_grpo_aggregate"
        ),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _parse_step(value: Mapping[str, Any], line_number: int) -> EpisodeStep:
    required = ("instruction", "observation", "action", "reward", "done")
    missing = [field for field in required if field not in value]
    if missing:
        raise ArtifactValidationError(
            f"line {line_number}: missing step fields {', '.join(missing)}"
        )
    diagnostics = value.get("policy_diagnostics") or {}
    return EpisodeStep(
        instruction=str(value["instruction"]),
        observation=str(value["observation"]),
        action=str(value["action"]),
        reward=float(value["reward"]),
        done=bool(value["done"]),
        raw_completion=diagnostics.get("raw_completion"),
        parsed_action=diagnostics.get("parsed_action"),
    )


def _parse_episode(value: Mapping[str, Any], line_number: int) -> EpisodeRecord:
    episode_id = value.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ArtifactValidationError(f"line {line_number}: episode_id missing or invalid")
    steps = value.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ArtifactValidationError(f"line {line_number}: steps missing or empty")
    parsed_steps = tuple(_parse_step(step, line_number) for step in steps)
    return EpisodeRecord(
        episode_id=episode_id,
        steps=parsed_steps,
        success=value.get("success") is True,
        total_reward=float(value.get("total_reward", 0.0)),
        source_line=line_number,
    )


def load_final_grpo_episodes(path: Path) -> tuple[EpisodeRecord, ...]:
    if not path.is_file():
        raise ArtifactValidationError(f"missing artifact: {path}")
    records: list[EpisodeRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(f"line {line_number} invalid JSON in {path}") from exc
        if not isinstance(value, dict):
            raise ArtifactValidationError(f"line {line_number} expected JSON object in {path}")
        try:
            records.append(_parse_episode(value, line_number))
        except (ArtifactValidationError, TypeError, ValueError) as exc:
            raise ArtifactValidationError(f"{path}: {exc}") from exc
    return tuple(records)


def validate_episode_records(
    records: Sequence[EpisodeRecord],
    source_path: Path,
    expected_count: int = 50,
    diagnostic_mode: bool = False,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if len(records) != expected_count:
        issue = ValidationIssue(
            source_line=None,
            episode_id=None,
            field_path="records",
            error_code="wrong_record_count",
            message=f"expected {expected_count}, found {len(records)}",
        )
        if not diagnostic_mode:
            raise ArtifactValidationError(f"{source_path}: {issue.message}")
        issues.append(issue)

    seen: set[str] = set()
    for record in records:
        if record.episode_id in seen:
            issue = ValidationIssue(
                record.source_line, record.episode_id, "episode_id", "duplicate_episode_id", "duplicate episode_id"
            )
            if not diagnostic_mode:
                raise ArtifactValidationError(f"{source_path}: {issue.message}")
            issues.append(issue)
        seen.add(record.episode_id)
        if not record.steps:
            issue = ValidationIssue(
                record.source_line, record.episode_id, "steps", "missing_steps", "steps must be non-empty"
            )
            if not diagnostic_mode:
                raise ArtifactValidationError(f"{source_path}: {issue.message}")
            issues.append(issue)
        for step_index, step in enumerate(record.steps):
            for field_name, value in (
                ("instruction", step.instruction),
                ("observation", step.observation),
                ("action", step.action),
            ):
                if not value:
                    issue = ValidationIssue(
                        record.source_line,
                        record.episode_id,
                        f"steps[{step_index}].{field_name}",
                        "empty_field",
                        f"{field_name} must be non-empty",
                    )
                    if not diagnostic_mode:
                        raise ArtifactValidationError(f"{source_path}: {issue.message}")
                    issues.append(issue)

    return ValidationReport(
        valid=not issues,
        source_path=source_path,
        record_count=len(records),
        required_fields=REQUIRED_STEP_FIELDS,
        invalid_records=tuple(issues),
    )


def reconcile_grpo_metrics(
    records: Sequence[EpisodeRecord],
    aggregate_metrics: Mapping[str, Any] | StageMetric,
) -> bool:
    count = len(records)
    if count == 0:
        raise ArtifactValidationError("cannot reconcile empty episode records")
    expected = {
        "success_rate": float(aggregate_metrics.success_rate if isinstance(aggregate_metrics, StageMetric) else aggregate_metrics["success_rate"]),
        "average_reward": float(aggregate_metrics.average_reward if isinstance(aggregate_metrics, StageMetric) else aggregate_metrics["average_reward"]),
        "average_steps": float(aggregate_metrics.average_steps if isinstance(aggregate_metrics, StageMetric) else aggregate_metrics["average_steps"]),
    }
    actual = {
        "success_rate": sum(record.success for record in records) / count,
        "average_reward": sum(record.total_reward for record in records) / count,
        "average_steps": sum(len(record.steps) for record in records) / count,
    }
    return all(abs(actual[key] - expected[key]) < 1e-9 for key in expected)


def load_sft_demonstrations(
    path: Path,
    limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise ArtifactValidationError(f"missing artifact: {path}")
    result: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ArtifactValidationError(f"line {line_number} invalid JSON in {path}") from exc
        if not isinstance(value, dict):
            raise ArtifactValidationError(f"line {line_number} expected JSON object in {path}")
        result.append({**value, "evidence_type": "SFT training demonstration"})
        if limit is not None and len(result) >= limit:
            break
    return tuple(result)


def extract_representative_trace(episode: EpisodeRecord) -> TraceView:
    if not episode.steps:
        raise ArtifactValidationError(f"episode {episode.episode_id} has no steps")
    return TraceView(
        episode_id=episode.episode_id,
        instruction=episode.steps[0].instruction,
        initial_observation=episode.steps[0].observation,
        raw_completions=tuple(
            step.raw_completion or "" for step in episode.steps
        ),
        parsed_actions=tuple(
            step.parsed_action or step.action for step in episode.steps
        ),
        executed_actions=tuple(step.action for step in episode.steps),
        rewards=tuple(step.reward for step in episode.steps),
        done_flags=tuple(step.done for step in episode.steps),
        success=episode.success,
        queue_is_derived=True,
    )


def summarize_final_episodes(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    lengths = [len(record.steps) for record in records]
    terminal_actions = [record.steps[-1].action for record in records if record.steps]
    return {
        "episode_count": len(records),
        "success_count": sum(record.success for record in records),
        "failure_count": sum(not record.success for record in records),
        "success_rate": sum(record.success for record in records) / len(records) if records else 0.0,
        "episode_lengths": lengths,
        "episode_successes": [record.success for record in records],
        "terminal_action_counts": {
            action: terminal_actions.count(action) for action in sorted(set(terminal_actions))
        },
        "diagnostic_step_count": sum(
            1
            for record in records
            for step in record.steps
            if step.raw_completion is not None or step.parsed_action is not None
        ),
    }


def _requested_labels(instruction: str) -> tuple[str, ...]:
    match = re.search(r"Select (.+) and click Submit\.", instruction)
    if match is None:
        raise ArtifactValidationError(f"cannot parse requested labels: {instruction}")
    return tuple(label.strip() for label in match.group(1).split(","))


def _visible_control_labels(observation: str) -> dict[str, str]:
    return {
        control_id: label
        for control_id, label in re.findall(
            r"\[(\d+)\] (?:checkbox|button) '([^']+)'", observation
        )
    }


def _action_target(action: str) -> str | None:
    match = re.search(r"click\('([^']+)'\)", action)
    return match.group(1) if match else None


def classify_failure(record: EpisodeRecord) -> FailureClassification:
    if record.success:
        raise ArtifactValidationError(f"cannot classify successful episode: {record.episode_id}")
    requested = _requested_labels(record.steps[0].instruction)
    labels_by_id = _visible_control_labels(record.steps[0].observation)
    clicked_labels = tuple(
        labels_by_id[target]
        for step in record.steps
        if (target := _action_target(step.action)) is not None
        and target in labels_by_id
        and labels_by_id[target] != "Submit"
    )
    submitted = any(
        labels_by_id.get(target) == "Submit"
        for step in record.steps
        if (target := _action_target(step.action)) is not None
    )
    has_wrong_selection = any(label not in requested for label in clicked_labels)
    all_clicked_correct = all(label in requested for label in clicked_labels)
    if not submitted and all_clicked_correct:
        category = "no_submit_after_correct_selection"
    elif submitted and all_clicked_correct and len(set(clicked_labels)) < len(requested):
        category = "premature_submit"
    elif submitted and has_wrong_selection:
        category = "wrong_selection_then_submit"
    else:
        raise ArtifactValidationError(
            f"unclassifiable failed episode {record.episode_id}: "
            f"actions={tuple(step.action for step in record.steps)}"
        )
    return FailureClassification(category, record.episode_id, clicked_labels, requested)


def summarize_failure_behavior(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    classifications = tuple(classify_failure(record) for record in records if not record.success)
    categories = (
        "no_submit_after_correct_selection",
        "premature_submit",
        "wrong_selection_then_submit",
    )
    counts = {category: sum(item.category == category for item in classifications) for category in categories}
    return {
        "counts": counts,
        "classifications": [item.__dict__ for item in classifications],
    }
