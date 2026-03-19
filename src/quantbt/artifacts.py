from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def safe_slug(value: Any) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "_")
        .lower()
    )


def _format_yyyymmdd(token: str) -> str:
    if not re.fullmatch(r"\d{8}", token):
        return safe_slug(token)
    return f"{token[0:4]}-{token[4:6]}-{token[6:8]}"


def dataset_folder_parts(dataset_tag: str) -> tuple[str, str]:
    cleaned = safe_slug(dataset_tag) or "dataset"
    parts = [p for p in cleaned.split("_") if p]
    if len(parts) >= 4 and re.fullmatch(r"\d{8}", parts[-1]) and re.fullmatch(r"\d{8}", parts[-2]):
        dataset_slug = "_".join(parts[:-2]) or "dataset"
        date_window = f"{_format_yyyymmdd(parts[-2])}_{_format_yyyymmdd(parts[-1])}"
        return dataset_slug, date_window
    return cleaned, "undated"


def run_id_now() -> str:
    return datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")


def _unique_run_dir(parent: Path) -> Path:
    stem = run_id_now()
    out = parent / stem
    suffix = 1
    while out.exists():
        suffix += 1
        out = parent / f"{stem}_{suffix:02d}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def strategies_root(base: str | Path = "runs") -> Path:
    return Path(base) / "strategies"


def make_strategy_run_dir(
    *,
    base: str | Path = "runs",
    strategy: str,
    workflow: str,
    category: str,
    dataset_tag: str,
    extra_parts: list[str] | None = None,
) -> Path:
    dataset_slug, date_window = dataset_folder_parts(dataset_tag)
    parent = (
        strategies_root(base)
        / safe_slug(strategy)
        / safe_slug(workflow)
        / safe_slug(category)
        / safe_slug(dataset_slug)
        / safe_slug(date_window)
    )
    for part in extra_parts or []:
        parent = parent / safe_slug(part)
    parent.mkdir(parents=True, exist_ok=True)
    return _unique_run_dir(parent)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "manifest.json"


def spec_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "spec.json"


def summary_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "summary.json"


def build_manifest(
    *,
    run_dir: str | Path,
    workflow: str,
    strategy: str,
    category: str,
    dataset_tag: str,
    created_at: str | None = None,
    parent_run_dir: str | Path | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_slug, date_window = dataset_folder_parts(dataset_tag)
    payload: dict[str, Any] = {
        "run_id": Path(run_dir).name,
        "workflow": safe_slug(workflow),
        "strategy": safe_slug(strategy),
        "category": safe_slug(category),
        "dataset_tag": str(dataset_tag),
        "dataset_slug": dataset_slug,
        "date_window": date_window,
        "created_at": created_at or iso_now(),
        "run_dir": str(Path(run_dir)),
    }
    if parent_run_dir is not None:
        payload["parent_run_dir"] = str(Path(parent_run_dir))
        payload["parent_run_id"] = Path(parent_run_dir).name
    if extras:
        payload.update(extras)
    return payload


def write_manifest(
    run_dir: str | Path,
    *,
    workflow: str,
    strategy: str,
    category: str,
    dataset_tag: str,
    created_at: str | None = None,
    parent_run_dir: str | Path | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    path = manifest_path(run_dir)
    write_json(
        path,
        build_manifest(
            run_dir=run_dir,
            workflow=workflow,
            strategy=strategy,
            category=category,
            dataset_tag=dataset_tag,
            created_at=created_at,
            parent_run_dir=parent_run_dir,
            extras=extras,
        ),
    )
    return path


def tables_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "tables"


def plots_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "plots"


def limited_iterations_path(run_dir: str | Path) -> Path:
    return tables_dir(run_dir) / "iterations.csv"


def limited_trades_path(run_dir: str | Path) -> Path:
    return tables_dir(run_dir) / "trades.csv"


def limited_html_path(run_dir: str | Path) -> Path:
    return plots_dir(run_dir) / "interactive.html"


def limited_iteration_html_path(run_dir: str | Path, iter_id: int) -> Path:
    return plots_dir(run_dir) / f"iteration_{int(iter_id)}_lightweight.html"


def walkforward_folds_path(run_dir: str | Path) -> Path:
    return tables_dir(run_dir) / "folds.csv"


def walkforward_oos_equity_path(run_dir: str | Path) -> Path:
    return tables_dir(run_dir) / "oos_equity_curve.csv"


def walkforward_oos_trades_path(run_dir: str | Path) -> Path:
    return tables_dir(run_dir) / "oos_trades.csv"


def walkforward_param_schedule_path(run_dir: str | Path) -> Path:
    return tables_dir(run_dir) / "parameter_schedule.json"


def walkforward_html_path(run_dir: str | Path) -> Path:
    return plots_dir(run_dir) / "interactive.html"


def walkforward_baseline_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "baseline"


def walkforward_baseline_results_path(run_dir: str | Path) -> Path:
    return walkforward_baseline_dir(run_dir) / "results.csv"


def walkforward_baseline_summary_path(run_dir: str | Path) -> Path:
    return walkforward_baseline_dir(run_dir) / "summary.json"


def walkforward_baseline_study_path(run_dir: str | Path) -> Path:
    return walkforward_baseline_dir(run_dir) / "study.db"


def walkforward_folds_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "folds"


def walkforward_fold_dir(run_dir: str | Path, fold_tag: str) -> Path:
    return walkforward_folds_dir(run_dir) / safe_slug(fold_tag)


def walkforward_fold_tables_dir(run_dir: str | Path, fold_tag: str) -> Path:
    return walkforward_fold_dir(run_dir, fold_tag) / "tables"


def walkforward_fold_is_results_path(run_dir: str | Path, fold_tag: str) -> Path:
    return walkforward_fold_tables_dir(run_dir, fold_tag) / "is_results.csv"


def walkforward_fold_oos_equity_path(run_dir: str | Path, fold_tag: str) -> Path:
    return walkforward_fold_tables_dir(run_dir, fold_tag) / "oos_equity_curve.csv"


def walkforward_fold_oos_trades_path(run_dir: str | Path, fold_tag: str) -> Path:
    return walkforward_fold_tables_dir(run_dir, fold_tag) / "oos_trades.csv"


def walkforward_fold_summary_path(run_dir: str | Path, fold_tag: str) -> Path:
    return walkforward_fold_dir(run_dir, fold_tag) / "summary.json"


def walkforward_fold_study_path(run_dir: str | Path, fold_tag: str) -> Path:
    return walkforward_fold_dir(run_dir, fold_tag) / "study.db"


def montecarlo_simulations_path(run_dir: str | Path) -> Path:
    return tables_dir(run_dir) / "simulations.csv"


def montecarlo_sample_paths_path(run_dir: str | Path) -> Path:
    return tables_dir(run_dir) / "sample_paths.csv"


def montecarlo_quantile_paths_path(run_dir: str | Path) -> Path:
    return tables_dir(run_dir) / "quantile_paths.csv"


def montecarlo_html_path(run_dir: str | Path) -> Path:
    return plots_dir(run_dir) / "interactive.html"


def make_montecarlo_run_dir(
    *,
    base: str | Path = "runs",
    strategy: str,
    dataset_tag: str,
    parent_run_id: str,
) -> Path:
    return make_strategy_run_dir(
        base=base,
        strategy=strategy,
        workflow="monte_carlo",
        category=f"from_{safe_slug(parent_run_id)}",
        dataset_tag=dataset_tag,
    )


MONKEY_ENTRY_PLUGIN_NAMES = {"monkey_entry", "random"}
MONKEY_EXIT_PLUGIN_NAMES = {"monkey_exit", "random_time_exit"}


def _entry_plugin_names(strategy_spec: dict[str, Any]) -> tuple[str, ...]:
    rules = strategy_spec.get("entry", {}).get("rules", [])
    names = [str(rule.get("name", "")).strip() for rule in rules if isinstance(rule, dict)]
    return tuple(name for name in names if name)


def infer_limited_scenario_slug(strategy_spec: dict[str, Any], *, test_focus: str, test_name: str = "") -> str:
    entry_names = _entry_plugin_names(strategy_spec)
    exit_name = str(strategy_spec.get("exit", {}).get("name", "")).strip()

    entry_is_monkey = bool(entry_names) and set(entry_names).issubset(MONKEY_ENTRY_PLUGIN_NAMES)
    exit_is_monkey = exit_name in MONKEY_EXIT_PLUGIN_NAMES

    if entry_is_monkey and exit_is_monkey:
        return "monkey_entry_exit"
    if entry_is_monkey:
        return "monkey_entry"
    if exit_is_monkey:
        return "monkey_exit"

    focus = str(test_focus or "").strip().lower()
    if focus == "core_system_test":
        return "core"
    if focus == "exit_test":
        return "exit_test"
    if focus == "entry_test":
        if exit_name == "time_exit":
            return "entry_fixed_bar_exit"
        if exit_name == "atr_brackets":
            return "entry_fixed_atr_exit"
        if exit_name == "fixed_pips_brackets":
            return "entry_fixed_pips_exit"
        return f"entry_test__{safe_slug(exit_name or test_name or 'custom_exit')}"

    return safe_slug(test_name or "limited_test")


def walkforward_profile_slug(*, optimizer: str, anchored: bool) -> str:
    return f"{safe_slug(optimizer)}_{'anchored' if anchored else 'unanchored'}"
