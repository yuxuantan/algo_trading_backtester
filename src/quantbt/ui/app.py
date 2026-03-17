from __future__ import annotations

import ast
import copy
from dataclasses import MISSING, fields, is_dataclass
from datetime import date
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlencode

from quantbt.artifacts import (
    limited_html_path,
    limited_iteration_html_path,
    limited_iterations_path,
    limited_trades_path,
    montecarlo_html_path,
    montecarlo_quantile_paths_path,
    montecarlo_simulations_path,
    spec_path,
    summary_path,
    walkforward_folds_path,
    walkforward_html_path,
    walkforward_oos_equity_path,
    walkforward_oos_trades_path,
    walkforward_param_schedule_path,
)
from quantbt.results import (
    load_limited_summary,
    load_montecarlo_summary,
    load_walkforward_summary,
)
from quantbt.io.datasets import read_dataset_meta

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from quantbt.ui.components import (
    _inject_app_css,
    _render_chip_group,
    _render_decision_badge,
    _render_metric_row,
    _render_status_badge,
    _status_label,
)
from quantbt.ui.pages.reference import render_reference_page as _render_reference_page

try:
    import altair as alt
except ImportError:  # pragma: no cover - fallback used when altair is unavailable
    alt = None

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - fallback used when plotly is unavailable
    px = None

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPO_ROOT / "runs"
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def _resolve_script_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


SCRIPT_PYTHON = _resolve_script_python()


COMMON_WF_OBJECTIVES = [
    "return_on_account",
    "total_return_%",
    "cagr_%",
    "sortino",
    "mar",
    "net_profit_abs",
    "max_drawdown_abs_%",
    "equity_linearity_score",
]

DEFAULT_ENTRY_PARAMS: dict[str, dict[str, Any]] = {
    "sma_cross": {"fast": [30, 50, 70], "slow": [150, 200, 250]},
    "donchian_breakout": {"lookback": [20]},
    "monkey_entry": {"target_entries": 120, "side": "both", "long_ratio": 0.5, "seed": 7},
    "random": {"target_entries": 120, "side": "both", "long_ratio": 0.5, "seed": 7},
}

DEFAULT_EXIT_PARAMS: dict[str, dict[str, Any]] = {
    "atr_brackets": {"rr": [1.5, 2.0, 2.5], "sldist_atr_mult": [1.0, 1.5], "atr_period": 14},
    "time_exit": {"hold_bars": [1]},
    "monkey_exit": {"avg_hold_bars": 15.0, "seed": 7},
    "random_time_exit": {"avg_hold_bars": 15.0, "seed": 7},
}

DEFAULT_SIZING_PARAMS: dict[str, dict[str, Any]] = {
    "fixed_risk": {"risk_pct": 0.01},
    "fixed_units": {"units": 100000},
}

LIMITED_WORKBOOK_GLOBAL_INPUTS: list[str] = [
    "Data start/end window",
    "Slippage / spread (pips)",
    "Commissions (round trip)",
]

LIMITED_WORKBOOK_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "ENTRY TEST: fixed stop and target exit",
        "test_type": "Fixed ATR Exit",
        "preset": "Fixed ATR Exit",
        "entry_params_label": "Baseline entry parameter set",
        "exit_params_label": "Fixed stop / target exit parameters",
        "seed_mode": "none",
        "show_monkey_helper": False,
        "inputs": [
            "Baseline entry parameter set",
            "Exit SL rule",
            "Exit TP rule (e.g. fixed RR)",
        ],
        "outputs": [
            "% of iterations profitable",
            "# iterations",
        ],
        "pass_criteria": [
            "% profitable > 70%",
            "# iterations > 50",
        ],
    },
    {
        "name": "ENTRY TEST: fixed bar exit",
        "test_type": "Fixed Bar Exit",
        "preset": "Fixed Bar Exit",
        "entry_params_label": "Baseline entry parameter set",
        "exit_params_label": "Exit bars parameter set",
        "seed_mode": "none",
        "show_monkey_helper": False,
        "inputs": [
            "Baseline entry parameter set",
            "Exit bars parameter set",
        ],
        "outputs": [
            "% of iterations profitable",
            "# iterations",
        ],
        "pass_criteria": [
            "% profitable > 70%",
            "# iterations > 50",
        ],
    },
    {
        "name": "EXIT TEST: similar approach entry",
        "test_type": "Similar Entry",
        "preset": "Similar Entry",
        "entry_params_label": "Similar-entry parameter set",
        "exit_params_label": "Baseline exit parameter set",
        "seed_mode": "none",
        "show_monkey_helper": False,
        "inputs": [
            "Similar-entry mechanics",
            "Limited optimisable params (max 2)",
            "Baseline exit param set",
        ],
        "outputs": [
            "% of iterations profitable",
            "# iterations",
        ],
        "pass_criteria": [
            "% profitable > 70%",
            "# iterations > 50",
        ],
    },
    {
        "name": "CORE TEST",
        "test_type": "Core",
        "preset": "Core",
        "entry_params_label": "Baseline entry parameter set",
        "exit_params_label": "Baseline exit parameter set",
        "seed_mode": "none",
        "show_monkey_helper": False,
        "inputs": [
            "Baseline entry parameter set",
            "Baseline exit parameter set",
        ],
        "outputs": [
            "% of iterations profitable",
            "# iterations",
        ],
        "pass_criteria": [
            "% profitable > 70%",
            "# iterations > 50",
        ],
    },
    {
        "name": "MONKEY ENTRY + BASELINE EXIT",
        "test_type": "Monkey Entry",
        "preset": "Monkey Entry",
        "entry_params_label": "Monkey entry parameters",
        "exit_params_label": "Baseline exit parameter set",
        "seed_mode": "entry",
        "show_monkey_helper": True,
        "inputs": [
            "Baseline trade count (target entries)",
            "Baseline long % (target long ratio)",
            "Baseline exit parameters",
        ],
        "outputs": [
            "% iterations with monkey net profit < baseline",
            "% iterations with monkey max dd > baseline",
        ],
        "pass_criteria": [
            "Profit dominance > 90%",
            "Drawdown dominance > 90%",
        ],
    },
    {
        "name": "BASELINE ENTRY + MONKEY EXIT",
        "test_type": "Monkey Exit",
        "preset": "Monkey Exit",
        "entry_params_label": "Baseline entry parameter set",
        "exit_params_label": "Monkey exit parameters",
        "seed_mode": "exit",
        "show_monkey_helper": True,
        "inputs": [
            "Baseline entry parameters",
            "Baseline avg hold bars (target)",
            "Baseline exit parameters",
        ],
        "outputs": [
            "% iterations with monkey net profit < baseline",
            "% iterations with monkey max dd > baseline",
        ],
        "pass_criteria": [
            "Profit dominance > 90%",
            "Drawdown dominance > 90%",
        ],
    },
    {
        "name": "MONKEY ENTRY + MONKEY EXIT",
        "test_type": "Monkey Entry + Exit",
        "preset": "Monkey Entry + Exit",
        "entry_params_label": "Monkey entry parameters",
        "exit_params_label": "Monkey exit parameters",
        "seed_mode": "entry",
        "show_monkey_helper": True,
        "inputs": [
            "Baseline trade count (target)",
            "Baseline long % (target)",
            "Baseline avg hold bars (target)",
        ],
        "outputs": [
            "% iterations with monkey net profit < baseline",
            "% iterations with monkey max dd > baseline",
        ],
        "pass_criteria": [
            "Profit dominance > 90%",
            "Drawdown dominance > 90%",
        ],
    },
]

WFA_WORKBOOK_INPUTS: list[str] = [
    "WFA data start/end",
    "Anchored / unanchored",
    "In-period years",
    "Out-period years",
    "Fitness factor",
    "Baseline entry param set",
    "Baseline exit param set",
]

WFA_WORKBOOK_REVIEW_ROWS: list[dict[str, str]] = [
    {"output": "total net profit", "pass_criteria": ">$5k per year per contract"},
    {"output": "profit factor", "pass_criteria": "> 1.0"},
    {"output": "avg trade net profit", "pass_criteria": "> $50 per trade per contract"},
    {"output": "tharp expectancy", "pass_criteria": "> 0.1"},
    {"output": "max dd", "pass_criteria": "< 20%"},
    {"output": "closed trade equity curve steadiness", "pass_criteria": "looks smooth diagonal up not mostly flat with rapid rises"},
    {"output": "dd period", "pass_criteria": "< 3 months (my own criteria)"},
    {"output": "win rate (my own metric)", "pass_criteria": ""},
]

WFA_FITNESS_FACTOR_PRESETS: dict[str, dict[str, str]] = {
    "Highest net profit": {"objective": "net_profit_abs", "direction": "maximize"},
    "Highest return on account": {"objective": "return_on_account", "direction": "maximize"},
    "Highest profit factor": {"objective": "profit_factor", "direction": "maximize"},
    "Lowest max drawdown %": {"objective": "max_drawdown_abs_%", "direction": "minimize"},
}

WORKBOOK_REVIEW_METRICS: list[dict[str, str]] = [
    {
        "label": "% of iterations profitable",
        "default_unit": "%",
        "aliases": "% profitable",
    },
    {
        "label": "# iterations",
        "default_unit": "count",
        "aliases": "iterations",
    },
    {
        "label": "% iterations with monkey net profit < baseline",
        "default_unit": "%",
        "aliases": "profit dominance|% of iterations with monkey net profit < baseline",
    },
    {
        "label": "% iterations with monkey max dd > baseline",
        "default_unit": "%",
        "aliases": "drawdown dominance|% of iterations with monkey max dd > baseline",
    },
    {
        "label": "Average hold bars",
        "default_unit": "bars",
        "aliases": "avg hold bars|avg bars held",
    },
    {
        "label": "total net profit",
        "default_unit": "raw",
        "aliases": "net profit|annualized net profit",
    },
    {
        "label": "profit factor",
        "default_unit": "ratio",
        "aliases": "",
    },
    {
        "label": "avg trade net profit",
        "default_unit": "raw",
        "aliases": "average trade net profit|avg trade profit",
    },
    {
        "label": "tharp expectancy",
        "default_unit": "ratio",
        "aliases": "expectancy",
    },
    {
        "label": "max dd",
        "default_unit": "%",
        "aliases": "max drawdown|max dd %",
    },
    {
        "label": "closed trade equity curve steadiness",
        "default_unit": "raw",
        "aliases": "equity curve steadiness|equity steadiness",
    },
    {
        "label": "dd period",
        "default_unit": "bars",
        "aliases": "drawdown period|max drawdown duration",
    },
    {
        "label": "win rate (my own metric)",
        "default_unit": "%",
        "aliases": "win rate",
    },
]
WORKBOOK_REVIEW_METRIC_OPTIONS: list[str] = [m["label"] for m in WORKBOOK_REVIEW_METRICS] + ["Custom"]
WORKBOOK_REVIEW_UNIT_OPTIONS: list[str] = ["%", "count", "bars", "ratio", "raw"]
WORKBOOK_REVIEW_OPERATOR_OPTIONS: list[str] = [">", ">=", "<", "<=", "=="]


def _multiline_items_text(items: list[Any]) -> str:
    return "\n".join(str(item).strip() for item in items if str(item).strip())


def _normalize_workbook_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _guess_workbook_metric_label(label: Any) -> str:
    text = _normalize_workbook_text(label)
    if not text:
        return "Custom"
    for item in WORKBOOK_REVIEW_METRICS:
        if text == _normalize_workbook_text(item.get("label", "")):
            return str(item["label"])
        aliases = [p.strip() for p in str(item.get("aliases", "")).split("|") if p.strip()]
        if any(text == _normalize_workbook_text(alias) for alias in aliases):
            return str(item["label"])
    return "Custom"


def _default_workbook_unit(metric_label: str) -> str:
    for item in WORKBOOK_REVIEW_METRICS:
        if str(item.get("label", "")) == str(metric_label):
            return str(item.get("default_unit", "raw"))
    return "raw"


def _build_workbook_output_rows(items: list[Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in items:
        label = str(item or "").strip()
        if not label:
            continue
        rows.append(
            {
                "output_label": label,
                "mapped_metric": _guess_workbook_metric_label(label),
                "notes": "",
            }
        )
    return rows


def _parse_workbook_rule_item(item: Any) -> dict[str, Any]:
    raw = str(item or "").strip()
    metric_label = _guess_workbook_metric_label(raw)
    rule_label = raw
    operator = ""
    threshold: float | None = None
    unit = _default_workbook_unit(metric_label)
    m = re.match(r"^(?P<label>.+?)\s*(?P<op>>=|<=|==|>|<)\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>%|count|bars|ratio|raw)?\s*$", raw)
    if m:
        rule_label = str(m.group("label") or "").strip()
        metric_label = _guess_workbook_metric_label(rule_label)
        operator = str(m.group("op") or "").strip()
        try:
            threshold = float(str(m.group("value") or "").strip())
        except Exception:
            threshold = None
        unit = str(m.group("unit") or _default_workbook_unit(metric_label)).strip() or "raw"
    return {
        "rule_label": rule_label,
        "mapped_metric": metric_label,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
    }


def _build_workbook_rule_rows(items: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        label = str(item or "").strip()
        if not label:
            continue
        rows.append(_parse_workbook_rule_item(label))
    return rows


def _clean_workbook_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        output_label = str(row.get("output_label", "") or "").strip()
        mapped_metric = str(row.get("mapped_metric", "Custom") or "Custom").strip()
        notes = str(row.get("notes", "") or "").strip()
        if not output_label and not notes and mapped_metric in {"", "Custom"}:
            continue
        if mapped_metric not in WORKBOOK_REVIEW_METRIC_OPTIONS:
            mapped_metric = "Custom"
        cleaned.append(
            {
                "output_label": output_label,
                "mapped_metric": mapped_metric,
                "notes": notes,
            }
        )
    return cleaned


def _clean_workbook_rule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rule_label = str(row.get("rule_label", "") or "").strip()
        mapped_metric = str(row.get("mapped_metric", "Custom") or "Custom").strip()
        operator = str(row.get("operator", "") or "").strip()
        unit = str(row.get("unit", "") or "").strip()
        threshold_raw = row.get("threshold")
        threshold: float | None
        try:
            threshold = None if threshold_raw is None or pd.isna(threshold_raw) else float(threshold_raw)
        except Exception:
            threshold = None
        if not rule_label and mapped_metric in {"", "Custom"} and not operator and threshold is None and not unit:
            continue
        if mapped_metric not in WORKBOOK_REVIEW_METRIC_OPTIONS:
            mapped_metric = "Custom"
        if operator and operator not in WORKBOOK_REVIEW_OPERATOR_OPTIONS:
            operator = ""
        if unit and unit not in WORKBOOK_REVIEW_UNIT_OPTIONS:
            unit = "raw"
        if not unit:
            unit = _default_workbook_unit(mapped_metric)
        cleaned.append(
            {
                "rule_label": rule_label,
                "mapped_metric": mapped_metric,
                "operator": operator,
                "threshold": threshold,
                "unit": unit,
            }
        )
    return cleaned


def _workbook_rows_fingerprint(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, sort_keys=True, default=str)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _abs_path(raw: str | Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _clear_session_paths_for_deleted_target(target: Path, *, keys: list[str]) -> None:
    target_resolved = target.resolve()
    for key in keys:
        raw_value = str(st.session_state.get(key, "") or "").strip()
        if not raw_value:
            continue
        try:
            current_path = _abs_path(raw_value).resolve()
        except Exception:
            continue
        if current_path == target_resolved or (target.is_dir() and _is_relative_to(current_path, target_resolved)):
            st.session_state[key] = ""


def _render_delete_target_ui(
    *,
    label: str,
    raw_path: str,
    allowed_roots: list[Path],
    session_keys_to_clear: list[str],
    widget_prefixes_to_clear: list[str],
    key_prefix: str,
    expect: str = "dir",
) -> None:
    raw_value = str(raw_path or "").strip()
    if not raw_value:
        return

    target = _abs_path(raw_value)
    allowed = any(_is_relative_to(target, root) for root in allowed_roots)

    with st.expander(f"Delete selected {label}", expanded=False):
        st.warning(f"This permanently deletes `{_rel(target)}`.")
        if not allowed:
            st.error("Deletion is only allowed for paths discovered under approved app roots.")
            return

        confirm = st.checkbox(
            f"I understand this will permanently delete the selected {label}.",
            key=f"{key_prefix}_confirm_delete",
        )
        if st.button(f"Delete {label}", key=f"{key_prefix}_delete_button", type="secondary"):
            if not confirm:
                st.error("Tick the confirmation checkbox first.")
                return
            if not target.exists():
                st.error("Selected path no longer exists.")
                return
            if expect == "dir" and not target.is_dir():
                st.error("Selected path is not a directory.")
                return
            if expect == "file" and not target.is_file():
                st.error("Selected path is not a file.")
                return

            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

            _clear_session_paths_for_deleted_target(target, keys=session_keys_to_clear)
            for prefix in widget_prefixes_to_clear:
                st.session_state.pop(f"{prefix}_select", None)
                st.session_state.pop(f"{prefix}_custom", None)
            st.session_state.pop(f"{key_prefix}_confirm_delete", None)
            st.session_state["_results_action_notice"] = f"Deleted `{_rel(target)}`."
            st.cache_data.clear()
            st.rerun()


def _qp_get(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
    except Exception:
        return default
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def _build_query_url(**updates: str | None) -> str:
    params: dict[str, str] = {}
    try:
        for k, v in st.query_params.items():
            if isinstance(v, list):
                params[k] = str(v[0]) if v else ""
            else:
                params[k] = str(v)
    except Exception:
        pass

    for k, v in updates.items():
        if v is None:
            params.pop(k, None)
        else:
            params[k] = str(v)

    return "?" + urlencode(params)


def _reference_link(label: str, ref_type: str, ref_name: str) -> str:
    url = _build_query_url(page="reference", ref_type=ref_type, ref_name=ref_name)
    return f"[{label}]({url})"


def _json_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


def _json_object_or_empty(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except Exception:
        return {}
    return copy.deepcopy(data) if isinstance(data, dict) else {}


def _sync_widget_from_source(widget_key: str, value: Any) -> None:
    source_key = f"_{widget_key}_source"
    if st.session_state.get(source_key) != value:
        st.session_state[widget_key] = value
        st.session_state[source_key] = value


def _value_to_csv_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _parse_csv_numeric_text(raw: str, *, kind: str) -> tuple[bool, Any, str]:
    text = str(raw or "").strip()
    if not text:
        return True, None, "blank (optional)"
    cleaned = text.strip("[]")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return True, None, "blank (optional)"
    try:
        if kind == "int":
            values = [int(p) for p in parts]
        else:
            values = [float(p) for p in parts]
    except Exception:
        label = "integers" if kind == "int" else "numbers"
        return False, None, f"enter comma-separated {label}"
    out: Any = values[0] if len(values) == 1 else values
    detail = "single value" if len(values) == 1 else f"{len(values)} values"
    return True, out, detail


def _build_numeric_range_values(start: float, stop: float, step: float, *, integer: bool) -> list[int] | list[float]:
    if not math.isfinite(start) or not math.isfinite(stop) or not math.isfinite(step):
        raise ValueError("range values must be finite")
    if step <= 0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")
    values: list[int] | list[float] = []
    i = 0
    cur = float(start)
    limit = 1000
    eps = max(abs(step), 1.0) * 1e-9
    while cur <= float(stop) + eps:
        values.append(int(round(cur)) if integer else round(cur, 10))
        i += 1
        if i > limit:
            raise ValueError("range generated too many values")
        cur = float(start) + (i * float(step))
    if not values:
        values = [int(round(start))] if integer else [round(float(start), 10)]
    return values


def _derive_numeric_range_spec(values: Any) -> tuple[float, float, float, bool] | None:
    if not isinstance(values, list) or not values:
        return None
    try:
        numeric = [float(v) for v in values]
    except Exception:
        return None
    integer = all(float(v).is_integer() for v in numeric)
    if len(numeric) == 1:
        step = 1.0 if integer else 0.1
        return float(numeric[0]), float(numeric[0]), float(step), bool(integer)
    diffs = [round(numeric[i + 1] - numeric[i], 10) for i in range(len(numeric) - 1)]
    if any(d <= 0 for d in diffs):
        return None
    step = diffs[0]
    if any(abs(d - step) > 1e-8 for d in diffs[1:]):
        return None
    return float(numeric[0]), float(numeric[-1]), float(step), bool(integer)


def _editable_range_defaults(
    spec: dict[str, Any],
    current_value: Any,
) -> tuple[float, float, float, bool] | None:
    explicit = spec.get("range")
    if isinstance(explicit, dict):
        try:
            return (
                float(explicit["start"]),
                float(explicit["end"]),
                float(explicit["step"]),
                bool(explicit.get("integer", spec.get("integer", False))),
            )
        except Exception:
            pass
    return _derive_numeric_range_spec(current_value)


def _build_strategy_limited_section_contract(
    strategy_cfg: dict[str, Any],
    *,
    section: str,
    default_params: dict[str, Any],
) -> dict[str, Any]:
    limited_cfg = strategy_cfg.get("limited_test", {}) if isinstance(strategy_cfg, dict) else {}
    section_cfg = limited_cfg.get(section, {}) if isinstance(limited_cfg, dict) else {}
    raw_editable = section_cfg.get("optimizable", {}) if isinstance(section_cfg, dict) else {}
    raw_fixed = section_cfg.get("non_optimizable", []) if isinstance(section_cfg, dict) else []

    if isinstance(raw_editable, list):
        raw_editable = {str(key): {} for key in raw_editable}
    if not isinstance(raw_editable, dict):
        raw_editable = {}
    if not isinstance(raw_fixed, list):
        raw_fixed = []

    editable_specs: dict[str, dict[str, Any]] = {}
    for raw_key, raw_spec in raw_editable.items():
        key = str(raw_key or "").strip()
        if not key or key not in default_params:
            continue

        default_value = default_params.get(key)
        integer_default = isinstance(default_value, int) and not isinstance(default_value, bool)
        spec: dict[str, Any] = {
            "key": key,
            "label": key.replace("_", " "),
            "integer": bool(integer_default),
            "default": copy.deepcopy(default_value),
        }
        if isinstance(raw_spec, dict):
            label = str(raw_spec.get("label", "") or "").strip()
            if label:
                spec["label"] = label
            if "integer" in raw_spec:
                spec["integer"] = bool(raw_spec.get("integer"))
            raw_end = raw_spec.get("end", raw_spec.get("stop"))
            if all(name in raw_spec for name in ("start", "step")) and raw_end is not None:
                try:
                    spec["range"] = {
                        "start": float(raw_spec["start"]),
                        "end": float(raw_end),
                        "step": float(raw_spec["step"]),
                        "integer": bool(spec["integer"]),
                    }
                    spec["values"] = _build_numeric_range_values(
                        float(spec["range"]["start"]),
                        float(spec["range"]["end"]),
                        float(spec["range"]["step"]),
                        integer=bool(spec["integer"]),
                    )
                except Exception:
                    pass

        if "values" not in spec:
            continue
        editable_specs[key] = spec

    fixed_keys: list[str] = []
    seen_fixed: set[str] = set()
    for raw_key in raw_fixed:
        key = str(raw_key or "").strip()
        if key in default_params and key not in editable_specs and key not in seen_fixed:
            fixed_keys.append(key)
            seen_fixed.add(key)
    for key in default_params:
        if key not in editable_specs and key not in seen_fixed:
            fixed_keys.append(key)
            seen_fixed.add(key)

    return {
        "has_policy": bool(editable_specs or fixed_keys),
        "editable_specs": editable_specs,
        "fixed_keys": fixed_keys,
        "fixed_params": {key: copy.deepcopy(default_params[key]) for key in fixed_keys},
    }


def _seed_strategy_limited_params(default_params: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    seeded = copy.deepcopy(default_params)
    for key, spec in dict(contract.get("editable_specs", {})).items():
        if key in seeded and isinstance(spec, dict) and "values" in spec:
            seeded[key] = copy.deepcopy(spec["values"])
    return seeded


def _sanitize_strategy_limited_params(
    raw_params: dict[str, Any],
    default_params: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if not contract.get("has_policy", False):
        return copy.deepcopy(raw_params) if raw_params else copy.deepcopy(default_params)

    sanitized = copy.deepcopy(default_params)
    editable_specs = dict(contract.get("editable_specs", {}))
    for key, spec in editable_specs.items():
        if key in raw_params:
            sanitized[key] = copy.deepcopy(raw_params[key])
        elif isinstance(spec, dict) and "values" in spec:
            sanitized[key] = copy.deepcopy(spec["values"])
    return sanitized


def _build_strategy_walkforward_param_contract(
    strategy_cfg: dict[str, Any],
    *,
    params_defaults: dict[str, Any],
) -> dict[str, Any]:
    limited_cfg = strategy_cfg.get("limited_test", {}) if isinstance(strategy_cfg, dict) else {}
    params_defaults = copy.deepcopy(params_defaults) if isinstance(params_defaults, dict) else {}

    sections: dict[str, dict[str, Any]] = {}
    combined_space: dict[str, Any] = {}
    editable_keys_seen: set[str] = set()
    fixed_keys_seen: set[str] = set()

    for section_name in ("entry", "exit"):
        section_cfg = limited_cfg.get(section_name, {}) if isinstance(limited_cfg, dict) else {}
        raw_editable = section_cfg.get("optimizable", {}) if isinstance(section_cfg, dict) else {}
        raw_fixed = section_cfg.get("non_optimizable", []) if isinstance(section_cfg, dict) else []
        if isinstance(raw_editable, list):
            raw_editable = {str(key): {} for key in raw_editable}
        if not isinstance(raw_editable, dict):
            raw_editable = {}
        if not isinstance(raw_fixed, list):
            raw_fixed = []

        editable_specs: dict[str, dict[str, Any]] = {}
        for raw_key, raw_spec in raw_editable.items():
            key = str(raw_key or "").strip()
            if not key or key in editable_keys_seen or key not in params_defaults:
                continue
            default_value = params_defaults.get(key)
            integer_default = isinstance(default_value, int) and not isinstance(default_value, bool)
            spec: dict[str, Any] = {
                "key": key,
                "label": key.replace("_", " "),
                "integer": bool(integer_default),
                "default": copy.deepcopy(default_value),
            }
            if isinstance(raw_spec, dict):
                label = str(raw_spec.get("label", "") or "").strip()
                if label:
                    spec["label"] = label
                if "integer" in raw_spec:
                    spec["integer"] = bool(raw_spec.get("integer"))
                raw_end = raw_spec.get("end", raw_spec.get("stop"))
                if all(name in raw_spec for name in ("start", "step")) and raw_end is not None:
                    try:
                        spec["range"] = {
                            "start": float(raw_spec["start"]),
                            "end": float(raw_end),
                            "step": float(raw_spec["step"]),
                            "integer": bool(spec["integer"]),
                        }
                        spec["values"] = _build_numeric_range_values(
                            float(spec["range"]["start"]),
                            float(spec["range"]["end"]),
                            float(spec["range"]["step"]),
                            integer=bool(spec["integer"]),
                        )
                    except Exception:
                        pass

            if "values" not in spec:
                continue
            editable_specs[key] = spec
            editable_keys_seen.add(key)
            combined_space[key] = copy.deepcopy(spec["values"])

        fixed_params: dict[str, Any] = {}
        for raw_key in raw_fixed:
            key = str(raw_key or "").strip()
            if (
                key
                and key in params_defaults
                and key not in editable_keys_seen
                and key not in fixed_keys_seen
            ):
                fixed_params[key] = copy.deepcopy(params_defaults[key])
                fixed_keys_seen.add(key)
                combined_space[key] = copy.deepcopy(params_defaults[key])

        sections[section_name] = {
            "editable_specs": editable_specs,
            "fixed_params": fixed_params,
        }

    has_policy = any(
        bool(section.get("editable_specs")) or bool(section.get("fixed_params"))
        for section in sections.values()
    )

    return {
        "has_policy": bool(has_policy),
        "sections": sections,
        "default_param_space": combined_space,
    }


def _normalize_walkforward_param_space_value(value: Any) -> Any:
    if isinstance(value, list):
        return copy.deepcopy(value)
    return copy.deepcopy(value)


def _render_walkforward_param_policy_section(
    *,
    title: str,
    section_contract: dict[str, Any],
    param_space: dict[str, Any],
    widget_prefix: str,
    checks: list[tuple[bool, str, str]],
) -> dict[str, Any]:
    working_space = copy.deepcopy(param_space)
    editable_specs = dict(section_contract.get("editable_specs", {}))
    fixed_params = dict(section_contract.get("fixed_params", {}))

    st.markdown(f"**{title}**")
    if editable_specs:
        st.caption(
            "Only the strategy-defined optimisable parameters are editable here. "
            "Locked strategy parameters remain fixed below."
        )
        for key, spec in editable_specs.items():
            label = str(spec.get("label", key.replace("_", " ")))
            integer_mode = bool(spec.get("integer", False))
            value_seed = copy.deepcopy(working_space.get(key, spec.get("values", [])))
            range_spec = _editable_range_defaults(spec, value_seed)
            widget_key_base = f"{widget_prefix}_{key}"

            if range_spec is not None:
                start_default, end_default, step_default, _ = range_spec
                start_seed = int(round(start_default)) if integer_mode else float(start_default)
                end_seed = int(round(end_default)) if integer_mode else float(end_default)
                step_seed = max(1, int(round(step_default))) if integer_mode else float(step_default)
                _sync_widget_from_source(f"{widget_key_base}_start", start_seed)
                _sync_widget_from_source(f"{widget_key_base}_end", end_seed)
                _sync_widget_from_source(f"{widget_key_base}_step", step_seed)
                st.markdown(f"`{key}`")
                range_cols = st.columns(3)
                if integer_mode:
                    start_val = int(range_cols[0].number_input(f"{label} start", step=1, key=f"{widget_key_base}_start"))
                    stop_val = int(range_cols[1].number_input(f"{label} end", step=1, key=f"{widget_key_base}_end"))
                    step_val = int(
                        range_cols[2].number_input(
                            f"{label} step",
                            min_value=1,
                            step=1,
                            key=f"{widget_key_base}_step",
                        )
                    )
                else:
                    number_step = float(step_seed) if math.isfinite(float(step_seed)) and float(step_seed) > 0 else 0.1
                    start_val = float(
                        range_cols[0].number_input(
                            f"{label} start",
                            step=number_step,
                            format="%.4f",
                            key=f"{widget_key_base}_start",
                        )
                    )
                    stop_val = float(
                        range_cols[1].number_input(
                            f"{label} end",
                            step=number_step,
                            format="%.4f",
                            key=f"{widget_key_base}_end",
                        )
                    )
                    step_val = float(
                        range_cols[2].number_input(
                            f"{label} step",
                            min_value=float(number_step / 10.0),
                            step=number_step,
                            format="%.4f",
                            key=f"{widget_key_base}_step",
                        )
                    )
                try:
                    range_values = _build_numeric_range_values(
                        start_val,
                        stop_val,
                        step_val,
                        integer=integer_mode,
                    )
                    working_space[key] = range_values
                    checks.append((True, label, f"{len(range_values)} values"))
                except Exception as e:
                    checks.append((False, label, str(e)))
            else:
                raw_default = _value_to_csv_text(value_seed)
                _sync_widget_from_source(f"{widget_key_base}_values", raw_default)
                raw_text = st.text_input(label, key=f"{widget_key_base}_values")
                value_kind = "int" if integer_mode else "float"
                parse_ok, parsed_value, parse_detail = _parse_csv_numeric_text(raw_text, kind=value_kind)
                checks.append((parse_ok, label, parse_detail))
                if parse_ok and parsed_value is not None:
                    working_space[key] = _normalize_walkforward_param_space_value(parsed_value)
    else:
        st.caption("No optimisable parameters are defined for this section.")

    if fixed_params:
        st.caption("Locked strategy params:")
        st.code(_json_pretty(fixed_params), language="json")
        for key, value in fixed_params.items():
            working_space.setdefault(key, copy.deepcopy(value))

    return working_space


def _estimate_limited_iterations(
    *,
    entry_plugin_name: str | None,
    entry_params_raw: str,
    exit_plugin_name: str | None,
    exit_params_raw: str,
    seed_count_raw: str,
    seed_start_raw: str,
    exit_seed_count_raw: str,
    exit_seed_start_raw: str,
) -> dict[str, Any]:
    from quantbt.experiments.limited.param_grid import (
        build_entry_variants,
        build_exit_param_space,
        total_iterations,
    )
    from quantbt.plugins import load_default_plugins

    def _parse_json_dict(raw_text: str, *, label: str) -> dict[str, Any]:
        text = str(raw_text or "").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception as e:
            raise ValueError(f"{label} invalid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"{label} must be a JSON object")
        return copy.deepcopy(data)

    def _apply_seed_preview(
        params: dict[str, Any],
        *,
        count_raw: str,
        start_raw: str,
        count_label: str,
    ) -> dict[str, Any]:
        count_text = str(count_raw or "").strip()
        if not count_text:
            return params
        try:
            count = int(count_text)
        except Exception as e:
            raise ValueError(f"{count_label} must be an integer") from e
        if count <= 0:
            raise ValueError(f"{count_label} must be > 0")
        if "seed" in params:
            raise ValueError(f"{count_label} conflicts with an existing `seed` in params")
        start_text = str(start_raw or "").strip()
        if start_text:
            try:
                start = int(start_text)
            except Exception as e:
                raise ValueError(f"{count_label} start must be an integer") from e
        else:
            start = 1
        params["seed"] = list(range(int(start), int(start) + int(count)))
        return params

    if not str(entry_plugin_name or "").strip():
        return {"ok": False, "error": "Entry plugin is not resolved."}
    if not str(exit_plugin_name or "").strip():
        return {"ok": False, "error": "Exit plugin is not resolved."}

    try:
        load_default_plugins()
        entry_params = _parse_json_dict(entry_params_raw, label="Entry params")
        exit_params = _parse_json_dict(exit_params_raw, label="Exit params")
        entry_params = _apply_seed_preview(
            entry_params,
            count_raw=seed_count_raw,
            start_raw=seed_start_raw,
            count_label="Seed count",
        )
        exit_params = _apply_seed_preview(
            exit_params,
            count_raw=exit_seed_count_raw,
            start_raw=exit_seed_start_raw,
            count_label="Exit seed count",
        )

        entry_variants, skipped = build_entry_variants(
            [{"name": str(entry_plugin_name).strip(), "params": entry_params}]
        )
        exit_param_space = build_exit_param_space(
            {"name": str(exit_plugin_name).strip(), "params": exit_params}
        )

        entry_variant_total = 1
        for variants in entry_variants:
            entry_variant_total *= len(variants)
        exit_variant_total = len(exit_param_space)
        total = total_iterations(entry_variants, exit_param_space)

        caveats: list[str] = []
        if skipped:
            caveats.append(f"{len(skipped)} invalid entry parameter sets skipped by plugin validation")

        return {
            "ok": True,
            "entry_variants": int(entry_variant_total),
            "exit_variants": int(exit_variant_total),
            "total": int(total),
            "skipped_entry_variants": int(len(skipped)),
            "caveats": caveats,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _parse_extra_args(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return shlex.split(raw)


def _cli_flag_present(args: list[str], flag: str) -> bool:
    return any((a == flag) or a.startswith(f"{flag}=") for a in args)


def _run_cli(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    cur_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{cur_pythonpath}" if cur_pythonpath else str(SRC_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout or ""


def _extract_saved_path(output: str, prefix: str) -> Path | None:
    for line in output.splitlines():
        text = line.strip()
        if text.startswith(prefix):
            raw = text[len(prefix):].strip()
            if not raw:
                return None
            p = Path(raw)
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            return p
    return None


def _extract_path_after_marker(output: str, marker: str) -> Path | None:
    for line in output.splitlines():
        text = line.strip()
        if marker in text:
            raw = text.split(marker, 1)[1].strip()
            if not raw:
                return None
            p = Path(raw)
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            return p
    return None


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _module_file_fingerprint(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    rows: list[tuple[str, int, int]] = []
    for path in sorted(paths):
        try:
            stat = path.stat()
            rows.append((_rel(path), int(stat.st_mtime_ns), int(stat.st_size)))
        except FileNotFoundError:
            rows.append((_rel(path), 0, 0))
    return tuple(rows)


def _strategy_catalog_fingerprint() -> tuple[tuple[str, int, int], ...]:
    strategies_dir = SRC_DIR / "quantbt" / "strategies"
    strategy_files = [p for p in strategies_dir.glob("*.py") if p.name != "__init__.py"]
    return _module_file_fingerprint(strategy_files)


def _run_manifest_fingerprint() -> tuple[tuple[str, int, int], ...]:
    root = RUNS_ROOT / "strategies"
    if not root.exists():
        return ()
    manifest_files = [p for p in root.rglob("manifest.json") if p.is_file()]
    return _module_file_fingerprint(manifest_files)


@st.cache_data(show_spinner=False)
def _discover_strategy_catalog(_fingerprint: tuple[tuple[str, int, int], ...]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    strategies_dir = SRC_DIR / "quantbt" / "strategies"
    importlib.invalidate_caches()

    for py in sorted(strategies_dir.glob("*.py")):
        if py.name == "__init__.py":
            continue

        short = py.stem
        module_path = f"quantbt.strategies.{short}"
        entry: dict[str, Any] = {
            "module": module_path,
            "file": _rel(py),
            "display_name": short,
            "param_space": {},
            "strategy_config": {},
            "params_defaults": {},
            "source": py.read_text(encoding="utf-8"),
            "error": None,
        }

        try:
            if module_path in sys.modules:
                mod = importlib.reload(sys.modules[module_path])
            else:
                mod = importlib.import_module(module_path)
            param_space = getattr(mod, "PARAM_SPACE", {}) or {}
            strategy_cfg = getattr(mod, "STRATEGY", {}) or {}
            params_cls = getattr(mod, "Params", None)

            defaults: dict[str, Any] = {}
            if params_cls is not None and inspect.isclass(params_cls) and is_dataclass(params_cls):
                for f in fields(params_cls):
                    if f.default is not MISSING:
                        defaults[f.name] = f.default

            entry["param_space"] = param_space
            entry["strategy_config"] = strategy_cfg
            entry["params_defaults"] = defaults
            display_name = str(strategy_cfg.get("name", short)).strip() if isinstance(strategy_cfg, dict) else short
            entry["display_name"] = display_name or short
        except Exception as e:
            entry["error"] = str(e)

        catalog[short] = entry

    return catalog


def _strategy_display_name(strategy_key: str, strategy_catalog: dict[str, dict[str, Any]]) -> str:
    key = str(strategy_key or "").strip()
    info = strategy_catalog.get(key, {}) if isinstance(strategy_catalog, dict) else {}
    display_name = str(info.get("display_name", "") or "").strip()
    return display_name or key


def _sorted_strategy_keys(strategy_catalog: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(strategy_catalog.keys(), key=lambda key: _strategy_display_name(key, strategy_catalog).lower())


@st.cache_data(show_spinner=False)
def _dataset_timestamp_bounds(dataset_path: str, ts_col: str) -> dict[str, Any]:
    path = _abs_path(dataset_path)
    meta = read_dataset_meta(path)
    meta_start = None
    meta_end = None
    meta_rows = None
    if isinstance(meta, dict):
        try:
            meta_start = pd.to_datetime(meta.get("start"), errors="coerce")
        except Exception:
            meta_start = None
        try:
            meta_end = pd.to_datetime(meta.get("end"), errors="coerce")
        except Exception:
            meta_end = None
        try:
            meta_rows = int(meta.get("rows", 0) or 0)
        except Exception:
            meta_rows = None

    try:
        df = pd.read_csv(path, usecols=[ts_col])
        ts = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values().reset_index(drop=True)
        if ts.empty:
            raise ValueError(f"No valid timestamps found in `{ts_col}`.")
        values = ts.astype("int64").to_numpy()
        return {
            "ok": True,
            "min_date": ts.iloc[0].date(),
            "max_date": ts.iloc[-1].date(),
            "row_count": int(len(ts)),
            "values_ns": values,
            "meta_start": meta_start,
            "meta_end": meta_end,
        }
    except Exception as e:
        if meta_start is not None and meta_end is not None and meta_rows and meta_rows > 0:
            return {
                "ok": False,
                "min_date": meta_start.date(),
                "max_date": meta_end.date(),
                "row_count": int(meta_rows),
                "values_ns": None,
                "meta_start": meta_start,
                "meta_end": meta_end,
                "error": str(e),
            }
        return {
            "ok": False,
            "min_date": None,
            "max_date": None,
            "row_count": 0,
            "values_ns": None,
            "meta_start": meta_start,
            "meta_end": meta_end,
            "error": str(e),
        }


def _walkforward_window_from_dates(
    *,
    bounds: dict[str, Any],
    start_value: date,
    end_value: date,
) -> dict[str, Any]:
    if start_value > end_value:
        raise ValueError("WFA data start must be on or before WFA data end.")

    values_ns = bounds.get("values_ns")
    if values_ns is None:
        row_count = int(bounds.get("row_count", 0) or 0)
        start_bar = 0
        end_bar = row_count if row_count > 0 else None
        span_days = max((pd.Timestamp(end_value) - pd.Timestamp(start_value)).days, 1)
        selected_rows = row_count
    else:
        arr = np.asarray(values_ns, dtype=np.int64)
        start_ns = pd.Timestamp(start_value).value
        end_exclusive_ns = (pd.Timestamp(end_value) + pd.Timedelta(days=1)).value
        start_bar = int(np.searchsorted(arr, start_ns, side="left"))
        end_bar = int(np.searchsorted(arr, end_exclusive_ns, side="left"))
        selected_rows = int(max(0, end_bar - start_bar))
        if selected_rows <= 0:
            raise ValueError("Selected WFA date range contains no rows.")
        first_ts = pd.Timestamp(int(arr[start_bar]), unit="ns")
        last_ts = pd.Timestamp(int(arr[end_bar - 1]), unit="ns")
        span_days = max(int((last_ts - first_ts).days), 1)

    years = float(span_days / 365.25) if span_days > 0 else float("nan")
    bars_per_year = float(selected_rows / years) if np.isfinite(years) and years > 0 else float("nan")
    return {
        "start_bar": int(start_bar),
        "end_bar": int(end_bar) if end_bar is not None else None,
        "selected_rows": int(selected_rows),
        "bars_per_year": float(bars_per_year) if np.isfinite(bars_per_year) else float("nan"),
        "years": float(years) if np.isfinite(years) else float("nan"),
    }


@st.cache_data(show_spinner=False)
def _discover_run_manifest_rows(_fingerprint: tuple[tuple[str, int, int], ...]) -> list[dict[str, Any]]:
    root = RUNS_ROOT / "strategies"
    if not root.exists():
        return []

    rows: list[dict[str, Any]] = []
    for manifest in sorted(root.rglob("manifest.json")):
        try:
            payload = _read_json(manifest)
        except Exception:
            continue
        run_dir = manifest.parent
        row = {
            "workflow": str(payload.get("workflow", "")).strip(),
            "strategy": str(payload.get("strategy", "")).strip(),
            "category": str(payload.get("category", "")).strip(),
            "category_label": str(payload.get("scenario_label", payload.get("category", ""))).strip(),
            "dataset_slug": str(payload.get("dataset_slug", "")).strip(),
            "date_window": str(payload.get("date_window", "")).strip(),
            "created_at": str(payload.get("created_at", "")).strip(),
            "run_id": str(payload.get("run_id", run_dir.name)).strip(),
            "path": _rel(run_dir),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            str(row.get("created_at", "")),
            str(row.get("run_id", "")),
        ),
        reverse=True,
    )
    return rows


def _build_results_focus_url(*, workflow: str, run_dir: Path) -> str:
    return _build_query_url(
        page="results",
        result_workflow=str(workflow),
        result_run=_rel(run_dir),
        ref_type=None,
        ref_name=None,
        html_file=None,
    )


@st.cache_data(show_spinner=False)
def _discover_plugin_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    from quantbt.plugins.registry import ENTRY_PLUGINS, EXIT_PLUGINS, SIZING_PLUGINS, load_default_plugins

    load_default_plugins()

    def _collect(plugin_map: dict[str, Any], kind: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, fn in sorted(plugin_map.items(), key=lambda kv: kv[0]):
            src_path_raw = inspect.getsourcefile(fn) or ""
            src_path = Path(src_path_raw) if src_path_raw else None
            source = ""
            rel_file = ""
            if src_path is not None and src_path.exists():
                source = src_path.read_text(encoding="utf-8")
                rel_file = _rel(src_path)

            out[name] = {
                "name": name,
                "kind": kind,
                "signature": str(inspect.signature(fn)),
                "doc": inspect.getdoc(fn) or "",
                "file": rel_file,
                "source": source,
            }
        return out

    return {
        "entry": _collect(dict(ENTRY_PLUGINS), "entry"),
        "exit": _collect(dict(EXIT_PLUGINS), "exit"),
        "sizing": _collect(dict(SIZING_PLUGINS), "sizing"),
    }


@st.cache_data(show_spinner=False)
def _discover_download_symbols() -> list[str]:
    cmd = [
        SCRIPT_PYTHON,
        str(SCRIPTS_DIR / "download_data.py"),
        "--provider",
        "dukascopy",
        "--list-symbols",
    ]
    rc, out = _run_cli(cmd)
    if rc != 0:
        return ["EURUSD"]

    symbols: list[str] = []
    for line in out.splitlines():
        sym = line.strip().upper()
        if re.fullmatch(r"[A-Z0-9]{6}", sym):
            symbols.append(sym)

    deduped = sorted(set(symbols))
    return deduped or ["EURUSD"]


def _sort_timeframes(values: list[str]) -> list[str]:
    units = {"S": 1, "M": 60, "H": 3600, "D": 86400, "W": 7 * 86400, "MO": 30 * 86400, "MN": 30 * 86400}

    def _tf_sort_key(tf: str):
        tf = tf.upper()
        m1 = re.fullmatch(r"(\d+)(MO|MN|S|M|H|D|W)", tf)
        if m1:
            n = int(m1.group(1))
            unit = m1.group(2)
            return (0, n * units[unit], n, unit, tf)
        m2 = re.fullmatch(r"(MO|MN|S|M|H|D|W)(\d+)", tf)
        if m2:
            unit = m2.group(1)
            n = int(m2.group(2))
            return (0, n * units[unit], n, unit, tf)
        return (1, tf)

    return sorted(set(values), key=_tf_sort_key)


@st.cache_data(show_spinner=False)
def _discover_download_timeframes() -> list[str]:
    cmd = [
        SCRIPT_PYTHON,
        str(SCRIPTS_DIR / "download_data.py"),
        "--provider",
        "dukascopy",
        "--list-timeframes",
    ]
    rc, out = _run_cli(cmd)
    fallback = ["1H", "4H", "1D"]
    if rc != 0:
        return fallback

    timeframes: list[str] = []
    for line in out.splitlines():
        tf = line.strip().upper()
        if re.fullmatch(r"[A-Z0-9_]+", tf):
            timeframes.append(tf)

    deduped = _sort_timeframes(timeframes)
    return deduped or fallback


@st.cache_data(show_spinner=False)
def _discover_mt5_download_timeframes(mt5_backend: str, mt5_host: str, mt5_port: int) -> list[str]:
    cmd = [
        SCRIPT_PYTHON,
        str(SCRIPTS_DIR / "download_data.py"),
        "--provider",
        "mt5_ftmo",
        "--list-timeframes",
        "--mt5-backend",
        str(mt5_backend).strip(),
        "--mt5-host",
        str(mt5_host).strip(),
        "--mt5-port",
        str(int(mt5_port)),
    ]
    rc, out = _run_cli(cmd)
    fallback = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    if rc != 0:
        return fallback

    timeframes: list[str] = []
    for line in out.splitlines():
        tf = line.strip().upper()
        if re.fullmatch(r"[A-Z0-9_]+", tf):
            timeframes.append(tf)
    deduped = _sort_timeframes(timeframes)
    return deduped or fallback


def _list_csv_data_files() -> list[Path]:
    root = REPO_ROOT / "data"
    if not root.exists():
        return []
    return sorted((p for p in root.rglob("*.csv") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def _discover_run_dirs(*, workflow: str) -> list[Path]:
    root = RUNS_ROOT / "strategies"
    if not root.exists():
        return []

    out: list[Path] = []
    for manifest in root.rglob("manifest.json"):
        try:
            payload = _read_json(manifest)
        except Exception:
            continue
        if str(payload.get("workflow", "")).strip().lower() != str(workflow).strip().lower():
            continue
        out.append(manifest.parent)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def _discover_walkforward_runs() -> list[Path]:
    return _discover_run_dirs(workflow="walkforward")


def _discover_mc_runs() -> list[Path]:
    return _discover_run_dirs(workflow="monte_carlo")


def _discover_limited_runs() -> list[Path]:
    return _discover_run_dirs(workflow="limited")


def _discover_live_trade_files() -> list[Path]:
    candidates: list[Path] = []
    search_roots = [RUNS_ROOT, REPO_ROOT / "data"]
    patterns = ("*live*.csv", "*pilot*.csv", "*paper*.csv")
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(p for p in root.rglob(pattern) if p.is_file())

    seen: set[str] = set()
    unique: list[Path] = []
    for p in sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True):
        key = p.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _select_path_from_options(label: str, options: list[str], default: str, *, key_prefix: str) -> str:
    merged = []
    seen = set()
    for v in [default, *options]:
        if v and v not in seen:
            seen.add(v)
            merged.append(v)

    custom = "<custom path>"
    chosen = st.selectbox(label, [*merged, custom], index=0, key=f"{key_prefix}_select")
    if chosen == custom:
        return st.text_input("Custom path", value=default, key=f"{key_prefix}_custom")
    return chosen


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _evaluate_scalar_rule(actual: Any, op: str, target: Any) -> bool:
    try:
        a = float(actual)
        t = float(target)
    except Exception:
        return False

    if pd.isna(a) or pd.isna(t):
        return False

    if op == ">":
        return a > t
    if op == ">=":
        return a >= t
    if op == "<":
        return a < t
    if op == "<=":
        return a <= t
    if op == "==":
        return a == t
    if op == "!=":
        return a != t
    return False


def _evaluate_series_rule(series: pd.Series, op: str, target: Any) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    t = _as_float(target)

    if op == ">":
        return s > t
    if op == ">=":
        return s >= t
    if op == "<":
        return s < t
    if op == "<=":
        return s <= t
    if op == "==":
        return s == t
    if op == "!=":
        return s != t
    return pd.Series(False, index=series.index)


def _criteria_table(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, bool]:
    df = pd.DataFrame(rows)
    if df.empty:
        return df, False

    df["Status"] = df["Passed"].map(_status_label)
    overall = bool(df["Passed"].all())
    out = df[["Criterion", "Target", "Actual", "Status"]].copy()
    # Arrow serialization in Streamlit is strict about mixed object dtypes.
    # Normalize all display columns to string to avoid intermittent warnings.
    for col in ("Criterion", "Target", "Actual", "Status"):
        out[col] = out[col].astype(str)
    return out, overall


def _median_numeric(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return float("nan")
    return float(s.median())


def _infer_initial_equity_from_spec(spec: dict[str, Any]) -> float:
    cfg = spec.get("config", {}) if isinstance(spec, dict) else {}
    initial_equity = _as_float(cfg.get("initial_equity", 100000.0))
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        return 100000.0
    return float(initial_equity)


def _estimate_montecarlo_parent_context(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {
            "trade_pool_count": 0,
            "historical_years": float("nan"),
            "estimated_trades_per_year": float("nan"),
        }

    trade_pool_count = 0
    historical_years = float("nan")
    oos_trades_path = walkforward_oos_trades_path(run_dir)
    if oos_trades_path.exists():
        try:
            oos_trades = pd.read_csv(oos_trades_path)
            trade_pool_count = int(len(oos_trades))
        except Exception:
            trade_pool_count = 0

    oos_eq_path = walkforward_oos_equity_path(run_dir)
    if oos_eq_path.exists():
        try:
            eq_df = pd.read_csv(oos_eq_path)
            if "time" in eq_df.columns:
                ts = pd.to_datetime(eq_df["time"], utc=True, errors="coerce").dropna()
                if not ts.empty:
                    historical_years = float(years_from_index(pd.DatetimeIndex(ts)))
        except Exception:
            historical_years = float("nan")

    if np.isfinite(historical_years) and historical_years > 0 and trade_pool_count > 0:
        estimated_trades_per_year = float(trade_pool_count / historical_years)
    else:
        estimated_trades_per_year = float("nan")

    return {
        "trade_pool_count": int(trade_pool_count),
        "historical_years": float(historical_years) if np.isfinite(historical_years) else float("nan"),
        "estimated_trades_per_year": float(estimated_trades_per_year) if np.isfinite(estimated_trades_per_year) else float("nan"),
    }


def _limited_net_profit_series(results: pd.DataFrame, *, initial_equity: float) -> pd.Series:
    if "net_profit_abs" in results.columns:
        s = pd.to_numeric(results["net_profit_abs"], errors="coerce")
        return s.replace([np.inf, -np.inf], np.nan)

    if "total_return_%" in results.columns:
        ret = pd.to_numeric(results["total_return_%"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return (ret / 100.0) * float(initial_equity)

    if "final_equity" in results.columns:
        final_eq = pd.to_numeric(results["final_equity"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return final_eq - float(initial_equity)

    return pd.Series(np.nan, index=results.index)


def _load_limited_run_artifacts(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]] | None:
    results_path = limited_iterations_path(run_dir)
    meta_path = spec_path(run_dir)
    pass_path = summary_path(run_dir)
    if not results_path.exists() or not meta_path.exists():
        return None
    try:
        results = pd.read_csv(results_path)
        run_meta = _read_json(meta_path)
        pass_summary = _read_json(pass_path) if pass_path.exists() else {}
        return results, run_meta, pass_summary
    except Exception:
        return None


def _pick_limited_row(results: pd.DataFrame, *, iter_id: int | None = None) -> pd.Series | None:
    if results.empty:
        return None
    if iter_id is not None and "iter" in results.columns:
        mask = pd.to_numeric(results["iter"], errors="coerce") == int(iter_id)
        picked = results.loc[mask]
        if not picked.empty:
            return picked.iloc[0]
    return results.iloc[0]


def _limited_baseline_metrics_from_row(row: pd.Series, *, initial_equity: float) -> dict[str, float]:
    trades = _as_float(row.get("trades"))
    long_trade_pct = _as_float(row.get("long_trade_pct"))
    short_trade_pct = _as_float(row.get("short_trade_pct"))
    avg_bars_held = _as_float(row.get("avg_bars_held"))
    total_return_pct = _as_float(row.get("total_return_%"))
    max_dd_pct = _as_float(row.get("max_drawdown_abs_%"))
    if not np.isfinite(max_dd_pct):
        max_dd_pct = _as_float(row.get("max_drawdown_%"))

    net_profit_abs = _as_float(row.get("net_profit_abs"))
    if not np.isfinite(net_profit_abs):
        final_equity = _as_float(row.get("final_equity"))
        if np.isfinite(final_equity):
            net_profit_abs = float(final_equity - initial_equity)
        elif np.isfinite(total_return_pct):
            net_profit_abs = float((total_return_pct / 100.0) * initial_equity)

    return {
        "trades": float(trades) if np.isfinite(trades) else float("nan"),
        "long_trade_pct": float(long_trade_pct) if np.isfinite(long_trade_pct) else float("nan"),
        "short_trade_pct": float(short_trade_pct) if np.isfinite(short_trade_pct) else float("nan"),
        "avg_bars_held": float(avg_bars_held) if np.isfinite(avg_bars_held) else float("nan"),
        "total_return_%": float(total_return_pct) if np.isfinite(total_return_pct) else float("nan"),
        "max_drawdown_abs_%": float(max_dd_pct) if np.isfinite(max_dd_pct) else float("nan"),
        "net_profit_abs": float(net_profit_abs) if np.isfinite(net_profit_abs) else float("nan"),
    }


def _build_monkey_dominance_criteria(*, baseline_return_pct: float, baseline_max_dd_pct: float) -> dict[str, Any]:
    return {
        "mode": "all",
        "rules": [
            {"metric": "total_return_%", "op": "<", "value": float(baseline_return_pct)},
            {"metric": "max_drawdown_abs_%", "op": ">", "value": float(baseline_max_dd_pct)},
        ],
    }


def _compute_monkey_constrained_dominance(
    *,
    monkey_results: pd.DataFrame,
    baseline_return_pct: float,
    baseline_max_dd_pct: float,
    baseline_trades: float,
    baseline_long_trade_pct: float,
    trade_count_tol_pct: float = 5.0,
    long_ratio_tol_pp: float = 5.0,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "valid_trials": 0,
        "total_trials": int(len(monkey_results)),
        "valid_pct": float("nan"),
        "dominance_valid_pct": float("nan"),
        "trade_low": float("nan"),
        "trade_high": float("nan"),
        "long_low": float("nan"),
        "long_high": float("nan"),
    }
    if monkey_results.empty:
        return out
    if not (
        np.isfinite(baseline_return_pct)
        and np.isfinite(baseline_max_dd_pct)
        and np.isfinite(baseline_trades)
        and baseline_trades > 0
        and np.isfinite(baseline_long_trade_pct)
    ):
        return out

    trades = pd.to_numeric(monkey_results.get("trades", pd.Series(index=monkey_results.index, dtype=float)), errors="coerce")
    long_pct = pd.to_numeric(monkey_results.get("long_trade_pct", pd.Series(index=monkey_results.index, dtype=float)), errors="coerce")
    ret = pd.to_numeric(monkey_results.get("total_return_%", pd.Series(index=monkey_results.index, dtype=float)), errors="coerce")
    dd = pd.to_numeric(monkey_results.get("max_drawdown_abs_%", pd.Series(index=monkey_results.index, dtype=float)), errors="coerce")
    if "max_drawdown_abs_%" not in monkey_results.columns:
        dd = pd.to_numeric(monkey_results.get("max_drawdown_%", pd.Series(index=monkey_results.index, dtype=float)), errors="coerce")

    trade_low = float(baseline_trades * (1.0 - float(trade_count_tol_pct) / 100.0))
    trade_high = float(baseline_trades * (1.0 + float(trade_count_tol_pct) / 100.0))
    long_low = float(baseline_long_trade_pct - float(long_ratio_tol_pp))
    long_high = float(baseline_long_trade_pct + float(long_ratio_tol_pp))

    valid_mask = (
        trades.notna()
        & long_pct.notna()
        & ret.notna()
        & dd.notna()
        & (trades >= trade_low)
        & (trades <= trade_high)
        & (long_pct >= long_low)
        & (long_pct <= long_high)
    )
    valid_trials = int(valid_mask.sum())
    out["valid_trials"] = valid_trials
    out["valid_pct"] = float((valid_trials / len(monkey_results)) * 100.0) if len(monkey_results) else float("nan")
    out["trade_low"] = trade_low
    out["trade_high"] = trade_high
    out["long_low"] = long_low
    out["long_high"] = long_high

    if valid_trials <= 0:
        return out

    dom_mask = valid_mask & (ret < float(baseline_return_pct)) & (dd > float(baseline_max_dd_pct))
    dominance_valid_pct = float((dom_mask.sum() / valid_trials) * 100.0)
    out["dominance_valid_pct"] = dominance_valid_pct
    return out


def _is_monkey_limited_run(run_meta: dict[str, Any]) -> bool:
    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    strategy_cfg = spec.get("strategy", {}) if isinstance(spec, dict) else {}
    entry_cfg = strategy_cfg.get("entry", {}) if isinstance(strategy_cfg, dict) else {}
    exit_cfg = strategy_cfg.get("exit", {}) if isinstance(strategy_cfg, dict) else {}

    rules = entry_cfg.get("rules", []) if isinstance(entry_cfg, dict) else []
    entry_plugins = {
        str(r.get("name", "")).strip()
        for r in rules
        if isinstance(r, dict) and str(r.get("name", "")).strip()
    }
    exit_plugin = str(exit_cfg.get("name", "")).strip()
    monkey_plugins = {"monkey_entry", "monkey_exit", "random", "random_time_exit"}
    return bool(entry_plugins.intersection(monkey_plugins) or exit_plugin in monkey_plugins)


def _classify_limited_core_decision(
    *,
    profitable_runs: int,
    median_profit: float,
    median_trades: float,
    best_over_median: float,
) -> str:
    pass_ok = (
        profitable_runs >= 70
        and median_profit > 0
        and median_trades >= 75
        and best_over_median <= 3.0
    )
    retry_ok = (
        (50 <= profitable_runs <= 69)
        or (30 <= median_trades <= 74)
        or (median_profit > 0 and 3.0 < best_over_median <= 5.0)
    )
    fail_ok = (
        profitable_runs < 50
        or median_profit <= 0
        or median_trades < 30
        or best_over_median > 5.0
    )
    if pass_ok:
        return "PASS"
    if fail_ok:
        return "FAIL"
    if retry_ok:
        return "RETRY"
    return "FAIL"


def _classify_monkey_dominance_decision(dominance_pct: float) -> str:
    if np.isfinite(dominance_pct) and dominance_pct >= 85.0:
        return "PASS"
    if np.isfinite(dominance_pct) and dominance_pct >= 75.0:
        return "RETRY"
    return "FAIL"


def _classify_walkforward_decision(*, segments: int, profitable_oos_pct: float, top2_share_pct: float) -> str:
    if (
        segments >= 8
        and np.isfinite(profitable_oos_pct)
        and np.isfinite(top2_share_pct)
        and profitable_oos_pct >= 60.0
        and top2_share_pct <= 50.0
    ):
        return "PASS"
    if (
        segments < 8
        or (np.isfinite(profitable_oos_pct) and profitable_oos_pct < 50.0)
        or (np.isfinite(top2_share_pct) and top2_share_pct > 60.0)
        or not np.isfinite(profitable_oos_pct)
        or not np.isfinite(top2_share_pct)
    ):
        return "FAIL"
    if (
        (np.isfinite(profitable_oos_pct) and 50.0 <= profitable_oos_pct < 60.0)
        or (np.isfinite(top2_share_pct) and 50.0 < top2_share_pct <= 60.0)
    ):
        return "RETRY"
    return "FAIL"


def _classify_monte_carlo_risk_decision(*, dd_ratio: float, original_profit: float, profit_after_removal: float) -> str:
    dd_pass = np.isfinite(dd_ratio) and dd_ratio <= 1.7
    dd_retry = np.isfinite(dd_ratio) and 1.7 < dd_ratio <= 2.0
    dd_fail = (not np.isfinite(dd_ratio)) or dd_ratio > 2.0

    outlier_pass = np.isfinite(profit_after_removal) and profit_after_removal > 0.0
    if np.isfinite(original_profit) and original_profit > 0:
        outlier_retry = (
            np.isfinite(profit_after_removal)
            and profit_after_removal <= 0.0
            and profit_after_removal > (-0.10 * original_profit)
        )
        outlier_fail = (not np.isfinite(profit_after_removal)) or profit_after_removal <= (-0.10 * original_profit)
    else:
        outlier_retry = False
        outlier_fail = (not np.isfinite(profit_after_removal)) or profit_after_removal <= 0.0

    if dd_pass and outlier_pass:
        return "PASS"
    if dd_fail or outlier_fail:
        return "FAIL"
    if dd_retry or outlier_retry:
        return "RETRY"
    return "FAIL"


def _classify_live_pilot_decision(*, live_net_profit: float, band_p5: float, band_p95: float, initial_equity: float) -> str:
    if not (np.isfinite(live_net_profit) and np.isfinite(band_p5) and np.isfinite(band_p95) and np.isfinite(initial_equity)):
        return "FAIL"
    if band_p5 <= live_net_profit <= band_p95:
        return "PASS"

    distance = (band_p5 - live_net_profit) if live_net_profit < band_p5 else (live_net_profit - band_p95)
    if distance > (0.05 * initial_equity):
        return "FAIL"
    return "RETRY"


def _strategy_run_result_label(*, workflow: str, run_dir: Path) -> str:
    workflow = str(workflow or "").strip().lower()
    try:
        if workflow == "limited":
            summary = load_limited_summary(run_dir)
            return str(summary.get("decision", "n/a")).upper()
        if workflow == "walkforward":
            summary = load_walkforward_summary(run_dir)
            decision = _classify_walkforward_decision(
                segments=int(summary.get("segment_count", 0) or 0),
                profitable_oos_pct=_as_float(summary.get("profitable_pct")),
                top2_share_pct=_as_float(summary.get("top2_share_pct")),
            )
            return str(decision).upper()
        if workflow == "monte_carlo":
            summary = load_montecarlo_summary(run_dir)
            checks = summary.get("checks", {}) if isinstance(summary.get("checks", {}), dict) else {}
            return "PASS" if bool(checks.get("all_passed")) else "FAIL"
        if workflow == "optimize":
            opt_summary_path = summary_path(run_dir)
            if not opt_summary_path.exists():
                return "MISSING"
            summary = _read_json(opt_summary_path)
            status = str(summary.get("status", "ok")).strip().upper()
            if status == "OK":
                return "OK"
            if status == "EMPTY_RESULTS":
                return "EMPTY"
            return status or "OK"
    except FileNotFoundError:
        return "MISSING"
    except Exception:
        return "ERROR"
    return "N/A"


def _annotate_strategy_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        run_dir = _abs_path(str(row.get("path", "")).strip())
        workflow = str(row.get("workflow", "")).strip().lower()
        item["result"] = _strategy_run_result_label(workflow=workflow, run_dir=run_dir)
        if workflow in {"limited", "walkforward", "monte_carlo"}:
            item["result_url"] = _build_results_focus_url(workflow=workflow, run_dir=run_dir)
        else:
            item["result_url"] = ""
        out.append(item)
    return out


RESULT_BROWSER_WORKFLOW_LABELS: dict[str, str] = {
    "limited": "Limited tests",
    "walkforward": "Walk-forward",
    "monte_carlo": "Monte Carlo",
}

RESULT_BROWSER_LEVELS: list[tuple[str, str]] = [
    ("strategy", "Strategies"),
    ("workflow", "Workflows"),
    ("category_label", "Scenarios / profiles"),
    ("dataset_slug", "Datasets"),
    ("date_window", "Date windows"),
    ("path", "Runs"),
]

RESULT_BROWSER_STATE_KEYS: dict[str, str] = {
    "strategy": "results_browser_strategy",
    "workflow": "results_browser_workflow",
    "category_label": "results_browser_category",
    "dataset_slug": "results_browser_dataset",
    "date_window": "results_browser_window",
    "path": "results_browser_run",
}


def _workflow_browser_label(workflow: str) -> str:
    key = str(workflow or "").strip().lower()
    return RESULT_BROWSER_WORKFLOW_LABELS.get(key, key.replace("_", " ").title() or "Unknown")


def _ordered_manifest_values(rows: list[dict[str, Any]], field: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        value = str(row.get(field, "") or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _select_browser_value(
    *,
    label: str,
    options: list[str],
    key: str,
    format_func: Callable[[str], str] | None = None,
    help: str | None = None,
) -> str:
    cleaned = [str(option).strip() for option in options if str(option).strip()]
    if not cleaned:
        st.session_state[key] = ""
        return ""
    current = str(st.session_state.get(key, "") or "").strip()
    if current not in cleaned:
        current = cleaned[0]
        st.session_state[key] = current
    return str(
        st.selectbox(
            label,
            cleaned,
            key=key,
            format_func=format_func if format_func is not None else (lambda value: value),
            help=help,
        )
    )


def _results_browser_state_key(field: str) -> str:
    return RESULT_BROWSER_STATE_KEYS[field]


def _clear_results_browser_from(level_index: int) -> None:
    for idx, (field, _) in enumerate(RESULT_BROWSER_LEVELS):
        if idx >= int(level_index):
            st.session_state.pop(_results_browser_state_key(field), None)


def _set_results_browser_value(field: str, value: str) -> None:
    for idx, (level_field, _) in enumerate(RESULT_BROWSER_LEVELS):
        if level_field == field:
            st.session_state[_results_browser_state_key(field)] = str(value or "").strip()
            _clear_results_browser_from(idx + 1)
            return


def _set_results_browser_to_row(row: dict[str, Any]) -> None:
    for field, _ in RESULT_BROWSER_LEVELS:
        st.session_state[_results_browser_state_key(field)] = str(row.get(field, "") or "").strip()


def _results_browser_button_token(*parts: Any) -> str:
    raw = "__".join(str(part or "").strip() for part in parts if str(part or "").strip())
    token = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return token[:120] or "item"


def _format_results_browser_value(field: str, value: str, row: dict[str, Any] | None = None) -> str:
    text = str(value or "").strip()
    if field == "workflow":
        return _workflow_browser_label(text)
    if field == "path":
        if row is not None:
            run_id = str(row.get("run_id", "") or "").strip()
            result = str(row.get("result", "") or "").strip()
            created = str(row.get("created_at", "") or "").strip()
            pieces = [piece for piece in [run_id, result, created] if piece]
            if pieces:
                return " | ".join(pieces)
        return Path(text).name or text
    return text


def _build_results_browser_entries(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for value in _ordered_manifest_values(rows, field):
        group = [row for row in rows if str(row.get(field, "") or "").strip() == value]
        if not group:
            continue
        latest = group[0]
        is_run = field == "path"
        entries.append(
            {
                "value": value,
                "label": _format_results_browser_value(field, value, latest),
                "kind": "Run" if is_run else "Folder",
                "count": len(group),
                "latest_created": str(latest.get("created_at", "") or "").strip(),
                "latest_result": str(latest.get("result", "") or "").strip(),
                "summary": _rel(_abs_path(value)) if is_run else "",
                "row": latest,
            }
        )
    return entries


def _render_results_browser_breadcrumb(selected_parts: list[tuple[str, str]]) -> None:
    crumbs: list[tuple[str, int, str]] = [("Runs", -1, "root")]
    for idx, (field, value) in enumerate(selected_parts):
        crumbs.append((_format_results_browser_value(field, value), idx, field))
    crumb_cols = st.columns(len(crumbs))
    for idx, (label, level_idx, field) in enumerate(crumbs):
        if crumb_cols[idx].button(
            label,
            key=f"results_browser_crumb_{idx}_{_results_browser_button_token(field, label)}",
            use_container_width=True,
        ):
            _clear_results_browser_from(level_idx + 1)
            st.rerun()


def _render_results_browser_entries(
    *,
    entries: list[dict[str, Any]],
    field: str,
    current_folder_label: str,
    selected_value: str = "",
) -> None:
    st.markdown(f"**{current_folder_label}**")
    if not entries:
        st.info("No items found in this folder.")
        return

    header_cols = st.columns([0.34, 0.11, 0.10, 0.15, 0.18, 0.12])
    header_cols[0].markdown("**Name**")
    header_cols[1].markdown("**Type**")
    header_cols[2].markdown("**Runs**")
    header_cols[3].markdown("**Latest Result**")
    header_cols[4].markdown("**Latest Created**")
    header_cols[5].markdown("**Action**")

    for idx, entry in enumerate(entries):
        row = entry.get("row", {}) if isinstance(entry.get("row", {}), dict) else {}
        value = str(entry.get("value", "") or "").strip()
        cols = st.columns([0.34, 0.11, 0.10, 0.15, 0.18, 0.12])
        selected_marker = " [selected]" if selected_value and value == selected_value else ""
        cols[0].markdown(f"`{entry.get('label', '')}`{selected_marker}")
        if entry.get("summary"):
            cols[0].caption(str(entry.get("summary", "")))
        cols[1].write(str(entry.get("kind", "")))
        cols[2].write(str(entry.get("count", "")))
        cols[3].write(str(entry.get("latest_result", "")))
        cols[4].write(str(entry.get("latest_created", "")))

        action_label = "Open" if field != "path" else "View"
        if field == "path" and selected_value and value == selected_value:
            action_label = "Selected"
        if cols[5].button(
            action_label,
            key=f"results_browser_open_{_results_browser_button_token(field, value, idx)}",
            use_container_width=True,
            disabled=bool(field == "path" and selected_value and value == selected_value),
        ):
            if field == "path":
                _set_results_browser_to_row(row)
            else:
                _set_results_browser_value(field, value)
            st.rerun()


def _mc_profit_distribution_from_pool(pool: np.ndarray, *, n_trades: int, n_sims: int, seed: int) -> np.ndarray:
    if pool.size == 0 or n_trades <= 0 or n_sims <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, pool.size, size=(int(n_sims), int(n_trades)))
    sampled = pool[idx]
    return sampled.sum(axis=1).astype(float)


def _max_drawdown_pct_from_equity_series(equity: pd.Series) -> float:
    s = pd.to_numeric(equity, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return float("nan")
    peak = s.cummax()
    dd = (peak - s) / peak
    return float(dd.max() * 100.0)


def _parse_params_cell(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _extract_numeric_param(raw: Any, keys: list[str]) -> float:
    parsed = _parse_params_cell(raw)
    candidates: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        candidates = [parsed]
    elif isinstance(parsed, list):
        candidates = [x for x in parsed if isinstance(x, dict)]

    for cand in candidates:
        for k in keys:
            if k in cand:
                v = _as_float(cand.get(k))
                if pd.notna(v):
                    return float(v)
    return float("nan")


def _compact_params_cell(raw: Any) -> str:
    parsed = _parse_params_cell(raw)
    if isinstance(parsed, (dict, list)):
        try:
            return json.dumps(parsed, sort_keys=True, separators=(", ", ": "), default=str)
        except Exception:
            return str(parsed)
    text = str(raw or "").strip()
    return text


def _time_window_label(start_value: Any, end_value: Any) -> str:
    start_text = str(start_value or "").strip()
    end_text = str(end_value or "").strip()
    if start_text and end_text:
        return f"{start_text} -> {end_text}"
    if start_text:
        return start_text
    if end_text:
        return end_text
    return ""


def _add_walkforward_fold_annotations(fig: Any, folds: pd.DataFrame) -> None:
    if folds is None or folds.empty or not hasattr(fig, "add_annotation"):
        return

    schedule_rows: list[dict[str, Any]] = []
    for _, row in folds.iterrows():
        start = pd.to_datetime(row.get("oos_start_time"), utc=True, errors="coerce")
        end = pd.to_datetime(row.get("oos_end_time_exclusive"), utc=True, errors="coerce")
        if pd.isna(start):
            continue
        if pd.isna(end) or end < start:
            end = start
        schedule_rows.append(
            {
                "fold": int(row.get("fold")) if pd.notna(row.get("fold")) else None,
                "start": start,
                "end": end,
            }
        )

    if not schedule_rows:
        return

    long_labels = len(schedule_rows) <= 12
    for i, item in enumerate(schedule_rows):
        start = item["start"]
        end = item["end"]
        fold_num = item.get("fold")
        fill = "rgba(148,163,184,0.08)" if i % 2 == 0 else "rgba(59,130,246,0.06)"
        line = "rgba(71,85,105,0.45)" if i % 2 == 0 else "rgba(30,64,175,0.38)"
        fig.add_vline(x=start, line_width=1, line_dash="dot", line_color=line)
        if end > start:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=fill,
                line_width=0,
                layer="below",
            )
        midpoint = start + (end - start) / 2 if end > start else start
        label = f"Fold {fold_num}" if long_labels else f"F{fold_num}"
        fig.add_annotation(
            x=midpoint,
            y=1.03 if i % 2 == 0 else 1.10,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            font={"size": 10, "color": "#475569"},
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(148,163,184,0.45)",
            borderwidth=1,
            borderpad=3,
        )


def _iqr_numeric(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return float("nan")
    q75 = float(s.quantile(0.75))
    q25 = float(s.quantile(0.25))
    return q75 - q25


def _render_limited_atr_robustness(results: pd.DataFrame) -> None:
    if results.empty or "exit_params" not in results.columns:
        return

    work = results.copy()
    work["_sldist_atr_mult"] = work["exit_params"].apply(
        lambda x: _extract_numeric_param(x, ["sldist_atr_mult", "sldist_atr"])
    )
    work["_rr"] = work["exit_params"].apply(lambda x: _extract_numeric_param(x, ["rr"]))

    work = work.loc[pd.to_numeric(work["_sldist_atr_mult"], errors="coerce").notna()].copy()
    if work.empty:
        return

    rr_unique = sorted(set(pd.to_numeric(work["_rr"], errors="coerce").dropna().astype(float).tolist()))
    group_cols = ["_sldist_atr_mult"] if len(rr_unique) <= 1 else ["_rr", "_sldist_atr_mult"]

    rows: list[dict[str, Any]] = []
    grouped = work.groupby(group_cols, dropna=False)
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        out: dict[str, Any] = {}
        if len(group_cols) == 1:
            out["sldist_atr_mult"] = float(key[0])
        else:
            out["rr"] = float(key[0])
            out["sldist_atr_mult"] = float(key[1])

        out["iters"] = int(len(g))
        if "favourable" in g.columns:
            fav_series = g["favourable"].astype(bool)
            out["favourable_%"] = float(fav_series.mean() * 100.0)
        else:
            out["favourable_%"] = float("nan")

        out["median_return_%"] = _median_numeric(g, "total_return_%")
        out["median_max_dd_%"] = _median_numeric(g, "max_drawdown_abs_%")
        out["median_mar"] = _median_numeric(g, "mar")
        out["median_sortino"] = _median_numeric(g, "sortino")
        out["return_iqr_%"] = _iqr_numeric(g, "total_return_%")

        ret_s = pd.to_numeric(g.get("total_return_%", pd.Series(dtype=float)), errors="coerce").dropna()
        out["negative_return_%"] = float((ret_s < 0).mean() * 100.0) if not ret_s.empty else float("nan")

        rows.append(out)

    rob = pd.DataFrame(rows)
    if rob.empty:
        return

    # Composite robustness rank: high favourable/return, low drawdown/dispersion.
    rank_cols: list[pd.Series] = []
    for col, asc in (
        ("favourable_%", False),
        ("median_return_%", False),
        ("median_max_dd_%", True),
        ("return_iqr_%", True),
    ):
        s = pd.to_numeric(rob[col], errors="coerce")
        rank_cols.append(s.rank(method="min", ascending=asc, na_option="bottom"))
    rob["robust_rank"] = pd.concat(rank_cols, axis=1).mean(axis=1)
    rob = rob.sort_values(["robust_rank", "iters", "favourable_%"], ascending=[True, False, False]).reset_index(drop=True)

    best = rob.iloc[0]
    if len(group_cols) == 1:
        best_label = f"sldist_atr_mult={float(best['sldist_atr_mult']):.3f}"
    else:
        best_label = f"rr={float(best['rr']):.3f}, sldist_atr_mult={float(best['sldist_atr_mult']):.3f}"

    st.subheader("ATR Exit Robustness")
    if len(group_cols) == 1:
        st.caption(
            "Grouped by `sldist_atr_mult` to identify a stable fixed stop-distance multiplier "
            "for entry-quality testing."
        )
    else:
        st.caption(
            "RR varies in this run, so robustness is grouped by `(rr, sldist_atr_mult)` pairs."
        )
    st.info(f"Suggested robust setting: `{best_label}` (lowest composite robustness rank).")

    display_cols = [c for c in ["rr", "sldist_atr_mult", "iters", "favourable_%", "median_return_%", "median_max_dd_%", "median_mar", "median_sortino", "return_iqr_%", "negative_return_%", "robust_rank"] if c in rob.columns]
    st.dataframe(rob[display_cols], use_container_width=True, hide_index=True)


def _render_limited_hold_bars_robustness(results: pd.DataFrame) -> None:
    if results.empty or "exit_params" not in results.columns:
        return

    work = results.copy()
    work["_hold_bars"] = work["exit_params"].apply(lambda x: _extract_numeric_param(x, ["hold_bars"]))
    work = work.loc[pd.to_numeric(work["_hold_bars"], errors="coerce").notna()].copy()
    if work.empty:
        return

    rows: list[dict[str, Any]] = []
    grouped = work.groupby(["_hold_bars"], dropna=False)
    for key, g in grouped:
        hold_bars = float(key[0] if isinstance(key, tuple) else key)
        out: dict[str, Any] = {
            "hold_bars": hold_bars,
            "iters": int(len(g)),
        }

        if "favourable" in g.columns:
            fav_series = g["favourable"].astype(bool)
            out["favourable_%"] = float(fav_series.mean() * 100.0)
        else:
            out["favourable_%"] = float("nan")

        out["median_return_%"] = _median_numeric(g, "total_return_%")
        out["median_max_dd_%"] = _median_numeric(g, "max_drawdown_abs_%")
        out["median_mar"] = _median_numeric(g, "mar")
        out["median_sortino"] = _median_numeric(g, "sortino")
        out["return_iqr_%"] = _iqr_numeric(g, "total_return_%")

        ret_s = pd.to_numeric(g.get("total_return_%", pd.Series(dtype=float)), errors="coerce").dropna()
        out["negative_return_%"] = float((ret_s < 0).mean() * 100.0) if not ret_s.empty else float("nan")

        rows.append(out)

    rob = pd.DataFrame(rows)
    if rob.empty:
        return

    rank_cols: list[pd.Series] = []
    for col, asc in (
        ("favourable_%", False),
        ("median_return_%", False),
        ("median_max_dd_%", True),
        ("return_iqr_%", True),
    ):
        s = pd.to_numeric(rob[col], errors="coerce")
        rank_cols.append(s.rank(method="min", ascending=asc, na_option="bottom"))
    rob["robust_rank"] = pd.concat(rank_cols, axis=1).mean(axis=1)
    rob = rob.sort_values(["robust_rank", "iters", "favourable_%"], ascending=[True, False, False]).reset_index(drop=True)

    best = rob.iloc[0]
    best_hold = float(best["hold_bars"])
    best_label = f"hold_bars={int(round(best_hold))}" if abs(best_hold - round(best_hold)) < 1e-9 else f"hold_bars={best_hold:.3f}"

    st.subheader("Fixed Bar Exit Robustness")
    st.caption(
        "Grouped by `hold_bars` to identify a stable fixed holding period for entry-quality testing."
    )
    st.info(f"Suggested robust setting: `{best_label}` (lowest composite robustness rank).")

    display_cols = [c for c in ["hold_bars", "iters", "favourable_%", "median_return_%", "median_max_dd_%", "median_mar", "median_sortino", "return_iqr_%", "negative_return_%", "robust_rank"] if c in rob.columns]
    st.dataframe(rob[display_cols], use_container_width=True, hide_index=True)


def _render_limited_test_param_space(run_meta: dict[str, Any]) -> None:
    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    strategy_cfg = spec.get("strategy", {}) if isinstance(spec, dict) else {}
    entry_cfg = strategy_cfg.get("entry", {}) if isinstance(strategy_cfg, dict) else {}
    exit_cfg = strategy_cfg.get("exit", {}) if isinstance(strategy_cfg, dict) else {}
    sizing_cfg = strategy_cfg.get("sizing", {}) if isinstance(strategy_cfg, dict) else {}

    entry_rules = entry_cfg.get("rules", []) if isinstance(entry_cfg, dict) else []
    if not isinstance(entry_rules, list):
        entry_rules = []

    st.subheader("Test Parameter Space")
    st.caption("Exact strategy/parameter space used for this run (from `spec.json`).")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Entry**")
        if entry_rules:
            entry_plugin_names = [str(r.get("name", "")) for r in entry_rules if isinstance(r, dict)]
            st.caption(
                f"Mode: `{entry_cfg.get('mode', 'all')}`"
                + (f" | Vote-k: `{entry_cfg.get('vote_k')}`" if entry_cfg.get("vote_k") is not None else "")
            )
            st.code(_json_pretty({"rules": entry_rules, "plugins": entry_plugin_names}), language="json")
        else:
            st.info("No entry rules found in run metadata.")

    with c2:
        st.markdown("**Exit**")
        if isinstance(exit_cfg, dict) and exit_cfg:
            st.caption(f"Plugin: `{exit_cfg.get('name', '')}`")
            st.code(_json_pretty(exit_cfg.get("params", {})), language="json")
        else:
            st.info("No exit config found in run metadata.")

    with c3:
        st.markdown("**Sizing**")
        if isinstance(sizing_cfg, dict) and sizing_cfg:
            st.caption(f"Plugin: `{sizing_cfg.get('name', '')}`")
            st.code(_json_pretty(sizing_cfg.get("params", {})), language="json")
        else:
            st.info("No sizing config found in run metadata.")


def _style_rows_by_metric(df: pd.DataFrame, metric_col: str) -> pd.io.formats.style.Styler:
    def _row_style(row: pd.Series) -> list[str]:
        try:
            v = float(pd.to_numeric(pd.Series([row.get(metric_col)]), errors="coerce").iloc[0])
        except Exception:
            v = float("nan")
        if pd.isna(v):
            bg = ""
        elif v > 0:
            bg = "background-color: rgba(22,163,74,0.12);"
        elif v < 0:
            bg = "background-color: rgba(220,38,38,0.12);"
        else:
            bg = ""
        return [bg] * len(row)

    return df.style.apply(_row_style, axis=1)


def _style_rows_by_verdict(df: pd.DataFrame, verdict_col: str = "Verdict") -> pd.io.formats.style.Styler:
    def _verdict_style(value: Any) -> str:
        verdict = str(value).strip().upper()
        if verdict == "PASS":
            return "background-color: rgba(22,163,74,0.16); color: #166534; font-weight: 700;"
        if verdict == "RETRY":
            return "background-color: rgba(245,158,11,0.18); color: #92400e; font-weight: 700;"
        if verdict == "FAIL":
            return "background-color: rgba(220,38,38,0.16); color: #991b1b; font-weight: 700;"
        if verdict in {"INFO", "N/A"}:
            return "background-color: rgba(100,116,139,0.12); color: #475569; font-weight: 600;"
        return ""

    if verdict_col not in df.columns:
        return df.style

    return df.style.apply(
        lambda col: [_verdict_style(v) for v in col],
        subset=[verdict_col],
        axis=0,
    )


def _render_win_loss_dataframe(
    df: pd.DataFrame,
    *,
    metric_col: str,
    hide_index: bool = True,
    max_styled_rows: int = 2500,
) -> None:
    if metric_col in df.columns and len(df) <= max_styled_rows:
        st.dataframe(_style_rows_by_metric(df, metric_col), use_container_width=True, hide_index=hide_index)
        return

    st.dataframe(df, use_container_width=True, hide_index=hide_index)
    if metric_col in df.columns and len(df) > max_styled_rows:
        st.caption(
            f"Row coloring disabled for large table ({len(df)} rows) to keep UI responsive."
        )


def _render_verdict_dataframe(
    df: pd.DataFrame,
    *,
    verdict_col: str = "Verdict",
    hide_index: bool = True,
    max_styled_rows: int = 2500,
) -> None:
    if verdict_col in df.columns and len(df) <= max_styled_rows:
        st.dataframe(_style_rows_by_verdict(df, verdict_col=verdict_col), use_container_width=True, hide_index=hide_index)
        return

    st.dataframe(df, use_container_width=True, hide_index=hide_index)
    if verdict_col in df.columns and len(df) > max_styled_rows:
        st.caption(
            f"Verdict coloring disabled for large table ({len(df)} rows) to keep UI responsive."
        )


def _render_histogram(series: pd.Series, *, title: str, bins: int = 60) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        st.info(f"No numeric data for: {title}")
        return

    vals = s.to_numpy(dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    bin_count = max(int(bins), 1)

    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.01, 1e-6)
        edges = np.array([vmin - pad, vmin + pad], dtype=float)
        bin_width = float(edges[1] - edges[0])
    else:
        raw_width = (vmax - vmin) / float(bin_count)
        magnitude = 10.0 ** math.floor(math.log10(raw_width))
        normalized = raw_width / magnitude
        if normalized <= 1.0:
            nice_step = 1.0
        elif normalized <= 2.0:
            nice_step = 2.0
        elif normalized <= 2.5:
            nice_step = 2.5
        elif normalized <= 5.0:
            nice_step = 5.0
        else:
            nice_step = 10.0
        bin_width = nice_step * magnitude
        start = math.floor(vmin / bin_width) * bin_width
        end = math.ceil(vmax / bin_width) * bin_width
        steps = max(int(round((end - start) / bin_width)), 1)
        edges = start + np.arange(steps + 1, dtype=float) * bin_width
        if edges[-1] < vmax:
            edges = np.append(edges, edges[-1] + bin_width)

    counts, _ = np.histogram(vals, bins=edges)
    if bin_width <= 0 or not np.isfinite(bin_width):
        precision = 4
    elif bin_width >= 1.0:
        precision = 2
    else:
        precision = int(min(8, max(2, math.ceil(-math.log10(bin_width)) + 2)))

    def _fmt(x: float) -> str:
        if abs(x) < 10 ** (-precision):
            x = 0.0
        return f"{x:.{precision}f}"

    labels = [f"[{_fmt(edges[i])}, {_fmt(edges[i + 1])})" for i in range(len(edges) - 1)]
    if labels:
        labels[-1] = f"[{_fmt(edges[-2])}, {_fmt(edges[-1])}]"

    hist_df = pd.DataFrame(
        {
            "bin": labels,
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "count": counts.astype(int),
        }
    )
    st.markdown(f"**{title}**")
    if len(hist_df) > 0:
        st.caption(
            f"Bin width: {bin_width:.{precision}f} ({len(hist_df)} equal intervals)"
        )

    if px is not None:
        fig = px.bar(hist_df, x="bin", y="count", title=title)
        fig.update_layout(
            xaxis_title="Value range",
            yaxis_title="Count",
            xaxis={"categoryorder": "array", "categoryarray": labels},
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    if alt is not None:
        chart = (
            alt.Chart(hist_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "bin:N",
                    title="Value range",
                    sort=alt.SortField(field="bin_start", order="ascending"),
                    axis=alt.Axis(labelAngle=-90),
                ),
                y=alt.Y("count:Q", title="Count"),
                tooltip=[
                    alt.Tooltip("bin:N", title="Range"),
                    alt.Tooltip("count:Q", title="Count"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
        return

    hist_df = hist_df[["bin_start", "count"]].set_index("bin_start")
    st.bar_chart(hist_df)


def _safe_key_from_path(prefix: str, path: Path) -> str:
    raw = f"{prefix}_{_rel(path)}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw)


def _coerce_json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            return _json_pretty(json.loads(text))
        except Exception:
            return text
    return _json_pretty(value)


def _resolve_strategy_short_from_run_meta(
    run_meta: dict[str, Any], strategy_catalog: dict[str, dict[str, Any]]
) -> str | None:
    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    test_cfg = spec.get("test", {}) if isinstance(spec, dict) else {}

    strategy_module = test_cfg.get("strategy_module")
    strategy_tag = test_cfg.get("strategy_tag")

    candidates: list[str] = []
    if isinstance(strategy_module, str) and strategy_module.strip():
        mod = strategy_module.strip()
        candidates.append(mod.rsplit(".", 1)[-1])
        candidates.append(mod)
    if isinstance(strategy_tag, str) and strategy_tag.strip():
        candidates.append(strategy_tag.strip())

    for c in candidates:
        if c in strategy_catalog:
            return c

    if isinstance(strategy_module, str) and strategy_module.strip():
        for short, info in strategy_catalog.items():
            if str(info.get("module", "")).strip() == strategy_module.strip():
                return short

    return candidates[0] if candidates else None


def _prefill_limited_form_from_run(
    *,
    run_meta: dict[str, Any],
    pass_summary: dict[str, Any],
    strategy_catalog: dict[str, dict[str, Any]],
) -> None:
    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    strategy_cfg = spec.get("strategy", {}) if isinstance(spec, dict) else {}
    test_cfg = spec.get("test", {}) if isinstance(spec, dict) else {}
    cfg = spec.get("config", {}) if isinstance(spec, dict) else {}

    strategy_short = _resolve_strategy_short_from_run_meta(run_meta, strategy_catalog)
    if strategy_short:
        st.session_state["limited_strategy"] = strategy_short

    data_raw = spec.get("data")
    if data_raw:
        ds = _rel(_abs_path(str(data_raw)))
        dataset_options = [_rel(p) for p in _list_csv_data_files()]
        if ds in dataset_options:
            st.session_state["limited_dataset_select"] = ds
        else:
            st.session_state["limited_dataset_select"] = "Custom path..."
            st.session_state["limited_dataset_custom"] = ds
        st.session_state["limited_dataset_custom"] = ds

    entry_cfg = strategy_cfg.get("entry", {}) if isinstance(strategy_cfg, dict) else {}
    rules = entry_cfg.get("rules", []) if isinstance(entry_cfg, dict) else []
    first_rule = rules[0] if isinstance(rules, list) and rules else {}
    entry_plugin = first_rule.get("name") if isinstance(first_rule, dict) else None
    entry_params = first_rule.get("params", {}) if isinstance(first_rule, dict) else {}

    exit_cfg = strategy_cfg.get("exit", {}) if isinstance(strategy_cfg, dict) else {}
    exit_plugin = exit_cfg.get("name") if isinstance(exit_cfg, dict) else None
    exit_params = exit_cfg.get("params", {}) if isinstance(exit_cfg, dict) else {}

    sizing_cfg = strategy_cfg.get("sizing", {}) if isinstance(strategy_cfg, dict) else {}
    sizing_plugin = sizing_cfg.get("name") if isinstance(sizing_cfg, dict) else None
    sizing_params = sizing_cfg.get("params", {}) if isinstance(sizing_cfg, dict) else {}

    if entry_plugin:
        st.session_state["limited_entry_plugin_select"] = str(entry_plugin)
    if exit_plugin:
        st.session_state["limited_exit_plugin_select"] = str(exit_plugin)
    if sizing_plugin:
        st.session_state["limited_sizing_plugin_select"] = str(sizing_plugin)

    st.session_state["limited_entry_params_json"] = _coerce_json_text(entry_params)
    st.session_state["limited_exit_params_json"] = _coerce_json_text(exit_params)
    st.session_state["limited_sizing_params_json"] = _coerce_json_text(sizing_params)

    st.session_state["limited_run_base"] = str(test_cfg.get("run_base", "") or "")
    st.session_state["limited_test_name"] = str(test_cfg.get("test_name", "") or "")

    commission_rt = cfg.get("commission_per_round_trip") if isinstance(cfg, dict) else None
    if commission_rt is not None:
        st.session_state["limited_commission_rt"] = str(commission_rt)
    spread_pips = cfg.get("spread_pips") if isinstance(cfg, dict) else None
    if spread_pips is not None:
        st.session_state["limited_spread_pips"] = str(spread_pips)
    pip_size = cfg.get("pip_size") if isinstance(cfg, dict) else None
    if pip_size is not None:
        st.session_state["limited_pip_size"] = str(pip_size)

    pass_threshold = test_cfg.get("pass_threshold_pct")
    if pass_threshold is None:
        pass_threshold = pass_summary.get("pass_threshold_%", run_meta.get("pass_threshold_%"))
    if pass_threshold is not None and str(pass_threshold).strip() != "":
        st.session_state["limited_pass_threshold"] = str(pass_threshold)

    min_trades = test_cfg.get("min_trades")
    if min_trades is None:
        min_trades = pass_summary.get("min_trades", run_meta.get("min_trades"))
    if min_trades is not None and str(min_trades).strip() != "":
        st.session_state["limited_min_trades"] = str(min_trades)

    fav_criteria = test_cfg.get("favourable_criteria", "")
    if fav_criteria:
        if isinstance(fav_criteria, str):
            st.session_state["limited_favourable_criteria"] = fav_criteria
        else:
            st.session_state["limited_favourable_criteria"] = json.dumps(fav_criteria, separators=(",", ":"))

    extra_parts: list[str] = []
    ts_col = spec.get("ts_col")
    if ts_col and str(ts_col).strip() and str(ts_col).strip() != "timestamp":
        extra_parts.extend(["--ts-col", str(ts_col).strip()])
    entry_mode = entry_cfg.get("mode") if isinstance(entry_cfg, dict) else None
    if entry_mode and str(entry_mode).strip() and str(entry_mode).strip() != "all":
        extra_parts.extend(["--entry-mode", str(entry_mode).strip()])
    vote_k = entry_cfg.get("vote_k") if isinstance(entry_cfg, dict) else None
    if vote_k is not None and str(vote_k).strip() != "":
        extra_parts.extend(["--vote-k", str(vote_k)])
    lot_size = cfg.get("lot_size") if isinstance(cfg, dict) else None
    if lot_size is not None and str(lot_size).strip() != "":
        extra_parts.extend(["--lot-size", str(lot_size)])
    st.session_state["limited_extra_args"] = " ".join(shlex.quote(str(x)) for x in extra_parts)

    # These are expansion helpers; exact seed grids are already captured in params JSON.
    st.session_state["limited_seed_count"] = ""
    st.session_state["limited_seed_start"] = ""
    st.session_state["limited_exit_seed_count"] = ""
    st.session_state["limited_exit_seed_start"] = ""

    # Keep template UI stable while using explicit prefilled values.
    st.session_state["limited_test_type"] = "Custom"
    st.session_state["limited_test_template"] = "Core"
    st.session_state["_limited_prefill_lock"] = True


def _render_saved_interactive_html(path: Path) -> None:
    rel = _rel(path)
    open_url = "/" + _build_query_url(
        page="html_view",
        html_file=rel,
        ref_type=None,
        ref_name=None,
    )
    with st.expander("Interactive HTML"):
        st.link_button("Open in new tab", open_url, type="primary")
        st.caption(f"File: `{rel}`")

        show_inline = st.checkbox(
            "Show inline preview (can look compressed in Streamlit)",
            value=False,
            key=_safe_key_from_path("inline_preview", path),
        )
        if show_inline:
            html = path.read_text(encoding="utf-8")
            components.html(html, height=980, scrolling=True)


def _render_html_view_page() -> None:
    raw = _qp_get("html_file", "")
    if not raw:
        st.error("Missing html_file query parameter.")
        st.markdown(f"[Back to Results]({_build_query_url(page='results', html_file=None)})")
        return

    path = _abs_path(raw)
    if not path.exists():
        st.error(f"HTML file not found: `{raw}`")
        st.markdown(f"[Back to Results]({_build_query_url(page='results', html_file=None)})")
        return

    st.header("Interactive HTML Viewer")
    st.caption(f"File: `{_rel(path)}`")
    c1, c2 = st.columns([1, 5])
    c1.markdown(f"[Back]({_build_query_url(page='results', html_file=None)})")
    c2.download_button(
        "Download HTML",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/html",
    )

    html = path.read_text(encoding="utf-8")
    components.html(html, height=1200, scrolling=True)


WF_FOLDS_RE = re.compile(r"folds=(\d+)")
WF_FOLD_RE = re.compile(r"\[WFA\] fold (\d+)/(\d+)")
MC_PROGRESS_RE = re.compile(r"\[MC\] (\d+)/(\d+)")
LIMITED_PROGRESS_RE = re.compile(r"^\[\s*(\d+)/(\d+)\]")
DOWNLOAD_BARS_PROGRESS_RE = re.compile(
    r"Fetch progress \(bars mode\):.*rows=(\d+)/(\d+)\s+\(([0-9]+(?:\.[0-9]+)?)%\)",
    re.IGNORECASE,
)
DOWNLOAD_BACKFILL_COVERAGE_RE = re.compile(
    r"Backfill progress:\s*batches=(\d+),\s*rows=(\d+),.*time-coverage~([0-9]+(?:\.[0-9]+)?)%",
    re.IGNORECASE,
)


def _progress_update(workflow: str, line: str, state: dict[str, Any]) -> tuple[float | None, str | None]:
    text = line.strip()

    if workflow == "walkforward":
        m_mode = WF_FOLDS_RE.search(text)
        if m_mode:
            state["fold_total"] = int(m_mode.group(1))

        m_fold = WF_FOLD_RE.search(text)
        if m_fold:
            i = int(m_fold.group(1))
            total = int(m_fold.group(2))
            if "done" in text:
                state["fold_done"] = i
                frac = i / total if total else 0.0
                return frac, f"Walk-forward folds complete: {i}/{total}"
            frac = max(0.0, (i - 1) / total) if total else 0.0
            return frac, f"Running fold {i}/{total}"

    if workflow == "monte_carlo":
        m = MC_PROGRESS_RE.search(text)
        if m:
            i = int(m.group(1))
            total = int(m.group(2))
            frac = i / total if total else 0.0
            return frac, f"Monte Carlo simulations: {i}/{total}"

    if workflow == "limited":
        m = LIMITED_PROGRESS_RE.search(text)
        if m:
            i = int(m.group(1))
            total = int(m.group(2))
            frac = i / total if total else 0.0
            return frac, f"Limited test iterations: {i}/{total}"

    if workflow == "download":
        m_bars = DOWNLOAD_BARS_PROGRESS_RE.search(text)
        if m_bars:
            done = int(m_bars.group(1))
            total = int(m_bars.group(2))
            frac = done / total if total else 0.0
            return frac, f"Download progress (bars): {done}/{total}"

        m_cov = DOWNLOAD_BACKFILL_COVERAGE_RE.search(text)
        if m_cov:
            batches = int(m_cov.group(1))
            rows = int(m_cov.group(2))
            cov_pct = float(m_cov.group(3))
            frac = max(0.0, min(1.0, cov_pct / 100.0))
            return frac, f"Backfill progress: batches={batches}, rows={rows}, coverage~{cov_pct:.2f}%"

    return None, None


def _run_cli_live(cmd: list[str], *, workflow: str) -> tuple[int, str]:
    env = os.environ.copy()
    cur_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{cur_pythonpath}" if cur_pythonpath else str(SRC_DIR)
    env["PYTHONUNBUFFERED"] = "1"

    st.code(shlex.join(cmd), language="bash")
    status_ph = st.empty()
    progress_ph = st.empty()
    log_ph = st.empty()

    status_ph.info("Running...")
    progress_bar = progress_ph.progress(0.0)

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    parse_state: dict[str, Any] = {}

    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip("\n")
        lines.append(text)

        frac, msg = _progress_update(workflow, text, parse_state)
        if frac is not None:
            progress_bar.progress(float(max(0.0, min(1.0, frac))))
        if msg:
            status_ph.info(msg)

        log_ph.code("\n".join(lines[-300:]), language="text")

    rc = proc.wait()
    out = "\n".join(lines)

    if rc == 0:
        progress_bar.progress(1.0)
        status_ph.success("Command finished successfully.")
    else:
        status_ph.error(f"Command failed with exit code {rc}.")

    log_ph.code("\n".join(lines[-300:]), language="text")
    return rc, out


def _auto_generate_interactive_html(*, run_kind: str, run_path: Path) -> tuple[bool, str]:
    if run_kind == "walkforward":
        cmd = [SCRIPT_PYTHON, str(SCRIPTS_DIR / "plot_oos_equity.py"), "--run-dir", run_path.as_posix()]
        out_file = walkforward_html_path(run_path)
    elif run_kind == "monte_carlo":
        cmd = [SCRIPT_PYTHON, str(SCRIPTS_DIR / "plot_monte_carlo.py"), "--mc-run-dir", run_path.as_posix()]
        out_file = montecarlo_html_path(run_path)
    elif run_kind == "limited":
        cmd = [SCRIPT_PYTHON, str(SCRIPTS_DIR / "plot_limited_results.py"), "--run-dir", run_path.as_posix()]
        out_file = limited_html_path(run_path)
    else:
        return False, f"Unsupported run_kind: {run_kind}"

    rc, out = _run_cli(cmd)
    if rc == 0 and out_file.exists():
        return True, _rel(out_file)
    return False, out.strip() or f"HTML generation failed for {run_kind}"


def _ensure_limited_iteration_chart(*, run_dir: Path, iter_id: int) -> tuple[bool, Path | str]:
    out_file = limited_iteration_html_path(run_dir, iter_id)
    cmd = [
        SCRIPT_PYTHON,
        str(SCRIPTS_DIR / "plot_limited_iteration_lightweight.py"),
        "--run-dir",
        run_dir.as_posix(),
        "--iter",
        str(int(iter_id)),
        "--include-black-lines",
    ]
    rc, out = _run_cli(cmd)
    if rc != 0:
        return False, out.strip() or "Failed to generate iteration candlestick chart."

    parsed = _extract_saved_path(out, "Saved lightweight chart:")
    chart_path = parsed if parsed is not None else out_file
    if chart_path.exists():
        return True, chart_path
    return False, out.strip() or f"Chart file not found after generation: {_rel(out_file)}"


def _fmt_currency_display(value: Any) -> str:
    x = _as_float(value)
    return "n/a" if not np.isfinite(x) else f"${x:,.2f}"


def _fmt_percent_display(value: Any) -> str:
    x = _as_float(value)
    return "n/a" if not np.isfinite(x) else f"{x:.2f}%"


def _fmt_ratio_display(value: Any) -> str:
    x = _as_float(value)
    return "n/a" if not np.isfinite(x) else f"{x:.2f}"


def _build_walkforward_workbook_review(summary: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    aggregated = summary.get("aggregated_oos_summary", {}) if isinstance(summary.get("aggregated_oos_summary"), dict) else {}
    initial_review = summary.get("initial_review", {}) if isinstance(summary.get("initial_review"), dict) else {}
    metrics = initial_review.get("metrics", {}) if isinstance(initial_review.get("metrics"), dict) else {}

    annualized_net = _as_float(metrics.get("annualized_net_profit_abs"))
    total_net = _as_float(metrics.get("total_net_profit_abs", aggregated.get("net_profit_abs")))
    profit_factor = _as_float(metrics.get("profit_factor"))
    avg_trade_net = _as_float(metrics.get("avg_trade_net_profit_abs"))
    tharp = _as_float(metrics.get("tharp_expectancy"))
    max_dd_pct = _as_float(metrics.get("max_drawdown_abs_%", aggregated.get("max_drawdown_abs_%")))
    slope = _as_float(metrics.get("equity_slope_per_bar", aggregated.get("equity_slope_per_bar")))
    r2 = _as_float(metrics.get("equity_linearity_r2", aggregated.get("equity_linearity_r2")))
    flat_ratio = _as_float(metrics.get("flat_bars_ratio"))
    fuzziness = _as_float(metrics.get("equity_fuzziness"))
    dd_bars = _as_float(metrics.get("max_drawdown_duration_bars"))
    dd_ratio = _as_float(metrics.get("max_drawdown_duration_ratio"))
    equity_years = _as_float(metrics.get("equity_years", aggregated.get("equity_years")))
    win_rate_pct = _as_float(aggregated.get("win_rate_%", metrics.get("win_rate_%")))

    dd_months = float("nan")
    if np.isfinite(dd_ratio) and np.isfinite(equity_years) and equity_years > 0:
        dd_months = float(dd_ratio * equity_years * 12.0)

    if np.isfinite(slope) or np.isfinite(r2) or np.isfinite(flat_ratio) or np.isfinite(fuzziness):
        steadiness_actual = (
            f"R2 {_fmt_ratio_display(r2)} | flat {_fmt_percent_display(flat_ratio * 100.0)} | "
            f"fuzz {_fmt_ratio_display(fuzziness)} | slope {'up' if np.isfinite(slope) and slope > 0 else 'down/flat'}"
        )
    else:
        steadiness_actual = "n/a"

    rows: list[dict[str, Any]] = [
        {
            "Metric": "total net profit",
            "Actual": (
                (f"{_fmt_currency_display(annualized_net)}/yr" if np.isfinite(annualized_net) else "n/a")
                + (f" | total {_fmt_currency_display(total_net)}" if np.isfinite(total_net) else "")
            ),
            "Pass Criteria": ">$5k per year per contract",
            "Verdict": "PASS" if np.isfinite(annualized_net) and annualized_net > 5000.0 else "FAIL",
        },
        {
            "Metric": "profit factor",
            "Actual": _fmt_ratio_display(profit_factor),
            "Pass Criteria": "> 1.0",
            "Verdict": "PASS" if np.isfinite(profit_factor) and profit_factor > 1.0 else "FAIL",
        },
        {
            "Metric": "avg trade net profit",
            "Actual": _fmt_currency_display(avg_trade_net),
            "Pass Criteria": "> $50 per trade per contract",
            "Verdict": "PASS" if np.isfinite(avg_trade_net) and avg_trade_net > 50.0 else "FAIL",
        },
        {
            "Metric": "tharp expectancy",
            "Actual": _fmt_ratio_display(tharp),
            "Pass Criteria": "> 0.1",
            "Verdict": "PASS" if np.isfinite(tharp) and tharp > 0.10 else "FAIL",
        },
        {
            "Metric": "max dd",
            "Actual": _fmt_percent_display(max_dd_pct),
            "Pass Criteria": "< 20%",
            "Verdict": "PASS" if np.isfinite(max_dd_pct) and max_dd_pct < 20.0 else "FAIL",
        },
        {
            "Metric": "closed trade equity curve steadiness",
            "Actual": steadiness_actual,
            "Pass Criteria": "looks smooth diagonal up; not mostly flat with rapid rises",
            "Verdict": (
                "PASS"
                if np.isfinite(slope)
                and slope > 0
                and np.isfinite(r2)
                and r2 >= 0.60
                and np.isfinite(flat_ratio)
                and flat_ratio <= 0.35
                and np.isfinite(fuzziness)
                and fuzziness <= 0.40
                else (
                    "RETRY"
                    if np.isfinite(slope)
                    and slope > 0
                    and np.isfinite(r2)
                    and r2 >= 0.45
                    and np.isfinite(flat_ratio)
                    and flat_ratio <= 0.50
                    and np.isfinite(fuzziness)
                    and fuzziness <= 0.55
                    else "FAIL"
                )
            ),
        },
        {
            "Metric": "dd period",
            "Actual": (
                f"{dd_months:.2f} months"
                + (f" ({int(round(dd_bars))} bars)" if np.isfinite(dd_bars) else "")
                if np.isfinite(dd_months)
                else (f"{int(round(dd_bars))} bars" if np.isfinite(dd_bars) else "n/a")
            ),
            "Pass Criteria": "< 3 months (my own criteria)",
            "Verdict": (
                "PASS"
                if np.isfinite(dd_months) and dd_months < 3.0
                else ("RETRY" if np.isfinite(dd_months) and dd_months < 4.5 else ("FAIL" if np.isfinite(dd_months) else "INFO"))
            ),
        },
        {
            "Metric": "win rate (my own metric)",
            "Actual": _fmt_percent_display(win_rate_pct),
            "Pass Criteria": "No workbook threshold defined",
            "Verdict": "INFO",
        },
    ]

    verdicts = [str(row.get("Verdict", "")).upper() for row in rows]
    if any(v == "FAIL" for v in verdicts):
        decision = "FAIL"
    elif any(v == "RETRY" for v in verdicts):
        decision = "RETRY"
    else:
        decision = "PASS"

    return pd.DataFrame(rows), decision


def _render_walkforward_workbook_review(summary: dict[str, Any]) -> None:
    review_df, decision = _build_walkforward_workbook_review(summary)
    _render_decision_badge("Walk-forward workbook review", decision)

    aggregated = summary.get("aggregated_oos_summary", {}) if isinstance(summary.get("aggregated_oos_summary"), dict) else {}
    initial_review = summary.get("initial_review", {}) if isinstance(summary.get("initial_review"), dict) else {}
    metrics = initial_review.get("metrics", {}) if isinstance(initial_review.get("metrics"), dict) else {}

    _render_metric_row(
        [
            ("Annualized Net", _fmt_currency_display(metrics.get("annualized_net_profit_abs")), "{}"),
            ("Profit Factor", _fmt_ratio_display(metrics.get("profit_factor")), "{}"),
            ("Avg Trade Net", _fmt_currency_display(metrics.get("avg_trade_net_profit_abs")), "{}"),
            ("Max DD %", _fmt_percent_display(metrics.get("max_drawdown_abs_%", aggregated.get("max_drawdown_abs_%"))), "{}"),
        ]
    )
    _render_verdict_dataframe(review_df)


def _render_walkforward_results(run_dir: Path) -> None:
    wf_summary_path = summary_path(run_dir)
    config_path = spec_path(run_dir)
    folds_path = walkforward_folds_path(run_dir)

    if not wf_summary_path.exists() or not config_path.exists() or not folds_path.exists():
        st.error("Missing one of `summary.json`, `spec.json`, `tables/folds.csv` in selected walk-forward run.")
        return

    summary = _read_json(wf_summary_path)
    config = _read_json(config_path)
    folds = pd.read_csv(folds_path)

    strategy_raw = str(config.get("strategy", ""))
    strategy_short = strategy_raw.rsplit(".", 1)[-1] if strategy_raw else ""

    st.subheader("Walk-Forward Result")
    if strategy_short:
        discovered_catalog = _discover_strategy_catalog(_strategy_catalog_fingerprint())
        strategy_info = discovered_catalog.get(strategy_short, {})
        strategy_label = _strategy_display_name(strategy_short, discovered_catalog)
        st.markdown(_reference_link(f"Strategy: {strategy_label}", "strategy", strategy_short))
        strategy_cfg = strategy_info.get("strategy_config", {}) if isinstance(strategy_info, dict) else {}
        entry_name = None
        if isinstance(strategy_cfg, dict):
            rules = strategy_cfg.get("entry", {}).get("rules", [])
            if isinstance(rules, list) and rules and isinstance(rules[0], dict):
                entry_name = rules[0].get("name")
            exit_name = strategy_cfg.get("exit", {}).get("name")
            link_cols = st.columns(2)
            if entry_name:
                link_cols[0].markdown(_reference_link(f"Entry: {entry_name}", "entry", str(entry_name)))
            if exit_name:
                link_cols[1].markdown(_reference_link(f"Exit: {exit_name}", "exit", str(exit_name)))

    _render_walkforward_workbook_review(summary)

    profit_col = "oos_net_profit_abs" if "oos_net_profit_abs" in folds.columns else "oos_total_return_%"
    profits_all = pd.to_numeric(folds.get(profit_col), errors="coerce").replace([np.inf, -np.inf], np.nan)
    segment_count = int(len(folds))
    profitable_count = int((profits_all > 0).sum()) if segment_count else 0
    profitable_oos_pct = (100.0 * profitable_count / segment_count) if segment_count else float("nan")

    profits_finite = profits_all.dropna()
    total_oos_profit = float(profits_finite.sum()) if not profits_finite.empty else float("nan")
    top2_sum = float(profits_finite.nlargest(min(2, len(profits_finite))).sum()) if not profits_finite.empty else float("nan")
    if np.isfinite(total_oos_profit) and total_oos_profit > 0 and np.isfinite(top2_sum):
        top2_share_pct = float((top2_sum / total_oos_profit) * 100.0)
    elif segment_count > 0:
        top2_share_pct = float("inf")
    else:
        top2_share_pct = float("nan")

    decision = _classify_walkforward_decision(
        segments=segment_count,
        profitable_oos_pct=profitable_oos_pct,
        top2_share_pct=top2_share_pct,
    )
    _render_decision_badge("Walk-Forward OOS Survival + Concentration", decision)

    st.markdown("**Secondary diagnostics**")
    _render_metric_row(
        [
            ("OOS Segments", segment_count, "{:d}"),
            ("ProfitableOOS %", profitable_oos_pct, "{:.2f}"),
            ("Top2Share %", top2_share_pct, "{:.2f}"),
            ("Total OOS Profit", total_oos_profit, "{:.2f}"),
        ]
    )

    threshold_rows = pd.DataFrame(
        [
            {
                "Metric": "ProfitableOOS %",
                "Actual": "n/a" if not np.isfinite(profitable_oos_pct) else f"{profitable_oos_pct:.2f}%",
                "PASS": ">= 60%",
                "RETRY": "50% to <60%",
                "FAIL": "< 50%",
            },
            {
                "Metric": "Top2Share %",
                "Actual": "n/a" if not np.isfinite(top2_share_pct) else f"{top2_share_pct:.2f}%",
                "PASS": "<= 50%",
                "RETRY": ">50% to <=60%",
                "FAIL": "> 60%",
            },
            {
                "Metric": "Runs Produced",
                "Actual": f"{segment_count} OOS segments",
                "PASS": ">= 8 segments",
                "RETRY": "-",
                "FAIL": "< 8 segments",
            },
        ]
    )
    st.dataframe(threshold_rows, use_container_width=True, hide_index=True)

    st.subheader("OOS Segment Results")
    fold_view = pd.DataFrame(
        {
            "Fold": folds.get("fold", pd.Series(index=folds.index, dtype=int)),
            "IS Period": [
                _time_window_label(row.get("is_start_time"), row.get("is_end_time_exclusive"))
                for _, row in folds.iterrows()
            ],
            "OOS Period": [
                _time_window_label(row.get("oos_start_time"), row.get("oos_end_time_exclusive"))
                for _, row in folds.iterrows()
            ],
            "Chosen Params": [
                _compact_params_cell(row.get("best_params"))
                for _, row in folds.iterrows()
            ],
            "OOS Trades": pd.to_numeric(
                folds.get("oos_trades", pd.Series(index=folds.index, dtype=float)),
                errors="coerce",
            ),
            "OOS Net Profit": pd.to_numeric(
                folds.get("oos_net_profit_abs", pd.Series(index=folds.index, dtype=float)),
                errors="coerce",
            ),
            "OOS Return %": pd.to_numeric(
                folds.get("oos_total_return_%", pd.Series(index=folds.index, dtype=float)),
                errors="coerce",
            ),
            "OOS Max DD %": pd.to_numeric(
                folds.get("oos_max_drawdown_abs_%", pd.Series(index=folds.index, dtype=float)),
                errors="coerce",
            ),
            "WFE %": pd.to_numeric(
                folds.get("wfe_pct", pd.Series(index=folds.index, dtype=float)),
                errors="coerce",
            ),
            "WFE Pass": folds.get("wfe_pass", pd.Series(index=folds.index, dtype=object)),
        }
    )
    metric_col = "OOS Net Profit" if "OOS Net Profit" in fold_view.columns else "OOS Return %"
    _render_win_loss_dataframe(fold_view, metric_col=metric_col, hide_index=True)

    equity_path = walkforward_oos_equity_path(run_dir)
    eq = pd.read_csv(equity_path) if equity_path.exists() else pd.DataFrame()
    if {"time", "equity"}.issubset(eq.columns):
        eq["time"] = pd.to_datetime(eq["time"], utc=True, errors="coerce")
        st.subheader("OOS Equity Curve")
        if px is not None:
            fig = px.line(eq, x="time", y="equity", title="OOS Equity Curve")
            _add_walkforward_fold_annotations(fig, folds)
            fig.update_layout(margin={"l": 60, "r": 30, "t": 110, "b": 50})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(eq.set_index("time")["equity"])

    html_path = walkforward_html_path(run_dir)
    if html_path.exists():
        _render_saved_interactive_html(html_path)


def _render_mc_results(mc_run_dir: Path) -> None:
    mc_summary_path = summary_path(mc_run_dir)
    if not mc_summary_path.exists():
        st.error("Missing `summary.json` in selected Monte Carlo run.")
        return

    summary = _read_json(mc_summary_path)
    metrics = summary.get("metrics", {})
    mc_cfg = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
    run_dir_raw = str(summary.get("run_dir", "")).strip()
    wf_run_dir = _abs_path(run_dir_raw) if run_dir_raw else None

    st.subheader("Monte Carlo Result")
    st.markdown("**DD + Outlier Dependence**")

    sims_path = montecarlo_simulations_path(mc_run_dir)
    sims = pd.read_csv(sims_path) if sims_path.exists() else pd.DataFrame()

    dd95_pct = _as_float(metrics.get("max_drawdown_%_p95"))
    if not np.isfinite(dd95_pct) and "max_drawdown_%" in sims.columns:
        dd95_pct = _as_float(pd.to_numeric(sims["max_drawdown_%"], errors="coerce").quantile(0.95))

    ddoos_pct = float("nan")
    original_oos_profit = float("nan")
    profit_after_top3_removal = float("nan")
    pnl_col = ""
    if wf_run_dir is not None:
        wf_summary_path = summary_path(wf_run_dir)
        if wf_summary_path.exists():
            wf_summary = _read_json(wf_summary_path)
            agg = wf_summary.get("aggregated_oos_summary", {}) if isinstance(wf_summary.get("aggregated_oos_summary"), dict) else {}
            ddoos_pct = _as_float(agg.get("max_drawdown_abs_%"))

        if not np.isfinite(ddoos_pct):
            oos_eq_path = walkforward_oos_equity_path(wf_run_dir)
            if oos_eq_path.exists():
                eq = pd.read_csv(oos_eq_path)
                if "equity" in eq.columns:
                    ddoos_pct = _max_drawdown_pct_from_equity_series(eq["equity"])

        oos_trades_path = walkforward_oos_trades_path(wf_run_dir)
        if oos_trades_path.exists():
            oos_trades = pd.read_csv(oos_trades_path)
            for candidate in ("pnl", "net_profit", "profit", "net_profit_abs"):
                if candidate in oos_trades.columns:
                    pnl_col = candidate
                    break
            if pnl_col:
                pnl_s = pd.to_numeric(oos_trades[pnl_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not pnl_s.empty:
                    original_oos_profit = float(pnl_s.sum())
                    top_n = min(3, len(pnl_s))
                    drop_idx = pnl_s.nlargest(top_n).index
                    profit_after_top3_removal = float(pnl_s.drop(index=drop_idx, errors="ignore").sum())

    dd_ratio = (
        float(dd95_pct / ddoos_pct)
        if np.isfinite(dd95_pct) and np.isfinite(ddoos_pct) and ddoos_pct > 0
        else float("nan")
    )
    decision = _classify_monte_carlo_risk_decision(
        dd_ratio=dd_ratio,
        original_profit=original_oos_profit,
        profit_after_removal=profit_after_top3_removal,
    )
    _render_decision_badge("Monte Carlo Risk (DD + Outlier Dependence)", decision)

    _render_metric_row(
        [
            ("MC DD95 %", dd95_pct, "{:.2f}"),
            ("OOS DDOOS %", ddoos_pct, "{:.2f}"),
            ("DD95/DDOOS", dd_ratio, "{:.3f}"),
            ("MC Simulations", int(mc_cfg.get("n_sims", len(sims))), "{:d}"),
        ]
    )
    _render_metric_row(
        [
            ("Original OOS Profit", original_oos_profit, "{:.2f}"),
            ("Profit After Top-3 Removal", profit_after_top3_removal, "{:.2f}"),
            ("Outlier Removal Calc", 1, "{:d}"),
            ("Trade PnL Column", pnl_col or "missing", "{}"),
        ]
    )
    threshold_df = pd.DataFrame(
        [
            {
                "Metric": "DD95/DDOOS",
                "Actual": "n/a" if not np.isfinite(dd_ratio) else f"{dd_ratio:.3f}",
                "PASS": "<= 1.7x",
                "RETRY": ">1.7x to <=2.0x",
                "FAIL": "> 2.0x",
            },
            {
                "Metric": "NetProfitAfterRemoval",
                "Actual": "n/a" if not np.isfinite(profit_after_top3_removal) else f"{profit_after_top3_removal:,.2f}",
                "PASS": "> 0",
                "RETRY": "<=0 but > -10% of original OOS profit",
                "FAIL": "<= -10% of original OOS profit",
            },
            {
                "Metric": "Runs Produced",
                "Actual": f"{int(mc_cfg.get('n_sims', len(sims)))} sims + 1 removal calc",
                "PASS": "10,000 sims + 1 removal calc",
                "RETRY": "-",
                "FAIL": "far below target sims",
            },
        ]
    )
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    if not sims.empty:
        st.subheader("DD Distribution")
        if "max_drawdown_%" in sims.columns:
            _render_histogram(sims["max_drawdown_%"], title="Max Drawdown % Distribution", bins=70)
        sim_cols = [c for c in ["sim", "max_drawdown_%", "return_%", "final_equity"] if c in sims.columns]
        sims_view = sims[sim_cols] if sim_cols else sims
        metric_col = "return_%" if "return_%" in sims_view.columns else sim_cols[1] if len(sim_cols) > 1 else sims_view.columns[0]
        _render_win_loss_dataframe(sims_view, metric_col=metric_col, hide_index=True)

    q_path = montecarlo_quantile_paths_path(mc_run_dir)
    if q_path.exists():
        qdf = pd.read_csv(q_path)
        st.subheader("Equity Quantile Paths")
        if "trade_n" in qdf.columns:
            melted = qdf.melt(id_vars=["trade_n"], var_name="quantile", value_name="equity")
            if px is not None:
                fig = px.line(melted, x="trade_n", y="equity", color="quantile", title="Equity Quantile Paths")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(qdf.set_index("trade_n"))

    html_path = montecarlo_html_path(mc_run_dir)
    if html_path.exists():
        _render_saved_interactive_html(html_path)


def _render_limited_results(run_dir: Path, *, strategy_catalog: dict[str, dict[str, Any]]) -> None:
    results_path = limited_iterations_path(run_dir)
    trades_path = limited_trades_path(run_dir)
    pass_path = summary_path(run_dir)
    meta_path = spec_path(run_dir)

    if not results_path.exists() or not pass_path.exists() or not meta_path.exists():
        st.error("Missing one of `tables/iterations.csv`, `summary.json`, `spec.json`.")
        return

    results = pd.read_csv(results_path)
    pass_summary = _read_json(pass_path)
    run_meta = _read_json(meta_path)

    st.subheader("Limited Test Result")
    c_rerun, _ = st.columns([0.2, 0.8])
    rerun_key = f"limited_rerun_{_safe_key_from_path('run', run_dir)}"
    if c_rerun.button("Rerun Test", key=rerun_key):
        try:
            _prefill_limited_form_from_run(
                run_meta=run_meta,
                pass_summary=pass_summary,
                strategy_catalog=strategy_catalog,
            )
            st.session_state["_force_nav_page"] = "run_test"
            st.rerun()
        except Exception as e:
            st.error(f"Unable to prefill rerun form: {e}")

    _render_limited_test_param_space(run_meta)

    spec = run_meta.get("spec", {}) if isinstance(run_meta, dict) else {}
    initial_equity = _infer_initial_equity_from_spec(spec)
    results = results.copy()
    results["_net_profit_abs"] = _limited_net_profit_series(results, initial_equity=initial_equity)
    total_iters = int(len(results))

    is_monkey = _is_monkey_limited_run(run_meta)
    if is_monkey:
        davey_cfg = pass_summary.get("davey_style", {})
        davey_enabled = isinstance(davey_cfg, dict) and bool(davey_cfg.get("enabled", False))
        pass_threshold = _as_float(pass_summary.get("pass_threshold_%"))
        if davey_enabled:
            ret_worse_pct = _as_float(davey_cfg.get("return_worse_pct"))
            dd_worse_pct = _as_float(davey_cfg.get("maxdd_worse_pct"))
            both_worse_pct = _as_float(davey_cfg.get("both_worse_pct", pass_summary.get("favourable_pct")))
            decision = "PASS" if bool(pass_summary.get("passed", False)) else "FAIL"
            _render_decision_badge("Monkey Test (Strict Davey Style)", decision)
            _render_metric_row(
                [
                    ("Runs Produced", total_iters, "{:d}"),
                    ("Return Worse %", ret_worse_pct, "{:.2f}"),
                    ("MaxDD Worse %", dd_worse_pct, "{:.2f}"),
                    ("Pass Threshold %", pass_threshold, "{:.2f}"),
                ]
            )
            _render_metric_row(
                [
                    ("Both Worse %", both_worse_pct, "{:.2f}"),
                    ("Target Trials", 8000, "{:d}"),
                    ("PASS Floor %", pass_threshold, "{:.0f}"),
                    ("Decision Rule", "Return Worse% >= floor AND MaxDD Worse% >= floor", "{}"),
                ]
            )
            monkey_df = pd.DataFrame(
                [
                    {
                        "Metric": "Return Worse % (trial return < baseline return)",
                        "Actual": "n/a" if not np.isfinite(ret_worse_pct) else f"{ret_worse_pct:.2f}%",
                        "Verdict": (
                            "PASS"
                            if np.isfinite(ret_worse_pct) and np.isfinite(pass_threshold) and ret_worse_pct >= pass_threshold
                            else "FAIL"
                        ),
                        "PASS": f">= {pass_threshold:.0f}%",
                        "FAIL": f"< {pass_threshold:.0f}%",
                    },
                    {
                        "Metric": "MaxDD Worse % (trial MaxDD > baseline MaxDD)",
                        "Actual": "n/a" if not np.isfinite(dd_worse_pct) else f"{dd_worse_pct:.2f}%",
                        "Verdict": (
                            "PASS"
                            if np.isfinite(dd_worse_pct) and np.isfinite(pass_threshold) and dd_worse_pct >= pass_threshold
                            else "FAIL"
                        ),
                        "PASS": f">= {pass_threshold:.0f}%",
                        "FAIL": f"< {pass_threshold:.0f}%",
                    },
                    {
                        "Metric": "Both Worse % (informational)",
                        "Actual": "n/a" if not np.isfinite(both_worse_pct) else f"{both_worse_pct:.2f}%",
                        "Verdict": "INFO",
                        "PASS": "n/a",
                        "FAIL": "n/a",
                    },
                ]
            )
            _render_verdict_dataframe(monkey_df)
        else:
            dominance_pct = _as_float(pass_summary.get("favourable_pct"))
            decision = _classify_monkey_dominance_decision(dominance_pct)
            _render_decision_badge("Monkey Test (Dominance Criteria)", decision)
            _render_metric_row(
                [
                    ("Runs Produced", total_iters, "{:d}"),
                    ("Dominance %", dominance_pct, "{:.2f}"),
                    ("Pass Threshold %", pass_threshold, "{:.2f}"),
                    ("Target Trials", 8000, "{:d}"),
                ]
            )
            _render_metric_row(
                [
                    ("PASS Floor %", 85.0, "{:.0f}"),
                    ("RETRY Floor %", 75.0, "{:.0f}"),
                    ("Test Window", "same as Limited Core", "{}"),
                    ("Decision Rule", "dominance >= 85 PASS / >= 75 RETRY", "{}"),
                ]
            )
            monkey_df = pd.DataFrame(
                [
                    {
                        "Metric": "Dominance % (return worse AND MaxDD worse)",
                        "Actual": "n/a" if not np.isfinite(dominance_pct) else f"{dominance_pct:.2f}%",
                        "Verdict": _classify_monkey_dominance_decision(dominance_pct),
                        "PASS": ">= 85%",
                        "RETRY": "75% to <85%",
                        "FAIL": "< 75%",
                    },
                    {
                        "Metric": "Runs Produced",
                        "Actual": f"{total_iters} trials",
                        "Verdict": "PASS" if total_iters >= 8000 else "FAIL",
                        "PASS": "8,000 trials",
                        "RETRY": "-",
                        "FAIL": "far below target trials",
                    },
                ]
            )
            _render_verdict_dataframe(monkey_df)

        with st.expander("Constrained Dominance (Baseline-matched monkey trials)", expanded=True):
            st.caption(
                "Post-filter monkey trials to baseline-matching trade count and long/short mix, "
                "then compute dominance % on that filtered subset."
            )
            baseline_run_candidates = [p for p in _discover_limited_runs() if p.resolve() != run_dir.resolve()]
            baseline_opts = [_rel(p) for p in baseline_run_candidates]
            default_baseline = baseline_opts[0] if baseline_opts else ""
            baseline_run_raw = _select_path_from_options(
                "Baseline limited run (core fixed strategy)",
                baseline_opts,
                default_baseline,
                key_prefix=f"monkey_base_{_safe_key_from_path('run', run_dir)}",
            )

            ctol1, ctol2 = st.columns(2)
            trade_tol_pct = float(
                ctol1.number_input(
                    "Trade count tolerance %",
                    min_value=0.0,
                    value=5.0,
                    step=0.5,
                    key=f"monkey_trade_tol_{run_dir.as_posix()}",
                )
            )
            long_tol_pp = float(
                ctol2.number_input(
                    "Long ratio tolerance (percentage points)",
                    min_value=0.0,
                    value=5.0,
                    step=0.5,
                    key=f"monkey_long_tol_{run_dir.as_posix()}",
                )
            )

            if baseline_run_raw.strip():
                base_run_dir = _abs_path(baseline_run_raw.strip())
                loaded = _load_limited_run_artifacts(base_run_dir)
                if loaded is None:
                    st.error("Could not load baseline limited run artifacts.")
                else:
                    base_results, base_meta, _base_pass = loaded
                    iter_values: list[int] = []
                    if "iter" in base_results.columns:
                        iter_values = (
                            pd.to_numeric(base_results["iter"], errors="coerce")
                            .dropna()
                            .astype(int)
                            .drop_duplicates()
                            .sort_values()
                            .tolist()
                        )
                    selected_base_iter: int | None = None
                    if len(iter_values) > 1:
                        selected_base_iter = st.selectbox(
                            "Baseline iteration",
                            iter_values,
                            index=0,
                            key=f"monkey_base_iter_{run_dir.as_posix()}",
                        )
                    baseline_row = _pick_limited_row(base_results, iter_id=selected_base_iter)
                    if baseline_row is None:
                        st.error("Baseline run has no rows in `tables/iterations.csv`.")
                    else:
                        base_initial_eq = _infer_initial_equity_from_spec(base_meta.get("spec", {}) if isinstance(base_meta, dict) else {})
                        baseline = _limited_baseline_metrics_from_row(baseline_row, initial_equity=base_initial_eq)

                        _render_metric_row(
                            [
                                ("Baseline Trades", baseline.get("trades", float("nan")), "{:.0f}"),
                                ("Baseline Long %", baseline.get("long_trade_pct", float("nan")), "{:.2f}"),
                                ("Baseline Return %", baseline.get("total_return_%", float("nan")), "{:.2f}"),
                                ("Baseline MaxDD %", baseline.get("max_drawdown_abs_%", float("nan")), "{:.2f}"),
                            ]
                        )

                        constrained = _compute_monkey_constrained_dominance(
                            monkey_results=results,
                            baseline_return_pct=float(baseline.get("total_return_%", float("nan"))),
                            baseline_max_dd_pct=float(baseline.get("max_drawdown_abs_%", float("nan"))),
                            baseline_trades=float(baseline.get("trades", float("nan"))),
                            baseline_long_trade_pct=float(baseline.get("long_trade_pct", float("nan"))),
                            trade_count_tol_pct=float(trade_tol_pct),
                            long_ratio_tol_pp=float(long_tol_pp),
                        )
                        constrained_dominance_pct = _as_float(constrained.get("dominance_valid_pct"))
                        constrained_decision = _classify_monkey_dominance_decision(constrained_dominance_pct)
                        _render_decision_badge("Constrained Dominance (filtered trials)", constrained_decision)
                        _render_metric_row(
                            [
                                ("Valid Trials", int(constrained.get("valid_trials", 0)), "{:d}"),
                                ("Valid %", _as_float(constrained.get("valid_pct")), "{:.2f}"),
                                ("Dominance on Valid %", constrained_dominance_pct, "{:.2f}"),
                                ("Target Valid Trials", 8000, "{:d}"),
                            ]
                        )
                        _render_metric_row(
                            [
                                ("PASS Floor %", 85.0, "{:.0f}"),
                                ("RETRY Floor %", 75.0, "{:.0f}"),
                                ("Trade Tol %", float(trade_tol_pct), "{:.1f}"),
                                ("Long Ratio Tol pp", float(long_tol_pp), "{:.1f}"),
                            ]
                        )
                        filt_df = pd.DataFrame(
                            [
                                {
                                    "Metric": "Trade count filter",
                                    "Actual": (
                                        f"[{_as_float(constrained.get('trade_low')):.1f}, "
                                        f"{_as_float(constrained.get('trade_high')):.1f}]"
                                    ),
                                    "Rule": f"baseline trades ±{trade_tol_pct:.1f}%",
                                },
                                {
                                    "Metric": "Long ratio filter",
                                    "Actual": (
                                        f"[{_as_float(constrained.get('long_low')):.1f}%, "
                                        f"{_as_float(constrained.get('long_high')):.1f}%]"
                                    ),
                                    "Rule": f"baseline long_trade_pct ±{long_tol_pp:.1f} pp",
                                },
                                {
                                    "Metric": "Dominance on filtered trials",
                                    "Actual": (
                                        "n/a"
                                        if not np.isfinite(constrained_dominance_pct)
                                        else f"{constrained_dominance_pct:.2f}%"
                                    ),
                                    "Rule": "trial return < baseline return AND trial MaxDD > baseline MaxDD",
                                },
                            ]
                        )
                        st.dataframe(filt_df, use_container_width=True, hide_index=True)
                        st.caption(
                            "Current runner does not hard-resample until 8,000 valid constrained trials; "
                            "this view computes constrained dominance by filtering generated trials."
                        )
    else:
        net_profit = pd.to_numeric(results["_net_profit_abs"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        trades_s = pd.to_numeric(results.get("trades", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)

        profitable_runs = int((net_profit > 0).sum()) if not net_profit.empty else 0
        median_profit = float(net_profit.median()) if not net_profit.dropna().empty else float("nan")
        median_trades = float(trades_s.median()) if not trades_s.dropna().empty else float("nan")
        best_profit = float(net_profit.max()) if not net_profit.dropna().empty else float("nan")
        best_over_median = (
            float(best_profit / median_profit)
            if np.isfinite(best_profit) and np.isfinite(median_profit) and median_profit > 0
            else float("inf")
        )

        decision = _classify_limited_core_decision(
            profitable_runs=profitable_runs,
            median_profit=median_profit,
            median_trades=median_trades,
            best_over_median=best_over_median,
        )
        _render_decision_badge("Limited Core Test (Grid + Robustness)", decision)
        _render_metric_row(
            [
                ("Runs Produced", total_iters, "{:d}"),
                ("ProfitableRuns", profitable_runs, "{:d}"),
                ("MedianProfit", median_profit, "{:.2f}"),
                ("MedianTrades", median_trades, "{:.1f}"),
            ]
        )
        _render_metric_row(
            [
                ("BestOverMedian", best_over_median, "{:.3f}"),
                ("BestProfit", best_profit, "{:.2f}"),
                ("Target Backtests", 100, "{:d}"),
                ("NetProfit Basis", f"initial_equity={initial_equity:,.0f}", "{}"),
            ]
        )

        core_df = pd.DataFrame(
            [
                {
                    "Metric": "ProfitableRuns",
                    "Actual": f"{profitable_runs}",
                    "Verdict": "PASS" if profitable_runs >= 70 else "RETRY" if profitable_runs >= 50 else "FAIL",
                    "PASS": ">= 70",
                    "RETRY": "50 to 69",
                    "FAIL": "< 50",
                },
                {
                    "Metric": "MedianProfit",
                    "Actual": "n/a" if not np.isfinite(median_profit) else f"{median_profit:,.2f}",
                    "Verdict": "PASS" if np.isfinite(median_profit) and median_profit > 0 else "FAIL",
                    "PASS": "> 0",
                    "RETRY": "must remain > 0",
                    "FAIL": "<= 0",
                },
                {
                    "Metric": "MedianTrades",
                    "Actual": "n/a" if not np.isfinite(median_trades) else f"{median_trades:.1f}",
                    "Verdict": "PASS" if median_trades >= 75 else "RETRY" if median_trades >= 30 else "FAIL",
                    "PASS": ">= 75",
                    "RETRY": "30 to 74",
                    "FAIL": "< 30",
                },
                {
                    "Metric": "BestOverMedian",
                    "Actual": "inf" if not np.isfinite(best_over_median) else f"{best_over_median:.3f}",
                    "Verdict": (
                        "PASS"
                        if np.isfinite(best_over_median) and best_over_median <= 3.0
                        else "RETRY"
                        if np.isfinite(best_over_median) and best_over_median <= 5.0
                        else "FAIL"
                    ),
                    "PASS": "<= 3.0",
                    "RETRY": ">3.0 to <=5.0",
                    "FAIL": "> 5.0",
                },
                {
                    "Metric": "Runs Produced",
                    "Actual": f"{total_iters} backtests",
                    "Verdict": "PASS" if total_iters >= 100 else "FAIL",
                    "PASS": "100 backtests",
                    "RETRY": "-",
                    "FAIL": "far below target backtests",
                },
            ]
        )
        _render_verdict_dataframe(core_df)

    st.subheader("Iterations")
    iter_cols = [
        c
        for c in [
            "iter",
            "favourable",
            "trades",
            "_net_profit_abs",
            "total_return_%",
            "max_drawdown_abs_%",
            "entry_params",
            "exit_params",
        ]
        if c in results.columns
    ]
    iter_view = results[iter_cols] if iter_cols else results
    _render_win_loss_dataframe(iter_view, metric_col="_net_profit_abs", hide_index=True)

    if trades_path.exists():
        trades = pd.read_csv(trades_path)
        st.subheader("Iteration Trade Drill-Down")
        if "iter" in trades.columns and "iter" in results.columns:
            iter_values = pd.to_numeric(results["iter"], errors="coerce").dropna().astype(int).tolist()
            iter_values = sorted(set(iter_values))
            if iter_values:
                default_iter = iter_values[0]
                selected_iter = st.selectbox(
                    "Select iteration",
                    iter_values,
                    index=iter_values.index(default_iter),
                    key=f"limited_selected_iter_{run_dir.as_posix()}",
                )

                row = results.loc[pd.to_numeric(results["iter"], errors="coerce") == int(selected_iter)]
                if not row.empty:
                    r = row.iloc[0]
                    trades_n = pd.to_numeric(pd.Series([r.get("trades", 0)]), errors="coerce").iloc[0]
                    trades_n = 0 if pd.isna(trades_n) else int(trades_n)
                    _render_metric_row(
                        [
                            ("Iter", int(selected_iter), "{:d}"),
                            ("Favourable", _status_label(bool(r.get("favourable", False))), "{}"),
                            ("Net Profit", float(r.get("_net_profit_abs", float("nan"))), "{:.2f}"),
                            ("Trades", trades_n, "{:d}"),
                            ("BestOverMedian Ref", "see scorecard", "{}"),
                        ]
                    )

                open_chart_key = (
                    f"limited_open_iter_chart_"
                    f"{_safe_key_from_path('run', run_dir)}_{int(selected_iter)}"
                )
                if st.button(
                    "Open iteration candlestick chart",
                    key=open_chart_key,
                    type="primary",
                ):
                    with st.spinner("Preparing iteration candlestick chart..."):
                        ok, chart_or_msg = _ensure_limited_iteration_chart(
                            run_dir=run_dir,
                            iter_id=int(selected_iter),
                        )
                    if ok:
                        chart_path = chart_or_msg if isinstance(chart_or_msg, Path) else _abs_path(str(chart_or_msg))
                        try:
                            st.query_params["page"] = "html_view"
                            st.query_params["html_file"] = _rel(chart_path)
                        except Exception:
                            pass
                        st.rerun()
                    else:
                        st.error(str(chart_or_msg))

                trades_iter = trades.loc[pd.to_numeric(trades["iter"], errors="coerce") == int(selected_iter)].copy()
                if trades_iter.empty:
                    st.info("No trades recorded for this iteration.")
                else:
                    trade_cols = [
                        c
                        for c in [
                            "entry_time",
                            "exit_time",
                            "side",
                            "entry",
                            "exit",
                            "exit_reason",
                            "units",
                            "pnl",
                            "r_multiple",
                            "bars_held",
                            "commission",
                            "mfe",
                            "mae",
                            "mfe_R",
                            "mae_R",
                            "giveback",
                            "equity_after",
                            "entry_params",
                            "exit_params",
                        ]
                        if c in trades_iter.columns
                    ]
                    trades_view = trades_iter[trade_cols] if trade_cols else trades_iter
                    _render_win_loss_dataframe(trades_view, metric_col="pnl", hide_index=True, max_styled_rows=5000)
            else:
                st.info("No iteration ids available in results.")
        else:
            st.info("`tables/trades.csv` found, but missing `iter` column for drill-down.")
    else:
        st.info("Per-iteration trade details are not available for this run. Re-run limited tests to generate `tables/trades.csv`.")

    html_path = limited_html_path(run_dir)
    if html_path.exists():
        _render_saved_interactive_html(html_path)


def _render_focused_results_page(*, strategy_catalog: dict[str, dict[str, Any]]) -> bool:
    workflow = str(_qp_get("result_workflow", "") or "").strip().lower()
    run_raw = str(_qp_get("result_run", "") or "").strip()
    if not workflow or not run_raw:
        return False

    st.subheader("Focused Result")
    clear_url = _build_query_url(page="results", result_workflow=None, result_run=None)
    st.caption(
        f"Opened from strategy run history for `{workflow}`. "
        f"[Back to full results browser]({clear_url})"
    )

    run_dir = _abs_path(run_raw)
    if workflow not in {"limited", "walkforward", "monte_carlo"}:
        st.error(f"Unsupported result workflow: `{workflow}`.")
        return True
    if not _is_relative_to(run_dir, RUNS_ROOT / "strategies"):
        st.error("Focused result path is outside the supported run roots.")
        return True
    if not run_dir.exists() or not run_dir.is_dir():
        st.error(f"Focused run does not exist: `{_rel(run_dir)}`")
        return True

    st.caption(f"Run: `{_rel(run_dir)}`")
    _render_result_run_detail(
        workflow=workflow,
        run_dir=run_dir,
        strategy_catalog=strategy_catalog,
        delete_key_prefix=f"delete_focus_{_safe_key_from_path('run', run_dir)}",
    )
    return True


def _render_result_run_detail(
    *,
    workflow: str,
    run_dir: Path,
    strategy_catalog: dict[str, dict[str, Any]],
    delete_key_prefix: str,
) -> None:
    workflow_key = str(workflow or "").strip().lower()
    if workflow_key == "walkforward":
        st.session_state["last_walkforward_run"] = run_dir.as_posix()
        _render_walkforward_results(run_dir)
        _render_delete_target_ui(
            label="walk-forward run",
            raw_path=run_dir.as_posix(),
            allowed_roots=[RUNS_ROOT / "strategies"],
            session_keys_to_clear=["last_walkforward_run", "last_mc_run"],
            widget_prefixes_to_clear=["res_wf", "res_mc", "res_live_mc"],
            key_prefix=delete_key_prefix,
            expect="dir",
        )
        return

    if workflow_key == "monte_carlo":
        st.session_state["last_mc_run"] = run_dir.as_posix()
        _render_mc_results(run_dir)
        _render_delete_target_ui(
            label="Monte Carlo run",
            raw_path=run_dir.as_posix(),
            allowed_roots=[RUNS_ROOT / "strategies"],
            session_keys_to_clear=["last_mc_run"],
            widget_prefixes_to_clear=["res_mc", "res_live_mc"],
            key_prefix=delete_key_prefix,
            expect="dir",
        )
        return

    st.session_state["last_limited_run"] = run_dir.as_posix()
    _render_limited_results(run_dir, strategy_catalog=strategy_catalog)
    _render_delete_target_ui(
        label="limited run",
        raw_path=run_dir.as_posix(),
        allowed_roots=[RUNS_ROOT / "strategies"],
        session_keys_to_clear=["last_limited_run"],
        widget_prefixes_to_clear=["res_limited"],
        key_prefix=delete_key_prefix,
        expect="dir",
    )


def _render_hierarchical_results_browser(
    *,
    strategy_run_rows: list[dict[str, Any]],
    strategy_catalog: dict[str, dict[str, Any]],
) -> None:
    rows = [
        dict(row)
        for row in strategy_run_rows
        if str(row.get("workflow", "")).strip().lower() in RESULT_BROWSER_WORKFLOW_LABELS
    ]
    if not rows:
        st.info("No limited, walk-forward, or Monte Carlo runs have been discovered yet.")
        return

    st.caption("Browse results by strategy, workflow, scenario, dataset, and date window.")

    strategy_options = _ordered_manifest_values(rows, "strategy")
    selected_strategy = _select_browser_value(
        label="Strategy",
        options=strategy_options,
        key="results_browser_strategy",
        format_func=lambda key: _strategy_display_name(key, strategy_catalog),
    )
    strategy_rows = [row for row in rows if str(row.get("strategy", "")).strip() == selected_strategy]

    workflow_options = _ordered_manifest_values(strategy_rows, "workflow")
    selected_workflow = _select_browser_value(
        label="Workflow",
        options=workflow_options,
        key="results_browser_workflow",
        format_func=_workflow_browser_label,
    )
    workflow_rows = [
        row
        for row in strategy_rows
        if str(row.get("workflow", "")).strip().lower() == selected_workflow
    ]

    category_options = _ordered_manifest_values(workflow_rows, "category_label")
    selected_category = _select_browser_value(
        label="Scenario / profile",
        options=category_options,
        key="results_browser_category",
    )
    category_rows = [
        row
        for row in workflow_rows
        if str(row.get("category_label", "")).strip() == selected_category
    ]

    dataset_options = _ordered_manifest_values(category_rows, "dataset_slug")
    selected_dataset = _select_browser_value(
        label="Dataset",
        options=dataset_options,
        key="results_browser_dataset",
    )
    dataset_rows = [
        row
        for row in category_rows
        if str(row.get("dataset_slug", "")).strip() == selected_dataset
    ]

    window_options = _ordered_manifest_values(dataset_rows, "date_window")
    selected_window = _select_browser_value(
        label="Date window",
        options=window_options,
        key="results_browser_window",
    )
    window_rows = [
        row
        for row in dataset_rows
        if str(row.get("date_window", "")).strip() == selected_window
    ]

    run_options = [str(row.get("path", "")).strip() for row in window_rows if str(row.get("path", "")).strip()]
    run_lookup = {str(row.get("path", "")).strip(): row for row in window_rows if str(row.get("path", "")).strip()}
    selected_run = _select_browser_value(
        label="Run",
        options=run_options,
        key="results_browser_run",
        format_func=lambda path: (
            f"{run_lookup[path].get('created_at', '')} | "
            f"{run_lookup[path].get('result', '')} | "
            f"{run_lookup[path].get('run_id', '')}"
        ),
        help="Newest runs appear first within the current folder selection.",
    )
    selected_row = run_lookup.get(selected_run)
    if not selected_row:
        st.info("No run selected.")
        return

    workflow_key = str(selected_row.get("workflow", "")).strip().lower()
    breadcrumb = " / ".join(
        [
            _strategy_display_name(selected_strategy, strategy_catalog),
            _workflow_browser_label(workflow_key),
            selected_category,
            selected_dataset,
            selected_window,
            str(selected_row.get("run_id", "")).strip(),
        ]
    )
    st.caption(f"Path: {breadcrumb}")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Matching runs", len(window_rows))
    summary_cols[1].metric("Selected result", str(selected_row.get("result", "n/a")))
    summary_cols[2].metric("Created", str(selected_row.get("created_at", "n/a")))
    summary_cols[3].metric("Run ID", str(selected_row.get("run_id", "n/a")))

    sibling_rows = []
    for row in window_rows:
        raw_path = str(row.get("path", "")).strip()
        sibling_rows.append(
            {
                "Selected": "Yes" if raw_path == selected_run else "",
                "Result": str(row.get("result", "")).strip(),
                "Created": str(row.get("created_at", "")).strip(),
                "Run ID": str(row.get("run_id", "")).strip(),
                "Folder": _rel(_abs_path(raw_path)),
                "Open": str(row.get("result_url", "")).strip(),
            }
        )
    if sibling_rows:
        st.markdown("**Runs in this folder**")
        st.dataframe(
            pd.DataFrame(sibling_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Open": st.column_config.LinkColumn("Open", display_text="Open"),
            },
        )

    run_dir = _abs_path(selected_run)
    if not run_dir.exists():
        st.error(f"Selected run no longer exists: `{_rel(run_dir)}`")
        return

    st.markdown("---")
    _render_result_run_detail(
        workflow=workflow_key,
        run_dir=run_dir,
        strategy_catalog=strategy_catalog,
        delete_key_prefix=f"delete_browser_{_safe_key_from_path('run', run_dir)}",
    )


def _render_live_pilot_browser(*, mc_runs: list[Path], live_trade_files: list[Path]) -> None:
    mc_options = [p.as_posix() for p in mc_runs]
    mc_default = st.session_state["last_mc_run"] or (mc_options[0] if mc_options else "")
    mc_run_raw = _select_path_from_options("Monte Carlo run", mc_options, mc_default, key_prefix="res_live_mc")

    live_options = [_rel(p) for p in live_trade_files]
    live_default = st.session_state["last_live_trades_file"] or (live_options[0] if live_options else "")
    live_trades_raw = _select_path_from_options("Live trades CSV", live_options, live_default, key_prefix="res_live_trades")

    if mc_run_raw.strip() and live_trades_raw.strip():
        live_abs = _abs_path(live_trades_raw.strip())
        st.session_state["last_live_trades_file"] = live_trades_raw.strip()
        _render_live_pilot_results(mc_run_dir=_abs_path(mc_run_raw.strip()), live_trades_path=live_abs)
        delete_live_cols = st.columns(2, gap="large")
        with delete_live_cols[0]:
            _render_delete_target_ui(
                label="Monte Carlo run",
                raw_path=mc_run_raw,
                allowed_roots=[RUNS_ROOT / "strategies"],
                session_keys_to_clear=["last_mc_run"],
                widget_prefixes_to_clear=["res_mc", "res_live_mc"],
                key_prefix="delete_live_mc_run",
                expect="dir",
            )
        with delete_live_cols[1]:
            _render_delete_target_ui(
                label="live-trades CSV",
                raw_path=live_trades_raw,
                allowed_roots=[RUNS_ROOT, REPO_ROOT / "data"],
                session_keys_to_clear=["last_live_trades_file"],
                widget_prefixes_to_clear=["res_live_trades"],
                key_prefix="delete_live_trades_file",
                expect="file",
            )
    else:
        st.info("Select both a Monte Carlo run and a live-trades CSV.")


def _render_limited_workbook_summary(run_dir: Path, *, strategy_catalog: dict[str, dict[str, Any]]) -> None:
    """Render a compact summary of a limited test run, suitable for the run-test page."""
    try:
        summary = load_limited_summary(run_dir)
    except FileNotFoundError:
        st.warning("Incomplete limited test run outputs. Run the test to generate results.")
        return

    decision = summary.get("decision")
    criteria = summary.get("criteria") or {}
    pass_threshold = summary.get("pass_threshold")
    min_trades = summary.get("min_trades")
    is_monkey = summary.get("is_monkey", False)
    metrics = summary.get("metrics", {})

    _render_decision_badge("Limited test decision", decision)

    st.markdown("**Pass / threshold inputs used**")
    st.write(f"- Favourable criteria: `{json.dumps(criteria) if criteria else '{}'}'`")
    st.write(f"- Pass threshold (%): {pass_threshold}")
    st.write(f"- Min trades: {min_trades}")

    if is_monkey:
        _render_metric_row(
            [
                ("Total trials", metrics.get("total_trials"), "{:d}"),
                ("Return worse %", metrics.get("davey_return_worse_pct"), "{:.2f}"),
                ("MaxDD worse %", metrics.get("davey_maxdd_worse_pct"), "{:.2f}"),
                ("Pass threshold %", _as_float(pass_threshold), "{:.2f}"),
            ]
        )
    else:
        _render_metric_row(
            [
                ("Runs produced", metrics.get("total_iters"), "{:d}"),
                ("Profitable runs", metrics.get("profitable_runs"), "{:d}"),
                ("Median profit", metrics.get("median_profit"), "{:.2f}"),
                ("Median trades", metrics.get("median_trades"), "{:.1f}"),
            ]
        )

    st.subheader("Detailed iteration table")
    results_path = limited_iterations_path(run_dir)
    if not results_path.exists():
        st.info("Results CSV missing; re-run the test to generate iteration-level details.")
        return

    results = pd.read_csv(results_path)
    iter_cols = [
        c
        for c in [
            "iter",
            "favourable",
            "trades",
            "_net_profit_abs",
            "total_return_%",
            "max_drawdown_abs_%",
            "entry_params",
            "exit_params",
        ]
        if c in results.columns
    ]
    iter_view = results[iter_cols] if iter_cols else results
    _render_win_loss_dataframe(iter_view, metric_col="_net_profit_abs", hide_index=True)



def _render_walkforward_workbook_summary(run_dir: Path) -> None:
    try:
        summary_data = load_walkforward_summary(run_dir)
    except FileNotFoundError:
        st.warning("Incomplete walk-forward run outputs. Run the walk-forward test to generate summary.")
        return

    summary = summary_data.get("summary", {})
    segment_count = summary_data.get("segment_count", 0)
    profitable_pct = summary_data.get("profitable_pct", float("nan"))
    top2_share_pct = summary_data.get("top2_share_pct", float("nan"))

    _render_walkforward_workbook_review(summary)

    decision = _classify_walkforward_decision(
        segments=segment_count,
        profitable_oos_pct=profitable_pct,
        top2_share_pct=top2_share_pct,
    )
    _render_decision_badge("Walk-forward OOS Survival + Concentration", decision)

    threshold_df = pd.DataFrame(
        [
            {
                "Metric": "ProfitableOOS %",
                "Actual": "n/a" if not np.isfinite(profitable_pct) else f"{profitable_pct:.2f}%",
                "PASS": ">= 60%",
                "RETRY": "50% to <60%",
                "FAIL": "< 50%",
            },
            {
                "Metric": "Top2Share %",
                "Actual": "n/a" if not np.isfinite(top2_share_pct) else f"{top2_share_pct:.2f}%",
                "PASS": "<= 50%",
                "RETRY": ">50% to <=60%",
                "FAIL": "> 60%",
            },
            {
                "Metric": "Runs Produced",
                "Actual": f"{segment_count} OOS segments",
                "PASS": ">= 8 segments",
                "RETRY": "-",
                "FAIL": "< 8 segments",
            },
        ]
    )
    st.markdown("**Secondary diagnostics**")
    _render_verdict_dataframe(
        pd.DataFrame(
            [
                {
                    "Metric": row["Metric"],
                    "Actual": row["Actual"],
                    "Pass Criteria": row["PASS"],
                    "Retry": row["RETRY"],
                    "Fail": row["FAIL"],
                    "Verdict": (
                        "PASS"
                        if row["Metric"] == "ProfitableOOS %" and np.isfinite(profitable_pct) and profitable_pct >= 60.0
                        else (
                            "RETRY"
                            if row["Metric"] == "ProfitableOOS %" and np.isfinite(profitable_pct) and profitable_pct >= 50.0
                            else (
                                "FAIL"
                                if row["Metric"] == "ProfitableOOS %"
                                else (
                                    "PASS"
                                    if row["Metric"] == "Top2Share %" and np.isfinite(top2_share_pct) and top2_share_pct <= 50.0
                                    else (
                                        "RETRY"
                                        if row["Metric"] == "Top2Share %" and np.isfinite(top2_share_pct) and top2_share_pct <= 60.0
                                        else (
                                            "FAIL"
                                            if row["Metric"] == "Top2Share %"
                                            else ("PASS" if segment_count >= 8 else "FAIL")
                                        )
                                    )
                                )
                            )
                        )
                    ),
                }
                for _, row in threshold_df.iterrows()
            ]
        )
    )

    wfe = summary.get("wfe", {})
    if isinstance(wfe, dict) and wfe:
        st.markdown("**WFE summary (from run)**")
        st.write(f"- Metric: {wfe.get('metric', 'n/a')}")
        st.write(f"- Pass rate: {wfe.get('pass_rate_pct', float('nan')):.2f}%")
        st.write(f"- Pass folds: {wfe.get('pass_fold_count', 'n/a')} / {wfe.get('valid_fold_count', 'n/a')}")


def _render_montecarlo_workbook_summary(mc_run_dir: Path) -> None:
    try:
        data = load_montecarlo_summary(mc_run_dir)
    except FileNotFoundError:
        st.warning("Missing Monte Carlo summary; run Monte Carlo to generate it.")
        return

    metrics = data.get("metrics", {})
    thresholds = data.get("thresholds", {})
    checks = data.get("checks", {})

    passed = bool(checks.get("all_passed")) if isinstance(checks, dict) else False
    decision = "PASS" if passed else "FAIL"
    _render_decision_badge("Monte Carlo pass/fail", decision)

    st.markdown("**Monte Carlo thresholds and checks**")
    if thresholds:
        st.write(thresholds)
    if checks:
        st.write(checks)

    st.markdown("**Key Monte Carlo metrics**")
    _render_metric_row(
        [
            ("Risk of ruin %", _as_float(metrics.get("risk_of_ruin_pct")), "{:.2f}"),
            ("Median max DD %", _as_float(metrics.get("median_max_drawdown_%")), "{:.2f}"),
            ("Median return %", _as_float(metrics.get("median_return_%")), "{:.2f}"),
            ("Return/DD ratio", _as_float(metrics.get("return_drawdown_ratio_median_of_ratios")), "{:.2f}"),
        ]
    )


def _render_live_pilot_results(mc_run_dir: Path, live_trades_path: Path) -> None:
    mc_summary_path = summary_path(mc_run_dir)
    if not mc_summary_path.exists():
        st.error("Missing `summary.json` in selected Monte Carlo run.")
        return
    if not live_trades_path.exists():
        st.error(f"Live trades file not found: `{_rel(live_trades_path)}`")
        return

    mc_summary = _read_json(mc_summary_path)
    mc_cfg = mc_summary.get("config", {}) if isinstance(mc_summary.get("config"), dict) else {}
    initial_equity = _as_float(mc_cfg.get("initial_equity", 100000.0))
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        initial_equity = 100000.0

    wf_run_dir_raw = str(mc_summary.get("run_dir", "")).strip()
    if not wf_run_dir_raw:
        st.error("Monte Carlo summary is missing `run_dir`; cannot locate OOS trades.")
        return
    wf_run_dir = _abs_path(wf_run_dir_raw)
    oos_trades_path = walkforward_oos_trades_path(wf_run_dir)
    if not oos_trades_path.exists():
        st.error(f"Missing OOS trades for selected Monte Carlo run: `{_rel(oos_trades_path)}`")
        return

    live_df = pd.read_csv(live_trades_path)
    oos_df = pd.read_csv(oos_trades_path)

    live_pnl_col = ""
    for c in ("pnl", "net_profit", "profit", "net_profit_abs"):
        if c in live_df.columns:
            live_pnl_col = c
            break
    oos_pnl_col = ""
    for c in ("pnl", "net_profit", "profit", "net_profit_abs"):
        if c in oos_df.columns:
            oos_pnl_col = c
            break

    if not live_pnl_col:
        st.error("Live trades file is missing a PnL column. Expected one of: `pnl`, `net_profit`, `profit`, `net_profit_abs`.")
        return
    if not oos_pnl_col:
        st.error("OOS trades file is missing a PnL column. Expected one of: `pnl`, `net_profit`, `profit`, `net_profit_abs`.")
        return

    live_pnl = pd.to_numeric(live_df[live_pnl_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    oos_pnl = pd.to_numeric(oos_df[oos_pnl_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if live_pnl.empty:
        st.error("No valid live-trade PnL values found.")
        return
    if oos_pnl.empty:
        st.error("No valid OOS-trade PnL values found in selected walk-forward run.")
        return

    target_trades = 50
    n_sims = 10000
    seed = int(mc_cfg.get("seed", 42))
    oos_pool = oos_pnl.to_numpy(dtype=float)
    dist50 = _mc_profit_distribution_from_pool(oos_pool, n_trades=target_trades, n_sims=n_sims, seed=seed)

    band_p5 = float(np.quantile(dist50, 0.05)) if dist50.size else float("nan")
    band_p95 = float(np.quantile(dist50, 0.95)) if dist50.size else float("nan")
    band_min = float(np.min(dist50)) if dist50.size else float("nan")
    band_max = float(np.max(dist50)) if dist50.size else float("nan")
    live_net_profit_50 = float(live_pnl.iloc[:target_trades].sum()) if len(live_pnl) >= target_trades else float("nan")

    decision = _classify_live_pilot_decision(
        live_net_profit=live_net_profit_50,
        band_p5=band_p5,
        band_p95=band_p95,
        initial_equity=initial_equity,
    )
    _render_decision_badge("Live Pilot Test (Execution Drift)", decision)

    _render_metric_row(
        [
            ("Live Trades Available", int(len(live_pnl)), "{:d}"),
            ("LiveNetProfit @50", live_net_profit_50, "{:.2f}"),
            ("MC Band P5", band_p5, "{:.2f}"),
            ("MC Band P95", band_p95, "{:.2f}"),
        ]
    )
    _render_metric_row(
        [
            ("MC Band Min", band_min, "{:.2f}"),
            ("MC Band Max", band_max, "{:.2f}"),
            ("Initial Equity", initial_equity, "{:.2f}"),
            ("MC Sims (50-trade)", n_sims, "{:d}"),
        ]
    )

    threshold_df = pd.DataFrame(
        [
            {
                "Metric": "LiveNetProfit @50 vs MC 5th-95th",
                "Actual": "n/a" if not np.isfinite(live_net_profit_50) else f"{live_net_profit_50:,.2f}",
                "PASS": "inside [P5, P95]",
                "RETRY": "outside [P5, P95] but not beyond 5% equity drift",
                "FAIL": "outside band by >5% of initial equity",
            },
            {
                "Metric": "Runs Produced",
                "Actual": f"{len(live_pnl)} live trades",
                "PASS": "50 trades",
                "RETRY": "extend once to 100 trades",
                "FAIL": "fail early / severe drift",
            },
        ]
    )
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    if decision == "RETRY":
        st.markdown("**Retry Check (100 trades, run once)**")
        if len(live_pnl) < 100:
            st.warning("Need at least 100 live trades to run the retry check.")
        else:
            dist100 = _mc_profit_distribution_from_pool(oos_pool, n_trades=100, n_sims=n_sims, seed=seed + 1)
            band100_p5 = float(np.quantile(dist100, 0.05)) if dist100.size else float("nan")
            band100_p95 = float(np.quantile(dist100, 0.95)) if dist100.size else float("nan")
            live_net_profit_100 = float(live_pnl.iloc[:100].sum())
            retry_pass = bool(np.isfinite(live_net_profit_100) and np.isfinite(band100_p5) and np.isfinite(band100_p95) and band100_p5 <= live_net_profit_100 <= band100_p95)
            _render_status_badge("Retry Result (100-trade band)", retry_pass)
            _render_metric_row(
                [
                    ("LiveNetProfit @100", live_net_profit_100, "{:.2f}"),
                    ("MC100 P5", band100_p5, "{:.2f}"),
                    ("MC100 P95", band100_p95, "{:.2f}"),
                    ("Retry PASS", _status_label(retry_pass), "{}"),
                ]
            )

    st.subheader("Monte Carlo Net-Profit Distribution (50 trades)")
    _render_histogram(pd.Series(dist50), title="MC Net-Profit Distribution (50 trades)", bins=70)


def main() -> None:
    st.set_page_config(
        page_title="QuantBT Streamlit Frontend",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_app_css()
    st.title("QuantBT Streamlit Frontend")
    c_meta, c_refresh = st.columns([0.8, 0.2])
    c_meta.caption(f"Script interpreter: `{SCRIPT_PYTHON}`")
    if c_refresh.button("Refresh discovered metadata"):
        st.cache_data.clear()
        st.rerun()

    for k in (
        "last_downloaded_dataset",
        "last_walkforward_run",
        "last_mc_run",
        "last_limited_run",
        "last_live_trades_file",
    ):
        st.session_state.setdefault(k, "")

    strategy_catalog = _discover_strategy_catalog(_strategy_catalog_fingerprint())
    strategy_run_rows = _annotate_strategy_run_rows(
        _discover_run_manifest_rows(_run_manifest_fingerprint())
    )
    plugin_catalog = _discover_plugin_catalog()
    data_files = _list_csv_data_files()
    walk_runs = _discover_walkforward_runs()
    mc_runs = _discover_mc_runs()
    limited_runs = _discover_limited_runs()
    live_trade_files = _discover_live_trade_files()

    fallback_dataset = "data/processed/eurusd_1h_20100101_20260209_dukascopy_python.csv"
    if data_files:
        default_dataset = _rel(data_files[0])
    else:
        default_dataset = fallback_dataset

    page_options = {
        "download_data": "Download data",
        "run_test": "Run test",
        "results": "Results",
        "reference": "Strategies",
    }
    requested_page = _qp_get("page", "run_test")
    if requested_page == "html_view":
        _render_html_view_page()
        return
    keys = list(page_options.keys())
    if requested_page not in keys:
        requested_page = "run_test"

    nav_key = "nav_page"
    last_qp_key = "_last_nav_qp_page"
    forced_page = st.session_state.pop("_force_nav_page", None)
    if forced_page in keys:
        requested_page = str(forced_page)
        st.session_state[nav_key] = requested_page
        st.session_state[last_qp_key] = requested_page
        try:
            st.query_params["page"] = requested_page
        except Exception:
            pass

    if nav_key not in st.session_state or st.session_state[nav_key] not in keys:
        st.session_state[nav_key] = requested_page

    last_qp_seen = st.session_state.get(last_qp_key)
    if last_qp_seen is None:
        st.session_state[last_qp_key] = requested_page
    elif requested_page in keys and requested_page != last_qp_seen:
        # Query param changed externally (e.g. markdown link); adopt it once.
        st.session_state[nav_key] = requested_page
        st.session_state[last_qp_key] = requested_page

    page = st.radio(
        "Navigation",
        options=keys,
        key=nav_key,
        format_func=lambda k: page_options[k],
        horizontal=True,
    )

    if _qp_get("page", "") != page:
        try:
            st.query_params["page"] = page
        except Exception:
            pass
    st.session_state[last_qp_key] = page

    if page == "download_data":
        st.header("Download data")
        st.markdown(
            "Download historical data to run tests against."
        )

        provider = st.selectbox(
            "Provider",
            options=("dukascopy", "mt5_ftmo"),
            format_func=lambda p: "Dukascopy FX" if p == "dukascopy" else "MT5 / FTMO",
            key="download_provider",
        )

        symbol = "EURUSD"
        timeframe = "1H"
        mt5_backend = "silicon"
        mt5_host = "localhost"
        mt5_port = 8001
        mt5_login_raw = ""
        mt5_password = ""
        mt5_server = ""
        mt5_terminal_path = ""
        mt5_batch_size = 1000
        mt5_max_backfill_batches = 15000
        mt5_progress_every = 25
        mt5_allow_incomplete = False

        min_download_date = date(1990, 1, 1)
        max_download_date = date.today()
        c_date_1, c_date_2 = st.columns(2)
        start_date = c_date_1.date_input(
            "Start date",
            value=date(2010, 1, 1),
            min_value=min_download_date,
            max_value=max_download_date,
            key="download_start_date",
        )
        end_date = c_date_2.date_input(
            "End date",
            value=max_download_date,
            min_value=min_download_date,
            max_value=max_download_date,
            key="download_end_date",
        )

        if provider == "dukascopy":
            download_symbols = _discover_download_symbols()
            download_timeframes = _discover_download_timeframes()
            default_idx = download_symbols.index("EURUSD") if "EURUSD" in download_symbols else 0
            default_tf_idx = download_timeframes.index("1H") if "1H" in download_timeframes else 0
            c1, c2 = st.columns(2)
            symbol = c1.selectbox("Symbol", download_symbols, index=default_idx, key="download_symbol_dk")
            timeframe = c2.selectbox("Timeframe", download_timeframes, index=default_tf_idx, key="download_tf_dk")
            st.caption(
                f"Available Dukascopy FX symbols: {len(download_symbols)} | "
                f"timeframes: {len(download_timeframes)}"
            )
        else:
            c1, c2, c3 = st.columns(3)
            mt5_backend = c1.selectbox(
                "MT5 backend",
                options=("silicon", "auto", "native"),
                index=0,
                key="download_mt5_backend",
            )
            mt5_host = c2.text_input("MT5 host", value="localhost", key="download_mt5_host").strip() or "localhost"
            mt5_port = int(c3.number_input("MT5 port", min_value=1, max_value=65535, value=8001, step=1, key="download_mt5_port"))

            mt5_timeframes = _discover_mt5_download_timeframes(mt5_backend, mt5_host, mt5_port)
            default_mt5_tf_idx = mt5_timeframes.index("M5") if "M5" in mt5_timeframes else 0
            c4, c5 = st.columns(2)
            symbol = c4.text_input("Symbol", value="EURUSD", key="download_symbol_mt5").strip().upper() or "EURUSD"
            timeframe = c5.selectbox("Timeframe", mt5_timeframes, index=default_mt5_tf_idx, key="download_tf_mt5")

            c6, c7, c8 = st.columns(3)
            mt5_login_raw = c6.text_input("MT5 login (optional)", value="", key="download_mt5_login")
            mt5_password = c7.text_input("MT5 password (optional)", value="", type="password", key="download_mt5_password")
            mt5_server = c8.text_input("MT5 server (optional)", value="", key="download_mt5_server")
            mt5_terminal_path = st.text_input("MT5 terminal path (native backend only)", value="", key="download_mt5_terminal_path")

            c9, c10, c11, c12 = st.columns(4)
            mt5_batch_size = int(c9.number_input("Batch size", min_value=10, max_value=20000, value=1000, step=10, key="download_mt5_batch_size"))
            mt5_max_backfill_batches = int(
                c10.number_input(
                    "Max backfill batches",
                    min_value=10,
                    max_value=200000,
                    value=15000,
                    step=100,
                    key="download_mt5_max_backfill_batches",
                )
            )
            mt5_progress_every = int(
                c11.number_input(
                    "Progress every N batches",
                    min_value=0,
                    max_value=10000,
                    value=25,
                    step=1,
                    key="download_mt5_progress_every",
                )
            )
            mt5_allow_incomplete = c12.checkbox("Allow incomplete", value=False, key="download_mt5_allow_incomplete")
            st.caption(f"Detected MT5 timeframes: {len(mt5_timeframes)} (from connected backend if available)")

        c_common_1, c_common_2, c_common_3 = st.columns(3)
        file_ext = c_common_1.selectbox("File extension", ("csv", "parquet"), key="download_file_ext")
        save_dir = c_common_2.text_input("Save directory", value="data/processed", key="download_save_dir")
        extra_args = c_common_3.text_input("Extra args (optional)", value="", key="download_extra_args")

        if st.button("Download data", type="primary"):
            try:
                if start_date >= end_date:
                    raise ValueError("Start date must be earlier than end date")

                cmd = [
                    SCRIPT_PYTHON,
                    str(SCRIPTS_DIR / "download_data.py"),
                    "--provider",
                    provider,
                    "--symbol",
                    symbol.strip().upper(),
                    "--timeframe",
                    timeframe,
                    "--start",
                    start_date.isoformat(),
                    "--end",
                    end_date.isoformat(),
                    "--save-dir",
                    save_dir.strip(),
                    "--file-ext",
                    file_ext,
                ]
                if provider == "mt5_ftmo":
                    cmd.extend(
                        [
                            "--mt5-backend",
                            mt5_backend,
                            "--mt5-host",
                            mt5_host,
                            "--mt5-port",
                            str(int(mt5_port)),
                            "--mt5-batch-size",
                            str(int(mt5_batch_size)),
                            "--mt5-max-backfill-batches",
                            str(int(mt5_max_backfill_batches)),
                            "--mt5-progress-every",
                            str(int(mt5_progress_every)),
                        ]
                    )
                    if mt5_login_raw.strip():
                        try:
                            int(mt5_login_raw.strip())
                        except ValueError as e:
                            raise ValueError("MT5 login must be an integer when provided.") from e
                        cmd.extend(["--mt5-login", mt5_login_raw.strip()])
                    if mt5_password.strip():
                        cmd.extend(["--mt5-password", mt5_password.strip()])
                    if mt5_server.strip():
                        cmd.extend(["--mt5-server", mt5_server.strip()])
                    if mt5_terminal_path.strip():
                        cmd.extend(["--mt5-terminal-path", mt5_terminal_path.strip()])
                    if mt5_allow_incomplete:
                        cmd.append("--mt5-allow-incomplete")
                cmd.extend(_parse_extra_args(extra_args))
            except ValueError as e:
                st.error(f"Invalid arguments: {e}")
            else:
                rc, out = _run_cli_live(cmd, workflow="download")
                if rc == 0:
                    dataset_path = _extract_path_after_marker(out, "rows to:")
                    if dataset_path is None:
                        dataset_path = _extract_path_after_marker(out, "Saved dataset:")
                    meta_path = _extract_path_after_marker(out, "Metadata:")
                    if dataset_path is not None:
                        st.session_state["last_downloaded_dataset"] = dataset_path.as_posix()
                        st.success(f"Dataset saved: {_rel(dataset_path)}")
                    if meta_path is not None:
                        st.info(f"Metadata saved: {_rel(meta_path)}")

    elif page == "run_test":
        st.header("Run tests")
        st.markdown(
            "Use this page to run limited tests, walk-forward analysis, and Monte Carlo."
        )

        st.subheader("1) Dataset")
        dataset_options = [_rel(p) for p in data_files]
        default_wb_dataset = st.session_state.get("wb_dataset", default_dataset)
        wb_dataset = st.selectbox(
            "Dataset CSV",
            dataset_options,
            index=dataset_options.index(default_wb_dataset) if default_wb_dataset in dataset_options else 0,
            key="wb_dataset",
        )
        st.markdown(f"**Selected dataset:** `{wb_dataset}`")

        t_limited, t_wf, t_mc = st.tabs(["Limited tests", "Walk-forward", "Monte Carlo"])

        with t_wf:
            strategy_names = _sorted_strategy_keys(strategy_catalog)
            default_strategy_idx = strategy_names.index("sma_cross_strategy") if "sma_cross_strategy" in strategy_names else 0
            st.subheader("2) Strategy + Dataset")
            setup_left, setup_right = st.columns([0.9, 1.1], gap="large")
            with setup_left:
                strategy = st.selectbox(
                    "Strategy",
                    strategy_names,
                    index=default_strategy_idx,
                    key="wf_strategy",
                    format_func=lambda key: _strategy_display_name(key, strategy_catalog),
                )
                st.markdown(_reference_link("View strategy config", "strategy", strategy))
            with setup_right:
                dataset_options = [_rel(p) for p in data_files]
                dataset = _select_path_from_options("Dataset CSV", dataset_options, wb_dataset, key_prefix="wf_dataset")
                st.caption(f"Dataset comes from the page-level selector: `{dataset}`")

            info = strategy_catalog.get(strategy, {})
            strategy_cfg = info.get("strategy_config", {}) if isinstance(info.get("strategy_config"), dict) else {}
            params_defaults = copy.deepcopy(info.get("params_defaults", {}) or {})
            wf_param_contract = _build_strategy_walkforward_param_contract(
                strategy_cfg,
                params_defaults=params_defaults,
            )

            ts_col_default = str(st.session_state.get("wf_ts_col", "timestamp") or "timestamp").strip() or "timestamp"
            bounds = _dataset_timestamp_bounds(dataset, ts_col_default)
            min_window_date = bounds.get("min_date") or date(2010, 1, 1)
            max_window_date = bounds.get("max_date") or min_window_date
            window_sig = f"{dataset}|{ts_col_default}|{min_window_date}|{max_window_date}"
            if st.session_state.get("_wf_window_sig") != window_sig:
                st.session_state["_wf_window_sig"] = window_sig
                st.session_state["wf_window_start_date"] = min_window_date
                st.session_state["wf_window_end_date"] = max_window_date

            st.subheader("3) Workbook Inputs")
            st.caption("The main WFA form follows the workbook fields. Lower-level optimizer and execution controls are below in Advanced.")

            if bounds.get("ok", False):
                st.caption(f"Detected dataset window: {min_window_date.isoformat()} to {max_window_date.isoformat()}")
            elif bounds.get("min_date") is not None and bounds.get("max_date") is not None:
                st.warning(
                    f"Timestamp parsing from `{ts_col_default}` failed; using dataset metadata bounds instead. "
                    f"Details: {bounds.get('error', 'unknown error')}"
                )
            else:
                st.error(
                    f"Unable to determine WFA date bounds from dataset `{dataset}` using timestamp column `{ts_col_default}`."
                )

            window_cols = st.columns(2)
            start_date = window_cols[0].date_input(
                "WFA data start",
                min_value=min_window_date,
                max_value=max_window_date,
                key="wf_window_start_date",
            )
            end_date = window_cols[1].date_input(
                "WFA data end",
                min_value=min_window_date,
                max_value=max_window_date,
                key="wf_window_end_date",
            )

            config_cols = st.columns(4)
            anchored_mode = config_cols[0].selectbox(
                "Anchored / unanchored",
                options=("unanchored", "anchored"),
                index=0,
                key="wf_window_mode",
            )
            in_period_years = float(
                config_cols[1].number_input(
                    "In-period years",
                    min_value=0.25,
                    value=4.0,
                    step=0.25,
                    format="%.2f",
                    key="wf_in_period_years",
                )
            )
            out_period_years = float(
                config_cols[2].number_input(
                    "Out-period years",
                    min_value=0.25,
                    value=1.0,
                    step=0.25,
                    format="%.2f",
                    key="wf_out_period_years",
                )
            )
            fitness_factor_label = config_cols[3].selectbox(
                "Fitness factor",
                options=list(WFA_FITNESS_FACTOR_PRESETS.keys()),
                index=0,
                key="wf_fitness_factor_label",
            )
            fitness_factor_cfg = dict(WFA_FITNESS_FACTOR_PRESETS.get(fitness_factor_label, {}))
            objective = str(fitness_factor_cfg.get("objective", "net_profit_abs") or "net_profit_abs")
            direction = str(fitness_factor_cfg.get("direction", "maximize") or "maximize")

            cost_cols = st.columns(2)
            spread_pips = float(
                cost_cols[0].number_input(
                    "Slippage / spread (pips)",
                    min_value=0.0,
                    value=0.2,
                    step=0.1,
                    key="wf_spread_pips",
                )
            )
            commission_rt = float(
                cost_cols[1].number_input(
                    "Commissions (round trip)",
                    min_value=0.0,
                    value=5.0,
                    step=0.5,
                    key="wf_commission_rt",
                )
            )

            window_error = ""
            window_info: dict[str, Any] | None = None
            is_bars = 0
            oos_bars = 0
            try:
                window_info = _walkforward_window_from_dates(
                    bounds=bounds,
                    start_value=start_date,
                    end_value=end_date,
                )
                bars_per_year = float(window_info.get("bars_per_year", float("nan")))
                if not math.isfinite(bars_per_year) or bars_per_year <= 0:
                    raise ValueError("Unable to derive bars per year from the selected WFA data window.")
                is_bars = max(1, int(round(float(in_period_years) * bars_per_year)))
                oos_bars = max(1, int(round(float(out_period_years) * bars_per_year)))
            except Exception as e:
                window_error = str(e)
                st.error(window_error)

            step_bars_override = 0
            effective_step_bars = oos_bars
            estimated_folds = 0
            if window_info is not None and not window_error:
                selected_rows = int(window_info.get("selected_rows", 0) or 0)
                effective_step_bars = max(1, oos_bars)
                if selected_rows >= (is_bars + oos_bars):
                    estimated_folds = 1 + max(0, int((selected_rows - is_bars - oos_bars) // effective_step_bars))
                _render_metric_row(
                    [
                        ("Selected rows", selected_rows, "{:,.0f}"),
                        ("Bars / year", float(window_info.get("bars_per_year", float("nan"))), "{:,.0f}"),
                        ("IS bars", is_bars, "{:,.0f}"),
                        ("OOS bars", oos_bars, "{:,.0f}"),
                        ("Estimated folds", estimated_folds, "{:,.0f}"),
                    ]
                )

            st.subheader("Baseline entry / exit param set")
            wf_param_checks: list[tuple[bool, str, str]] = []
            wf_param_space_obj = copy.deepcopy(wf_param_contract.get("default_param_space", {}) or {})

            if wf_param_contract.get("has_policy", False):
                sections = dict(wf_param_contract.get("sections", {}))
                entry_section = sections.get("entry", {})
                exit_section = sections.get("exit", {})
                entry_has_content = bool(entry_section.get("editable_specs") or entry_section.get("fixed_params"))
                exit_has_content = bool(exit_section.get("editable_specs") or exit_section.get("fixed_params"))

                if entry_has_content or exit_has_content:
                    param_cols = st.columns(2, gap="large")
                    with param_cols[0]:
                        if entry_has_content:
                            wf_param_space_obj = _render_walkforward_param_policy_section(
                                title="Baseline entry param set",
                                section_contract=entry_section,
                                param_space=wf_param_space_obj,
                                widget_prefix=f"wf_param_entry_{strategy}",
                                checks=wf_param_checks,
                            )
                        else:
                            st.markdown("**Baseline entry param set**")
                            st.caption("No strategy-defined entry parameters are configurable for walk-forward.")
                    with param_cols[1]:
                        if exit_has_content:
                            wf_param_space_obj = _render_walkforward_param_policy_section(
                                title="Baseline exit param set",
                                section_contract=exit_section,
                                param_space=wf_param_space_obj,
                                widget_prefix=f"wf_param_exit_{strategy}",
                                checks=wf_param_checks,
                            )
                        else:
                            st.markdown("**Baseline exit param set**")
                            st.caption("No strategy-defined exit parameters are configurable for walk-forward.")
                else:
                    fallback_section = sections.get("strategy", {})
                    wf_param_space_obj = _render_walkforward_param_policy_section(
                        title="Baseline strategy parameter set",
                        section_contract=fallback_section,
                        param_space=wf_param_space_obj,
                        widget_prefix=f"wf_param_strategy_{strategy}",
                        checks=wf_param_checks,
                    )
            else:
                st.info(
                    "This strategy does not define structured walk-forward optimisation metadata. "
                    "Use the advanced JSON override below."
                )

            generated_param_space_text = _json_pretty(wf_param_space_obj)
            with st.expander("Advanced param-space override", expanded=False):
                use_param_space_override = st.checkbox(
                    "Use custom param-space JSON instead of the structured fields above",
                    value=False,
                    key=f"wf_use_param_space_override_{strategy}",
                )
                if use_param_space_override:
                    _sync_widget_from_source(f"wf_param_space_override_{strategy}", generated_param_space_text)
                    param_space_text = st.text_area(
                        "Custom param space JSON",
                        height=180,
                        key=f"wf_param_space_override_{strategy}",
                    )
                else:
                    param_space_text = generated_param_space_text
                    st.code(generated_param_space_text, language="json")

            st.subheader("4) Workbook Outputs + Pass Criteria")
            review_defaults = copy.deepcopy(WFA_WORKBOOK_REVIEW_ROWS)
            review_seed_rows = copy.deepcopy(st.session_state.get("wf_workbook_review_rows", []))
            if not review_seed_rows:
                review_seed_rows = copy.deepcopy(review_defaults)
            review_header_cols = st.columns([0.78, 0.22])
            with review_header_cols[0]:
                st.caption("Workbook-aligned review checklist. Edit outputs or review thresholds here if this run needs a different checklist.")
            with review_header_cols[1]:
                if st.button("Reset to workbook defaults", key="wf_workbook_review_reset"):
                    st.session_state["wf_workbook_review_rows"] = copy.deepcopy(review_defaults)
                    st.session_state.pop("wf_workbook_review_rows_editor", None)
                    st.rerun()

            review_editor_df = st.data_editor(
                pd.DataFrame(review_seed_rows, columns=["output", "pass_criteria"]),
                key="wf_workbook_review_rows_editor",
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "output": st.column_config.TextColumn("Output", width="large"),
                    "pass_criteria": st.column_config.TextColumn("Pass criteria", width="large"),
                },
            )
            wf_workbook_review_rows = [
                {
                    "output": str(row.get("output", "") or "").strip(),
                    "pass_criteria": str(row.get("pass_criteria", "") or "").strip(),
                }
                for row in review_editor_df.to_dict("records")
                if str(row.get("output", "") or "").strip() or str(row.get("pass_criteria", "") or "").strip()
            ]
            st.session_state["wf_workbook_review_rows"] = wf_workbook_review_rows

            st.subheader("5) Advanced / technical overrides")
            with st.expander("Advanced / technical overrides", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                optimizer = c1.selectbox("Optimizer", ("grid", "optuna"), key="wf_optimizer")
                optimization_mode = c2.selectbox(
                    "Optimization mode",
                    ("peak", "stability_robustness"),
                    key="wf_optimization_mode",
                    help="`stability_robustness` emphasizes parameter stability, WFE, and anti-outlier filters.",
                )
                selection_mode = c3.selectbox("Selection mode", ("peak", "plateau"), key="wf_selection_mode")
                sampler = c4.selectbox("Sampler", ("tpe", "random"), key="wf_sampler")

                c5, c6, c7, c8 = st.columns(4)
                use_objective_override = c5.checkbox(
                    "Override fitness objective",
                    value=False,
                    key="wf_use_objective_override",
                )
                objective_choice = c6.selectbox(
                    "Objective",
                    [*COMMON_WF_OBJECTIVES, "profit_factor", "custom..."],
                    key="wf_objective_choice",
                    disabled=not use_objective_override,
                )
                if use_objective_override and objective_choice == "custom...":
                    objective = c6.text_input("Custom objective", value="return_on_account", key="wf_custom_objective")
                elif use_objective_override:
                    objective = objective_choice
                c7.caption(f"Fitness preset objective: `{fitness_factor_label}` -> `{objective}`")
                if use_objective_override:
                    direction = c8.selectbox("Direction", ("maximize", "minimize"), key="wf_direction")
                else:
                    c8.caption(f"Direction from fitness preset: `{direction}`")

                c9, c10, c11, c12 = st.columns(4)
                wfe_metric_choice = c9.selectbox(
                    "WFE metric",
                    [*COMMON_WF_OBJECTIVES, "profit_factor", "custom..."],
                    index=1 if "total_return_%" in COMMON_WF_OBJECTIVES else 0,
                    key="wf_wfe_metric_choice",
                )
                if wfe_metric_choice == "custom...":
                    wfe_metric = c9.text_input("Custom WFE metric", value="total_return_%", key="wf_custom_wfe_metric")
                else:
                    wfe_metric = wfe_metric_choice
                step_bars_override = int(
                    c10.number_input(
                        "Step bars override (0 = out-period bars)",
                        min_value=0,
                        value=0,
                        step=100,
                        key="wf_step_bars_override",
                    )
                )
                run_base = c11.text_input("Run base", value="runs", key="wf_run_base")
                ts_col = c12.text_input("Timestamp column", value=ts_col_default, key="wf_ts_col")

                c13, c14, c15 = st.columns(3)
                min_trades = int(c13.number_input("Min trades (fallback)", min_value=0, value=30, step=1, key="wf_min_trades"))
                min_is_trades = int(c14.number_input("Min IS trades", min_value=0, value=30, step=1, key="wf_min_is_trades"))
                min_oos_trades = int(c15.number_input("Min OOS trades", min_value=0, value=30, step=1, key="wf_min_oos_trades"))

                c16, c17, c18 = st.columns(3)
                wfe_min_pct = float(c16.number_input("WFE min %", min_value=0.0, value=0.0, step=1.0, format="%.1f", key="wf_wfe_min_pct"))
                max_top_trade_share = float(
                    c17.number_input(
                        "Max top trade share",
                        min_value=0.01,
                        max_value=1.0,
                        value=1.0,
                        step=0.01,
                        format="%.2f",
                        key="wf_max_top_trade_share",
                    )
                )
                plateau_min_neighbors = int(
                    c18.number_input("Plateau min neighbors", min_value=0, value=3, step=1, key="wf_plateau_min_neighbors")
                )

                c19, c20, c21 = st.columns(3)
                plateau_stability_penalty = float(
                    c19.number_input(
                        "Plateau stability penalty",
                        min_value=0.0,
                        value=0.5,
                        step=0.1,
                        format="%.2f",
                        key="wf_plateau_stability_penalty",
                    )
                )
                progress_every = int(
                    c20.number_input("Progress every", min_value=1, value=20, step=1, key="wf_progress_every")
                )
                seed = int(c21.number_input("Seed", min_value=0, value=42, step=1, key="wf_seed"))

                c22, c23, c24 = st.columns(3)
                initial_equity = float(
                    c22.number_input("Initial equity", min_value=0.0, value=100000.0, step=1000.0, key="wf_initial_equity")
                )
                risk_pct = float(
                    c23.number_input("Risk %", min_value=0.0, value=0.01, step=0.001, format="%.4f", key="wf_risk_pct")
                )
                pip_size = float(
                    c24.number_input("Pip size", min_value=0.0, value=0.0001, format="%.6f", key="wf_pip_size")
                )

                c25, c26, c27 = st.columns(3)
                margin_rate = float(
                    c25.number_input("Margin rate", min_value=0.0, value=0.0, step=0.001, format="%.4f", key="wf_margin_rate")
                )
                lot_size = float(
                    c26.number_input("Lot size", min_value=1.0, value=100000.0, step=1000.0, key="wf_lot_size")
                )
                required_margin_abs = c27.text_input("Required margin abs (optional)", value="", key="wf_required_margin_abs")

                c28, c29, c30 = st.columns(3)
                n_trials = int(c28.number_input("Optuna n-trials", min_value=1, value=200, step=1, key="wf_n_trials"))
                timeout_s = int(c29.number_input("Optuna timeout (sec)", min_value=0, value=0, step=10, key="wf_timeout_s"))
                extra_args = c30.text_input("Extra args (optional)", value="", key="wf_extra_args")

                c31, c32, c33 = st.columns(3)
                baseline_full_data = c31.checkbox("Run baseline full-data compare", value=True, key="wf_baseline_full_data")
                compound_oos = c32.checkbox("Compound OOS equity", value=True, key="wf_compound_oos")
                conservative_same_bar = c33.checkbox(
                    "Conservative same-bar handling",
                    value=False,
                    key="wf_conservative_same_bar",
                )
                if optimization_mode == "stability_robustness":
                    st.caption(
                        "stability_robustness mode enforces stricter floors: "
                        "min IS/OOS trades >= 50, WFE >= 50%, top-trade-share <= 0.30."
                    )

            effective_step_bars = max(1, step_bars_override or oos_bars or 1)
            if window_info is not None and not window_error:
                selected_rows = int(window_info.get("selected_rows", 0) or 0)
                if selected_rows >= (is_bars + oos_bars):
                    estimated_folds = 1 + max(0, int((selected_rows - is_bars - oos_bars) // effective_step_bars))
                else:
                    estimated_folds = 0

            st.subheader("6) Run")
            _render_metric_row(
                [
                    ("Strategy", _strategy_display_name(strategy, strategy_catalog), "{}"),
                    ("Fitness factor", fitness_factor_label, "{}"),
                    ("Objective", objective, "{}"),
                    ("Step bars", effective_step_bars, "{:,.0f}"),
                    ("Estimated folds", estimated_folds, "{:,.0f}"),
                ]
            )

            if st.button("Run walk-forward", type="primary"):
                try:
                    if window_error:
                        raise ValueError(window_error)
                    failed_param_checks = [item for item in wf_param_checks if not bool(item[0])]
                    if use_param_space_override:
                        custom_param_space = json.loads(str(param_space_text or "").strip() or "{}")
                        if not isinstance(custom_param_space, dict):
                            raise ValueError("Custom param space JSON must be an object.")
                        param_space_payload = json.dumps(custom_param_space, sort_keys=True, separators=(",", ":"), default=str)
                    else:
                        if failed_param_checks:
                            first_bad = failed_param_checks[0]
                            raise ValueError(f"Invalid walk-forward param space: {first_bad[1]} ({first_bad[2]})")
                        param_space_payload = json.dumps(wf_param_space_obj, sort_keys=True, separators=(",", ":"), default=str)
                    if window_info is None:
                        raise ValueError("WFA data window is not valid.")
                    cmd = [
                        SCRIPT_PYTHON,
                        str(SCRIPTS_DIR / "run_walkforward.py"),
                        "--strategy",
                        strategy,
                        "--dataset",
                        dataset,
                        "--ts-col",
                        ts_col,
                        "--run-base",
                        run_base,
                        "--optimizer",
                        optimizer,
                        "--optimization-mode",
                        optimization_mode,
                        "--selection-mode",
                        selection_mode,
                        "--objective",
                        objective,
                        "--direction",
                        direction,
                        "--min-trades",
                        str(min_trades),
                        "--min-is-trades",
                        str(min_is_trades),
                        "--min-oos-trades",
                        str(min_oos_trades),
                        "--max-top-trade-share",
                        str(max_top_trade_share),
                        "--wfe-metric",
                        wfe_metric,
                        "--wfe-min-pct",
                        str(wfe_min_pct),
                        "--plateau-min-neighbors",
                        str(plateau_min_neighbors),
                        "--plateau-stability-penalty",
                        str(plateau_stability_penalty),
                        "--is-bars",
                        str(is_bars),
                        "--oos-bars",
                        str(oos_bars),
                        "--start-bar",
                        str(int(window_info.get("start_bar", 0) or 0)),
                        "--end-bar",
                        str(int(window_info.get("end_bar", 0) or 0)),
                        "--n-trials",
                        str(n_trials),
                        "--timeout-s",
                        str(timeout_s),
                        "--sampler",
                        sampler,
                        "--seed",
                        str(seed),
                        "--progress-every",
                        str(progress_every),
                        "--initial-equity",
                        str(initial_equity),
                        "--risk-pct",
                        str(risk_pct),
                        "--spread-pips",
                        str(spread_pips),
                        "--pip-size",
                        str(pip_size),
                        "--commission-rt",
                        str(commission_rt),
                        "--lot-size",
                        str(lot_size),
                        "--margin-rate",
                        str(margin_rate),
                    ]
                    if anchored_mode == "unanchored":
                        cmd.append("--unanchored")
                    else:
                        cmd.append("--anchored")
                    if effective_step_bars > 0:
                        cmd.extend(["--step-bars", str(effective_step_bars)])
                    if required_margin_abs.strip():
                        cmd.extend(["--required-margin-abs", required_margin_abs.strip()])
                    cmd.extend(["--param-space", param_space_payload])
                    if not baseline_full_data:
                        cmd.append("--no-baseline-full-data")
                    if not compound_oos:
                        cmd.append("--no-compound-oos")
                    if conservative_same_bar:
                        cmd.append("--conservative-same-bar")
                    cmd.extend(_parse_extra_args(extra_args))
                except ValueError as e:
                    st.error(f"Invalid arguments: {e}")
                else:
                    rc, out = _run_cli_live(cmd, workflow="walkforward")
                    if rc == 0:
                        p = _extract_saved_path(out, "Saved walk-forward run:")
                        if p is not None:
                            st.session_state["last_walkforward_run"] = p.as_posix()
                            st.success(f"Saved run: {_rel(p)}")
                            ok, msg = _auto_generate_interactive_html(run_kind="walkforward", run_path=p)
                            if ok:
                                st.info(f"Auto-generated interactive HTML: `{msg}`")
                            else:
                                st.warning(f"Walk-forward HTML auto-generation failed: {msg}")

            last_wf = st.session_state.get("last_walkforward_run", "")
            if last_wf:
                st.markdown("---")
                st.subheader("Last walk-forward output")
                _render_walkforward_workbook_summary(_abs_path(last_wf))

        with t_mc:
            walk_run_options = [p.as_posix() for p in walk_runs]
            default_walk_run = st.session_state["last_walkforward_run"] or (walk_run_options[0] if walk_run_options else "runs/strategies/.../walkforward/...")
            st.subheader("2) Parent Walk-Forward Run")
            run_dir = _select_path_from_options(
                "Walk-forward run used",
                walk_run_options,
                default_walk_run,
                key_prefix="mc_run_dir",
            )
            mc_parent_run_dir = _abs_path(run_dir) if str(run_dir or "").strip() else None
            mc_parent_spec: dict[str, Any] = {}
            if mc_parent_run_dir is not None:
                mc_spec_path = spec_path(mc_parent_run_dir)
                if mc_spec_path.exists():
                    mc_parent_spec = _read_json(mc_spec_path)
            mc_initial_equity = _infer_initial_equity_from_spec(mc_parent_spec)
            mc_parent_ctx = _estimate_montecarlo_parent_context(mc_parent_run_dir)
            estimated_trades_per_year = _as_float(mc_parent_ctx.get("estimated_trades_per_year"))
            trade_pool_count = int(mc_parent_ctx.get("trade_pool_count", 0) or 0)
            historical_years = _as_float(mc_parent_ctx.get("historical_years"))
            n_trades_default = int(round(estimated_trades_per_year)) if np.isfinite(estimated_trades_per_year) and estimated_trades_per_year > 0 else max(1, trade_pool_count)

            st.subheader("3) Core Monte Carlo Inputs")
            c1, c2, c3 = st.columns(3)
            n_sims = int(
                c1.number_input(
                    "Number of simulations",
                    min_value=1,
                    value=8000,
                    step=100,
                    help="How many Monte Carlo equity paths to generate.",
                )
            )
            ruin_equity = float(
                c2.number_input(
                    "Stop trading if equity drops below $",
                    min_value=0.0,
                    value=70000.0,
                    step=500.0,
                    help="A simulation counts as ruined once equity drops below this level.",
                )
            )
            seed = int(
                c3.number_input(
                    "Seed (reproducibility)",
                    min_value=0,
                    value=42,
                    step=1,
                    key="mc_seed",
                    help="Controls the random sampling. Same seed + same inputs gives the same Monte Carlo run.",
                )
            )
            ruin_pct_of_start = float("nan")
            if np.isfinite(mc_initial_equity) and mc_initial_equity > 0:
                ruin_pct_of_start = float((ruin_equity / mc_initial_equity) * 100.0)

            _render_metric_row(
                [
                    ("Starting equity", f"${mc_initial_equity:,.2f}", "{}"),
                    ("OOS trade pool", trade_pool_count, "{:,.0f}"),
                    ("OOS history years", "n/a" if not np.isfinite(historical_years) else f"{historical_years:.2f}", "{}"),
                    (
                        "Est trades / year",
                        "n/a" if not np.isfinite(estimated_trades_per_year) else f"{estimated_trades_per_year:.0f}",
                        "{}",
                    ),
                    ("Ruin threshold", f"${ruin_equity:,.2f}", "{}"),
                ]
            )
            st.caption(
                "Starting equity is inherited from the selected walk-forward run. "
                "Monte Carlo does not choose a separate starting balance."
            )
            st.caption(
                "The trade-pool estimate comes from the parent walk-forward OOS trades. "
                "A common default is to simulate about one year of trades."
            )

            core_cols = st.columns(2)
            n_trades_value = int(
                core_cols[0].number_input(
                    "Trades per simulation (~1 year)",
                    min_value=1,
                    value=max(1, n_trades_default),
                    step=1,
                    key="mc_n_trades_value",
                    help="Defaults to the estimated number of trades per year from the selected WFA run. Increase it to simulate a longer horizon.",
                )
            )
            core_cols[1].markdown(
                f"**Ruin threshold % of start**\n\n"
                + ("`n/a`" if not np.isfinite(ruin_pct_of_start) else f"`{ruin_pct_of_start:.2f}%`")
            )

            st.subheader("4) Workbook Outputs + Pass Criteria")
            threshold_cols = st.columns(4)
            tor_max = float(
                threshold_cols[0].number_input(
                    "Max risk of ruin %",
                    value=10.0,
                    step=0.5,
                    help="Acceptance threshold for the % of simulations that breach the ruin threshold.",
                )
            )
            mdd_max = float(
                threshold_cols[1].number_input(
                    "Max median DD %",
                    value=40.0,
                    step=0.5,
                    help="Acceptance threshold for the median simulated max drawdown.",
                )
            )
            ret_min = float(
                threshold_cols[2].number_input(
                    "Min median return %",
                    value=40.0,
                    step=0.5,
                    help="Acceptance threshold for the median simulated return.",
                )
            )
            ratio_min = float(
                threshold_cols[3].number_input(
                    "Min return / DD ratio",
                    value=2.0,
                    step=0.1,
                    help="Acceptance threshold for return-to-drawdown ratio.",
                )
            )

            st.subheader("5) Advanced / technical overrides")
            with st.expander("Advanced / technical overrides", expanded=False):
                c4, c5, c6 = st.columns(3)
                sample_with_replacement = c4.checkbox(
                    "Sample with replacement",
                    value=True,
                    help="When enabled, the same historical trade can appear multiple times in one simulation. Disable to sample each trade at most once per path.",
                )
                stop_at_ruin = c5.checkbox(
                    "Stop trading after ruin hit",
                    value=True,
                    help="If enabled, the equity path flatlines after the ruin threshold is breached instead of continuing to sample later trades.",
                )
                progress_every = int(
                    c6.number_input(
                        "Progress every",
                        min_value=1,
                        value=200,
                        step=10,
                        key="mc_progress_every",
                        help="How often progress logs are printed during the run.",
                    )
                )

                c7, c8, c9 = st.columns(3)
                pnl_mode = c7.selectbox(
                    "Trade outcome sizing mode",
                    ("actual", "fixed_risk"),
                    format_func=lambda mode: (
                        "Use real OOS trade $PnL"
                        if mode == "actual"
                        else "Normalize each trade to fixed $ risk"
                    ),
                    help=(
                        "`actual` uses the real dollar PnL from each OOS trade. "
                        "`fixed_risk` rescales each trade as r_multiple x fixed-risk dollars."
                    ),
                )
                fixed_risk_dollars = c8.text_input(
                    "Fixed-risk $ per trade",
                    value="",
                    help="Required only when trade outcome sizing mode is `fixed_risk`.",
                )
                extra_args = c9.text_input("Extra args (optional)", value="", key="mc_extra_args")

                c10, c11 = st.columns(2)
                sample_paths = int(
                    c10.number_input(
                        "Saved individual paths",
                        min_value=0,
                        value=120,
                        step=10,
                        help="How many individual simulation paths to save for later plotting/debugging.",
                    )
                )
                save_quantiles = c11.checkbox(
                    "Save percentile envelope paths (p05/p25/p50/p75/p95)",
                    value=True,
                    help="Saves percentile equity envelopes across all simulations for charting the Monte Carlo distribution.",
                )

                if pnl_mode == "actual":
                    c7.caption("Each sampled trade keeps its original dollar PnL from the WFA OOS trade pool.")
                else:
                    c7.caption("Each sampled trade becomes `r_multiple x fixed-risk $ per trade`, so sizing is normalized.")
                c10.caption("These are a small subset of full simulation paths, useful for inspection.")
                c11.caption("These are summary curves across all simulations, useful for fan/envelope charts.")

                st.caption(
                    "Difference: `Ruin threshold equity` defines when an individual simulation is counted as ruined. "
                    "`Max risk of ruin %` is the acceptance rule applied after all simulations finish."
                )

            if st.button("Run Monte Carlo", type="primary"):
                try:
                    if pnl_mode == "fixed_risk" and not fixed_risk_dollars.strip():
                        raise ValueError("fixed risk dollars is required when pnl mode is fixed_risk")
                    cmd = [
                        SCRIPT_PYTHON,
                        str(SCRIPTS_DIR / "run_monte_carlo.py"),
                        "--run-dir",
                        run_dir,
                        "--n-sims",
                        str(n_sims),
                        "--n-trades",
                        str(int(n_trades_value)),
                        "--ruin-equity",
                        str(ruin_equity),
                        "--seed",
                        str(seed),
                        "--pnl-mode",
                        pnl_mode,
                        "--progress-every",
                        str(progress_every),
                        "--save-sample-paths-count",
                        str(sample_paths),
                        "--target-risk-of-ruin-pct-max",
                        str(tor_max),
                        "--target-median-max-dd-pct-max",
                        str(mdd_max),
                        "--target-median-return-pct-min",
                        str(ret_min),
                        "--target-return-dd-ratio-min",
                        str(ratio_min),
                    ]
                    if sample_with_replacement:
                        cmd.append("--replace")
                    else:
                        cmd.append("--without-replacement")
                    if stop_at_ruin:
                        cmd.append("--stop-at-ruin")
                    if pnl_mode == "fixed_risk" and fixed_risk_dollars.strip():
                        cmd.extend(["--fixed-risk-dollars", fixed_risk_dollars.strip()])
                    if not save_quantiles:
                        cmd.append("--no-save-quantile-paths")
                    cmd.extend(_parse_extra_args(extra_args))
                except ValueError as e:
                    st.error(f"Invalid arguments: {e}")
                else:
                    rc, out = _run_cli_live(cmd, workflow="monte_carlo")
                    if rc == 0:
                        p = _extract_saved_path(out, "Saved Monte Carlo run:")
                        if p is not None:
                            st.session_state["last_mc_run"] = p.as_posix()
                            st.success(f"Saved run: {_rel(p)}")
                            ok, msg = _auto_generate_interactive_html(run_kind="monte_carlo", run_path=p)
                            if ok:
                                st.info(f"Auto-generated interactive HTML: `{msg}`")
                            else:
                                st.warning(f"Monte Carlo HTML auto-generation failed: {msg}")

            last_mc = st.session_state.get("last_mc_run", "")
            if last_mc:
                st.markdown("---")
                st.subheader("Last Monte Carlo output")
                _render_montecarlo_workbook_summary(_abs_path(last_mc))

        with t_limited:
            strategy_names = _sorted_strategy_keys(strategy_catalog)
            default_strategy_idx = strategy_names.index("sma_cross_strategy") if "sma_cross_strategy" in strategy_names else 0
            st.subheader("2) Strategy + Scenario")
            setup_left, setup_right = st.columns([0.92, 1.08], gap="large")
            with setup_left:
                strategy_short = st.selectbox(
                    "Strategy",
                    strategy_names,
                    index=default_strategy_idx,
                    key="limited_strategy",
                    format_func=lambda key: _strategy_display_name(key, strategy_catalog),
                )
            strategy_module = f"quantbt.strategies.{strategy_short}"
            strategy_info = strategy_catalog.get(strategy_short, {})
            strategy_cfg = strategy_info.get("strategy_config", {})
            data_path = wb_dataset

            entry_plugins = sorted(plugin_catalog.get("entry", {}).keys())
            exit_plugins = sorted(plugin_catalog.get("exit", {}).keys())
            sizing_plugins = sorted(plugin_catalog.get("sizing", {}).keys())

            default_entry = "(use strategy default)"
            default_exit = "(use strategy default)"
            default_sizing = "(use strategy default)"

            entry_default_name = (
                strategy_cfg.get("entry", {}).get("rules", [{}])[0].get("name")
                if isinstance(strategy_cfg.get("entry", {}).get("rules"), list)
                else None
            )
            exit_default_name = strategy_cfg.get("exit", {}).get("name")
            sizing_default_name = strategy_cfg.get("sizing", {}).get("name")

            def _strategy_default_entry_params() -> dict[str, Any]:
                if (
                    isinstance(strategy_cfg.get("entry", {}).get("rules"), list)
                    and strategy_cfg.get("entry", {}).get("rules")
                    and isinstance(strategy_cfg.get("entry", {}).get("rules")[0], dict)
                ):
                    return dict(strategy_cfg.get("entry", {}).get("rules", [{}])[0].get("params", {}) or {})
                return {}

            def _strategy_default_exit_params() -> dict[str, Any]:
                return dict(strategy_cfg.get("exit", {}).get("params", {}) or {})

            def _strategy_default_sizing_params() -> dict[str, Any]:
                return dict(strategy_cfg.get("sizing", {}).get("params", {}) or {})

            monkey_criteria = {
                "mode": "all",
                "rules": [
                    {"metric": "total_return_%", "op": "<", "value": 16.3},
                    {"metric": "max_drawdown_abs_%", "op": ">", "value": 11.4},
                ],
            }
            similar_entry_criteria = {"total_return_%": {">": 0}}
            strategy_param_space = copy.deepcopy(strategy_info.get("param_space", {}) or {})
            strategy_entry_contract = _build_strategy_limited_section_contract(
                strategy_cfg,
                section="entry",
                default_params=_strategy_default_entry_params(),
            )
            strategy_exit_contract = _build_strategy_limited_section_contract(
                strategy_cfg,
                section="exit",
                default_params=_strategy_default_exit_params(),
            )

            def _strategy_seed_entry_params() -> dict[str, Any]:
                return _seed_strategy_limited_params(_strategy_default_entry_params(), strategy_entry_contract)

            def _strategy_seed_exit_params() -> dict[str, Any]:
                return _seed_strategy_limited_params(_strategy_default_exit_params(), strategy_exit_contract)

            def _similar_entry_template_params() -> dict[str, Any]:
                if strategy_entry_contract.get("has_policy", False):
                    return _strategy_seed_entry_params()

                params = _strategy_default_entry_params()
                preferred_order = [
                    "atr_dist_for_liq_generation",
                    "liq_move_away_atr",
                    "lookback",
                    "fast",
                    "slow",
                    "length",
                    "period",
                    "window",
                    "max_rr",
                ]
                candidates = [
                    key
                    for key, value in strategy_param_space.items()
                    if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value)
                ]
                if not candidates:
                    return params
                ranked = sorted(
                    candidates,
                    key=lambda key: (
                        preferred_order.index(key) if key in preferred_order else len(preferred_order),
                        str(key),
                    ),
                )
                for key in ranked[:2]:
                    params[key] = copy.deepcopy(strategy_param_space[key])
                return params

            preset_defs: dict[str, dict[str, Any]] = {
                "Core": {
                    "entry_plugin": default_entry,
                    "entry_params": _strategy_seed_entry_params(),
                    "exit_plugin": default_exit,
                    "exit_params": _strategy_seed_exit_params(),
                    "sizing_plugin": default_sizing,
                    "sizing_params": _strategy_default_sizing_params(),
                    "seed_count": "",
                    "seed_start": "",
                    "exit_seed_count": "",
                    "exit_seed_start": "",
                    "favourable_criteria": "",
                    "pass_threshold": "",
                },
                "Fixed ATR Exit": {
                    "entry_plugin": default_entry,
                    "entry_params": _strategy_seed_entry_params(),
                    "exit_plugin": "atr_brackets",
                    "exit_params": copy.deepcopy(DEFAULT_EXIT_PARAMS.get("atr_brackets", {"rr": [1.5, 2.0, 2.5], "sldist_atr_mult": [1.0, 1.5], "atr_period": 14})),
                    "seed_count": "",
                    "seed_start": "",
                    "exit_seed_count": "",
                    "exit_seed_start": "",
                    "favourable_criteria": "",
                    "pass_threshold": "",
                },
                "Fixed Bar Exit": {
                    "entry_plugin": default_entry,
                    "entry_params": _strategy_seed_entry_params(),
                    "exit_plugin": "time_exit",
                    "exit_params": copy.deepcopy(DEFAULT_EXIT_PARAMS.get("time_exit", {"hold_bars": [1]})),
                    "seed_count": "",
                    "seed_start": "",
                    "exit_seed_count": "",
                    "exit_seed_start": "",
                    "favourable_criteria": "",
                    "pass_threshold": "",
                },
                "Similar Entry": {
                    "entry_plugin": default_entry,
                    "entry_params": _similar_entry_template_params(),
                    "exit_plugin": default_exit,
                    "exit_params": _strategy_seed_exit_params(),
                    "seed_count": "",
                    "seed_start": "",
                    "exit_seed_count": "",
                    "exit_seed_start": "",
                    "favourable_criteria": json.dumps(similar_entry_criteria, separators=(",", ":")),
                    "pass_threshold": "",
                },
                "Monkey Entry": {
                    "entry_plugin": "monkey_entry",
                    "entry_params": {"target_entries": 132, "side": "both", "long_ratio": 0.5},
                    "exit_plugin": default_exit,
                    "exit_params": _strategy_seed_exit_params(),
                    "seed_count": "8000",
                    "seed_start": "1",
                    "exit_seed_count": "",
                    "exit_seed_start": "",
                    "favourable_criteria": json.dumps(monkey_criteria, separators=(",", ":")),
                    "pass_threshold": "90",
                    "extra_args": "--monkey-davey-style",
                },
                "Monkey Exit": {
                    "entry_plugin": default_entry,
                    "entry_params": _strategy_seed_entry_params(),
                    "exit_plugin": "monkey_exit",
                    "exit_params": {"avg_hold_bars": 15.75},
                    "seed_count": "",
                    "seed_start": "",
                    "exit_seed_count": "8000",
                    "exit_seed_start": "1",
                    "favourable_criteria": json.dumps(monkey_criteria, separators=(",", ":")),
                    "pass_threshold": "90",
                    "extra_args": "--monkey-davey-style --monkey-fast-summary",
                },
                "Monkey Entry + Exit": {
                    "entry_plugin": "monkey_entry",
                    "entry_params": {"target_entries": 132, "side": "both", "long_ratio": 0.5},
                    "exit_plugin": "monkey_exit",
                    "exit_params": {"avg_hold_bars": 15.75},
                    "seed_count": "8000",
                    "seed_start": "1",
                    "exit_seed_count": "",
                    "exit_seed_start": "",
                    "favourable_criteria": json.dumps(monkey_criteria, separators=(",", ":")),
                    "pass_threshold": "90",
                    "extra_args": "--monkey-davey-style --monkey-fast-summary",
                },
            }

            workbook_scenario_map: dict[str, dict[str, Any]] = {
                str(x.get("name", "")).strip(): x
                for x in LIMITED_WORKBOOK_SCENARIOS
                if isinstance(x, dict) and str(x.get("name", "")).strip()
            }
            default_workbook_template = (
                "CORE TEST"
                if "CORE TEST" in workbook_scenario_map
                else (next(iter(workbook_scenario_map.keys())) if workbook_scenario_map else "Custom")
            )
            st.session_state.setdefault("limited_workbook_template", default_workbook_template)
            with setup_right:
                selected_workbook_template = st.selectbox(
                    "Workbook scenario template",
                    [*list(workbook_scenario_map.keys()), "Custom"],
                    key="limited_workbook_template",
                )

            selected_workbook: dict[str, Any] | None = workbook_scenario_map.get(selected_workbook_template)
            if isinstance(selected_workbook, dict):
                preset_label = str(selected_workbook.get("preset", "Core")).strip() or "Core"
                st.session_state["limited_test_template"] = preset_label
                st.session_state["limited_test_type"] = str(selected_workbook.get("test_type", "Custom") or "Custom")
                if preset_label.startswith("Monkey"):
                    st.info(
                        "Monkey tests default to strict Davey mode in this UI. "
                        "Use the baseline-matched setup section below before running."
                    )
            else:
                st.session_state["limited_test_type"] = "Custom"
                preset_label = "Custom"

            with setup_left:
                st.markdown(_reference_link("View strategy config", "strategy", strategy_short))
                st.caption(f"Dataset comes from the page-level selector: `{wb_dataset}`")
            with setup_right:
                if isinstance(selected_workbook, dict):
                    st.caption(f"Preset route: `{preset_label}`. Scenario selection auto-loads the defaults shown below.")
                else:
                    st.caption("Custom mode leaves the fields below fully editable.")

            def _safe_plugin_choice(value: str, options: list[str], fallback: str) -> str:
                return value if value in options else fallback

            def _as_json_or_blank(value: Any) -> str:
                if value is None:
                    return ""
                if isinstance(value, str):
                    return value
                return _json_pretty(value)

            def _apply_limited_template(label: str) -> None:
                tpl = preset_defs[label]
                entry_choice = _safe_plugin_choice(str(tpl.get("entry_plugin", default_entry)), [default_entry, *entry_plugins], default_entry)
                exit_choice = _safe_plugin_choice(str(tpl.get("exit_plugin", default_exit)), [default_exit, *exit_plugins], default_exit)
                sizing_choice = _safe_plugin_choice(
                    str(tpl.get("sizing_plugin", default_sizing)),
                    [default_sizing, *sizing_plugins],
                    default_sizing,
                )

                st.session_state["limited_entry_plugin_select"] = entry_choice
                st.session_state["limited_exit_plugin_select"] = exit_choice
                st.session_state["limited_sizing_plugin_select"] = sizing_choice
                st.session_state["limited_entry_params_json"] = _as_json_or_blank(tpl.get("entry_params"))
                st.session_state["limited_exit_params_json"] = _as_json_or_blank(tpl.get("exit_params"))
                st.session_state["limited_sizing_params_json"] = _as_json_or_blank(
                    tpl.get("sizing_params", _strategy_default_sizing_params())
                )
                st.session_state["limited_seed_count"] = str(tpl.get("seed_count", ""))
                st.session_state["limited_seed_start"] = str(tpl.get("seed_start", ""))
                st.session_state["limited_exit_seed_count"] = str(tpl.get("exit_seed_count", ""))
                st.session_state["limited_exit_seed_start"] = str(tpl.get("exit_seed_start", ""))
                st.session_state["limited_favourable_criteria"] = str(tpl.get("favourable_criteria", ""))
                st.session_state["limited_pass_threshold"] = str(tpl.get("pass_threshold", ""))
                st.session_state["limited_extra_args"] = str(tpl.get("extra_args", ""))
                st.session_state.setdefault("limited_commission_rt", "5")
                st.session_state.setdefault("limited_spread_pips", "0.2")
                if not str(st.session_state.get("limited_commission_rt", "")).strip():
                    st.session_state["limited_commission_rt"] = "5"
                if not str(st.session_state.get("limited_spread_pips", "")).strip():
                    st.session_state["limited_spread_pips"] = "0.2"
                st.session_state.setdefault("limited_pip_size", "0.0001")

            def _apply_workbook_review_defaults(workbook_def: dict[str, Any]) -> None:
                st.session_state["limited_workbook_output_rows"] = _build_workbook_output_rows(list(workbook_def.get("outputs", [])))
                st.session_state["limited_workbook_rule_rows"] = _build_workbook_rule_rows(list(workbook_def.get("pass_criteria", [])))
                st.session_state.pop("limited_workbook_output_rows_editor", None)
                st.session_state.pop("limited_workbook_rule_rows_editor", None)

            workbook_track_key = "_limited_last_workbook_signature"
            prefill_lock = bool(st.session_state.pop("_limited_prefill_lock", False))
            selected_workbook_signature = f"{strategy_short}::{selected_workbook_template}::{preset_label}"

            st.session_state.setdefault("limited_entry_plugin_select", default_entry)
            st.session_state.setdefault("limited_exit_plugin_select", default_exit)
            st.session_state.setdefault("limited_sizing_plugin_select", default_sizing)
            st.session_state.setdefault("limited_entry_params_json", _json_pretty(_strategy_seed_entry_params()))
            st.session_state.setdefault("limited_exit_params_json", _json_pretty(_strategy_seed_exit_params()))
            st.session_state.setdefault("limited_sizing_params_json", _json_pretty(_strategy_default_sizing_params()))
            st.session_state.setdefault("limited_run_base", "")
            st.session_state.setdefault("limited_test_name", "")
            st.session_state.setdefault("limited_progress_every", 10)
            st.session_state.setdefault("limited_commission_rt", "5")
            st.session_state.setdefault("limited_spread_pips", "0.2")
            if not str(st.session_state.get("limited_commission_rt", "")).strip():
                st.session_state["limited_commission_rt"] = "5"
            if not str(st.session_state.get("limited_spread_pips", "")).strip():
                st.session_state["limited_spread_pips"] = "0.2"
            st.session_state.setdefault("limited_pip_size", "0.0001")
            st.session_state.setdefault("limited_pass_threshold", "")
            st.session_state.setdefault("limited_min_trades", "")
            st.session_state.setdefault("limited_favourable_criteria", "")
            st.session_state.setdefault("limited_extra_args", "")
            st.session_state.setdefault("limited_seed_count", "")
            st.session_state.setdefault("limited_seed_start", "")
            st.session_state.setdefault("limited_exit_seed_count", "")
            st.session_state.setdefault("limited_exit_seed_start", "")
            st.session_state.setdefault("limited_workbook_output_rows", [])
            st.session_state.setdefault("limited_workbook_rule_rows", [])
            st.session_state.setdefault("limited_monkey_unlock_overrides", False)
            st.session_state.setdefault("limited_monkey_derived_fields", {})
            st.session_state.setdefault("_limited_pending_form_updates", {})

            pending_limited_updates = st.session_state.pop("_limited_pending_form_updates", None)
            if isinstance(pending_limited_updates, dict) and pending_limited_updates:
                for _k, _v in pending_limited_updates.items():
                    st.session_state[_k] = _v

            if isinstance(selected_workbook, dict) and not prefill_lock:
                if st.session_state.get(workbook_track_key) != selected_workbook_signature:
                    _apply_limited_template(preset_label)
                    _apply_workbook_review_defaults(selected_workbook)
                    st.session_state["limited_monkey_unlock_overrides"] = False
            st.session_state[workbook_track_key] = selected_workbook_signature

            st.markdown(
                '<div class="qt-subtle-card"><div class="qt-kicker">Scenario Brief</div>'
                '<div class="qt-card-title">Workbook-aligned test definition</div>'
                '<div class="qt-card-note">The selected scenario decides which limited-test fields are shown and which defaults are loaded.</div></div>',
                unsafe_allow_html=True,
            )
            _render_chip_group("Global Inputs", LIMITED_WORKBOOK_GLOBAL_INPUTS, tone="input")

            if isinstance(selected_workbook, dict):
                w1, w2, w3 = st.columns(3)
                with w1:
                    _render_chip_group("Inputs", list(selected_workbook.get("inputs", [])), tone="input")
                with w2:
                    _render_chip_group("Outputs", list(selected_workbook.get("outputs", [])), tone="output")
                with w3:
                    _render_chip_group("Pass Criteria", list(selected_workbook.get("pass_criteria", [])), tone="pass")

                selected_workbook_preset = str(selected_workbook.get("preset", "")).strip()
                if selected_workbook_preset.startswith("Monkey"):
                    st.caption(
                        "Monkey scenarios should use the baseline-matched section below to bind targets "
                        "(trade count/long ratio/avg hold) to a selected core baseline run."
                    )
            else:
                st.info(
                    "`Custom` selected. Fields below remain manual except for strategy-defined non-optimisable "
                    "baseline entry/exit params, which stay locked when you use the strategy defaults."
                )

            monkey_modes = {"Monkey Entry", "Monkey Exit", "Monkey Entry + Exit"}
            selected_monkey_mode = preset_label if preset_label in monkey_modes else None

            def _reapply_locked_monkey_fields(mode_label: str | None) -> None:
                if not mode_label:
                    return
                if bool(st.session_state.get("limited_monkey_unlock_overrides", False)):
                    return
                payload = st.session_state.get("limited_monkey_derived_fields")
                if not isinstance(payload, dict):
                    return
                if str(payload.get("mode", "")) != str(mode_label):
                    return
                protected_keys = [
                    "limited_entry_plugin_select",
                    "limited_exit_plugin_select",
                    "limited_entry_params_json",
                    "limited_exit_params_json",
                    "limited_seed_count",
                    "limited_seed_start",
                    "limited_exit_seed_count",
                    "limited_exit_seed_start",
                    "limited_favourable_criteria",
                    "limited_pass_threshold",
                    "limited_min_trades",
                    "limited_test_name",
                    "limited_extra_args",
                ]
                for k in protected_keys:
                    if k in payload:
                        st.session_state[k] = payload[k]

            _reapply_locked_monkey_fields(selected_monkey_mode)

            monkey_base_run_raw = ""
            monkey_baseline_loaded_ok: bool | None = None

            scenario_entry_label = (
                str(selected_workbook.get("entry_params_label", "Entry params JSON"))
                if isinstance(selected_workbook, dict)
                else "Entry params JSON"
            )
            scenario_exit_label = (
                str(selected_workbook.get("exit_params_label", "Exit params JSON"))
                if isinstance(selected_workbook, dict)
                else "Exit params JSON"
            )
            current_entry_plugin = str(st.session_state.get("limited_entry_plugin_select", default_entry) or default_entry)
            current_exit_plugin = str(st.session_state.get("limited_exit_plugin_select", default_exit) or default_exit)
            current_sizing_plugin = str(st.session_state.get("limited_sizing_plugin_select", default_sizing) or default_sizing)
            current_entry_name = entry_default_name if current_entry_plugin == default_entry else current_entry_plugin
            current_exit_name = exit_default_name if current_exit_plugin == default_exit else current_exit_plugin
            current_sizing_name = sizing_default_name if current_sizing_plugin == default_sizing else current_sizing_plugin

            st.subheader("3) Scenario Inputs")
            st.caption("Scenario defaults are loaded automatically. Only the controls needed for this limited test stay visible here.")
            input_assumption_cols = st.columns(2)
            spread_pips = input_assumption_cols[0].text_input("Slippage / spread (pips, required)", key="limited_spread_pips")
            commission_rt = input_assumption_cols[1].text_input("Commission RT (required)", key="limited_commission_rt")
            st.caption("Slippage / spread and Commission RT are required for every limited test run.")

            st.caption(
                f"Active setup: entry=`{current_entry_name or 'none'}` | "
                f"exit=`{current_exit_name or 'none'}` | "
                f"sizing=`{current_sizing_name or 'none'}` | "
                f"preset=`{preset_label}`"
            )

            link_cols = st.columns(3)
            if current_entry_name:
                link_cols[0].markdown(_reference_link("View entry plugin", "entry", str(current_entry_name)))
            if current_exit_name:
                link_cols[1].markdown(_reference_link("View exit plugin", "exit", str(current_exit_name)))
            if current_sizing_name:
                link_cols[2].markdown(_reference_link("View sizing plugin", "sizing", str(current_sizing_name)))

            entry_params_dict = _json_object_or_empty(st.session_state.get("limited_entry_params_json", ""))
            exit_params_dict = _json_object_or_empty(st.session_state.get("limited_exit_params_json", ""))
            structured_input_checks: list[tuple[bool, str, str]] = []
            show_entry_json_in_main = True
            show_exit_json_in_main = True
            monkey_entry_plugin_names = {"monkey_entry", "random"}
            monkey_exit_plugin_names = {"monkey_exit", "random_time_exit"}
            rr_exit_plugin_names = {"atr_brackets", "interequity_liqsweep_exit", "interequity_liqsweepb_exit"}
            similar_entry_structured_keys: list[str] = []
            entry_uses_strategy_policy = bool(
                current_entry_name
                and current_entry_name == entry_default_name
                and strategy_entry_contract.get("has_policy", False)
            )
            exit_uses_strategy_policy = bool(
                current_exit_name
                and current_exit_name == exit_default_name
                and strategy_exit_contract.get("has_policy", False)
            )
            if preset_label == "Similar Entry":
                entry_priority = [
                    "atr_dist_for_liq_generation",
                    "liq_move_away_atr",
                    "lookback",
                    "fast",
                    "slow",
                    "length",
                    "period",
                    "window",
                    "max_rr",
                ]
                similar_entry_structured_keys = [
                    key
                    for key, value in entry_params_dict.items()
                    if _derive_numeric_range_spec(value) is not None
                ]
                similar_entry_structured_keys = sorted(
                    similar_entry_structured_keys,
                    key=lambda key: (
                        entry_priority.index(key) if key in entry_priority else len(entry_priority),
                        str(key),
                    ),
                )[:2]

            def _render_strategy_policy_inputs(
                *,
                param_kind: str,
                params_dict: dict[str, Any],
                default_params: dict[str, Any],
                contract: dict[str, Any],
                state_key: str,
                widget_prefix: str,
            ) -> str:
                working_params = _sanitize_strategy_limited_params(params_dict, default_params, contract)
                editable_specs = dict(contract.get("editable_specs", {}))
                fixed_params = dict(contract.get("fixed_params", {}))

                if editable_specs:
                    st.caption(
                        f"Only the strategy-defined optimisable {param_kind} parameters are editable here. "
                        "All other baseline fields remain locked."
                    )
                    for key, spec in editable_specs.items():
                        label = str(spec.get("label", key.replace("_", " ")))
                        integer_mode = bool(spec.get("integer", False))
                        value_seed = copy.deepcopy(working_params.get(key, spec.get("values", [])))
                        range_spec = _editable_range_defaults(spec, value_seed)
                        widget_key_base = f"{widget_prefix}_{key}"

                        if range_spec is not None:
                            start_default, end_default, step_default, _ = range_spec
                            start_seed = int(round(start_default)) if integer_mode else float(start_default)
                            end_seed = int(round(end_default)) if integer_mode else float(end_default)
                            step_seed = max(1, int(round(step_default))) if integer_mode else float(step_default)
                            _sync_widget_from_source(f"{widget_key_base}_start", start_seed)
                            _sync_widget_from_source(f"{widget_key_base}_end", end_seed)
                            _sync_widget_from_source(f"{widget_key_base}_step", step_seed)
                            st.markdown(f"`{key}`")
                            range_cols = st.columns(3)
                            if integer_mode:
                                start_val = int(range_cols[0].number_input(f"{label} start", step=1, key=f"{widget_key_base}_start"))
                                stop_val = int(range_cols[1].number_input(f"{label} end", step=1, key=f"{widget_key_base}_end"))
                                step_val = int(
                                    range_cols[2].number_input(
                                        f"{label} step",
                                        min_value=1,
                                        step=1,
                                        key=f"{widget_key_base}_step",
                                    )
                                )
                            else:
                                number_step = float(step_seed) if math.isfinite(float(step_seed)) and float(step_seed) > 0 else 0.1
                                start_val = float(
                                    range_cols[0].number_input(
                                        f"{label} start",
                                        step=number_step,
                                        format="%.4f",
                                        key=f"{widget_key_base}_start",
                                    )
                                )
                                stop_val = float(
                                    range_cols[1].number_input(
                                        f"{label} end",
                                        step=number_step,
                                        format="%.4f",
                                        key=f"{widget_key_base}_end",
                                    )
                                )
                                step_val = float(
                                    range_cols[2].number_input(
                                        f"{label} step",
                                        min_value=float(number_step / 10.0),
                                        step=number_step,
                                        format="%.4f",
                                        key=f"{widget_key_base}_step",
                                    )
                                )
                            try:
                                range_values = _build_numeric_range_values(
                                    start_val,
                                    stop_val,
                                    step_val,
                                    integer=integer_mode,
                                )
                                working_params[key] = range_values
                                structured_input_checks.append((True, label, f"{len(range_values)} values"))
                            except Exception as e:
                                structured_input_checks.append((False, label, str(e)))
                        else:
                            raw_default = _value_to_csv_text(value_seed)
                            _sync_widget_from_source(f"{widget_key_base}_values", raw_default)
                            raw_text = st.text_input(label, key=f"{widget_key_base}_values")
                            value_kind = "int" if integer_mode else "float"
                            parse_ok, parsed_value, parse_detail = _parse_csv_numeric_text(raw_text, kind=value_kind)
                            structured_input_checks.append((parse_ok, label, parse_detail))
                            if parse_ok and parsed_value is not None:
                                working_params[key] = parsed_value

                else:
                    st.caption(
                        f"No optimisable {param_kind} parameters are defined for this strategy. "
                        "The baseline values below are locked for limited tests."
                    )

                if fixed_params:
                    st.caption(f"Locked {param_kind} params from strategy:")
                    st.code(_json_pretty(fixed_params), language="json")

                working_params = _sanitize_strategy_limited_params(working_params, default_params, contract)
                st.session_state[state_key] = _json_pretty(working_params)
                return st.session_state[state_key]

            scenario_param_cols = st.columns(2)
            with scenario_param_cols[0]:
                st.markdown(f"**{scenario_entry_label}**")
                if str(current_entry_name or "") in monkey_entry_plugin_names:
                    show_entry_json_in_main = False
                    entry_target_default = int(entry_params_dict.get("target_entries", 0) or 0)
                    entry_long_ratio_default = float(entry_params_dict.get("long_ratio", entry_params_dict.get("long_pct", 0.5)) or 0.5)
                    _sync_widget_from_source("limited_structured_target_entries", entry_target_default)
                    _sync_widget_from_source(
                        "limited_structured_long_ratio_pct",
                        int(round(max(0.0, min(1.0, entry_long_ratio_default)) * 100.0)),
                    )
                    target_entries = int(
                        st.number_input(
                            "Target entries",
                            min_value=0,
                            step=1,
                            key="limited_structured_target_entries",
                        )
                    )
                    long_ratio_pct = int(
                        st.slider(
                            "Long ratio (%)",
                            min_value=0,
                            max_value=100,
                            step=1,
                            key="limited_structured_long_ratio_pct",
                        )
                    )
                    entry_side = str(entry_params_dict.get("side", "both") or "both")
                    entry_params_dict["target_entries"] = int(target_entries)
                    entry_params_dict["long_ratio"] = round(float(long_ratio_pct) / 100.0, 4)
                    entry_params_dict.pop("prob", None)
                    st.session_state["limited_entry_params_json"] = _json_pretty(entry_params_dict)
                    entry_params = st.session_state["limited_entry_params_json"]
                    st.caption(
                        f"Side mode: `{entry_side}`. Use Advanced to change side, spacing, seed, or other monkey-entry parameters."
                    )
                elif entry_uses_strategy_policy:
                    show_entry_json_in_main = False
                    entry_params = _render_strategy_policy_inputs(
                        param_kind="entry",
                        params_dict=entry_params_dict,
                        default_params=_strategy_default_entry_params(),
                        contract=strategy_entry_contract,
                        state_key="limited_entry_params_json",
                        widget_prefix="limited_strategy_entry",
                    )
                elif similar_entry_structured_keys:
                    show_entry_json_in_main = False
                    for key in similar_entry_structured_keys:
                        spec = _derive_numeric_range_spec(entry_params_dict.get(key))
                        if spec is None:
                            continue
                        start_default, stop_default, step_default, integer_mode = spec
                        widget_prefix = f"limited_structured_{key}"
                        start_seed = int(round(start_default)) if integer_mode else float(start_default)
                        stop_seed = int(round(stop_default)) if integer_mode else float(stop_default)
                        step_seed = max(1, int(round(step_default))) if integer_mode else float(step_default)
                        _sync_widget_from_source(f"{widget_prefix}_start", start_seed)
                        _sync_widget_from_source(f"{widget_prefix}_end", stop_seed)
                        _sync_widget_from_source(f"{widget_prefix}_step", step_seed)
                        nice_label = key.replace("_", " ")
                        st.markdown(f"`{key}`")
                        range_cols = st.columns(3)
                        if integer_mode:
                            start_val = int(range_cols[0].number_input("Start", step=1, key=f"{widget_prefix}_start"))
                            stop_val = int(range_cols[1].number_input("End", step=1, key=f"{widget_prefix}_end"))
                            step_val = int(range_cols[2].number_input("Step", min_value=1, step=1, key=f"{widget_prefix}_step"))
                        else:
                            number_step = float(step_seed) if math.isfinite(float(step_seed)) and float(step_seed) > 0 else 0.1
                            start_val = float(range_cols[0].number_input("Start", step=number_step, format="%.4f", key=f"{widget_prefix}_start"))
                            stop_val = float(range_cols[1].number_input("End", step=number_step, format="%.4f", key=f"{widget_prefix}_end"))
                            step_val = float(
                                range_cols[2].number_input(
                                    "Step",
                                    min_value=float(number_step / 10.0),
                                    step=number_step,
                                    format="%.4f",
                                    key=f"{widget_prefix}_step",
                                )
                            )
                        try:
                            range_values = _build_numeric_range_values(start_val, stop_val, step_val, integer=integer_mode)
                            entry_params_dict[key] = range_values
                            structured_input_checks.append((True, f"{nice_label} range", f"{len(range_values)} values"))
                        except Exception as e:
                            structured_input_checks.append((False, f"{nice_label} range", str(e)))
                    st.session_state["limited_entry_params_json"] = _json_pretty(entry_params_dict)
                    entry_params = st.session_state["limited_entry_params_json"]
                    st.caption("These are the optimisable entry ranges for the similar-entry test. Fixed baseline entry fields are preserved.")
                else:
                    entry_params = st.text_area(
                        scenario_entry_label,
                        height=150,
                        key="limited_entry_params_json",
                    )

            with scenario_param_cols[1]:
                st.markdown(f"**{scenario_exit_label}**")
                if exit_uses_strategy_policy:
                    show_exit_json_in_main = False
                    exit_params = _render_strategy_policy_inputs(
                        param_kind="exit",
                        params_dict=exit_params_dict,
                        default_params=_strategy_default_exit_params(),
                        contract=strategy_exit_contract,
                        state_key="limited_exit_params_json",
                        widget_prefix="limited_strategy_exit",
                    )
                elif str(current_exit_name or "") in rr_exit_plugin_names and (
                    str(current_exit_name or "") == "atr_brackets" or "rr" in exit_params_dict
                ):
                    show_exit_json_in_main = False
                    is_atr_brackets = str(current_exit_name or "") == "atr_brackets"
                    if str(current_exit_name or "") == "atr_brackets" or "rr" in exit_params_dict:
                        rr_key = "rr"
                    elif "fallback_rr" in exit_params_dict:
                        rr_key = "fallback_rr"
                    else:
                        rr_key = "min_rr"
                    rr_default = _value_to_csv_text(
                        exit_params_dict.get(
                            rr_key,
                            exit_params_dict.get("fallback_rr", exit_params_dict.get("rr", exit_params_dict.get("min_rr", 1.5))),
                        )
                    )
                    _sync_widget_from_source("limited_structured_rr_values", rr_default)
                    rr_text = st.text_input(
                        "RR value(s)",
                        key="limited_structured_rr_values",
                        help="Enter a single RR or comma-separated RR values to sweep multiple settings.",
                    )
                    rr_ok, rr_value, rr_detail = _parse_csv_numeric_text(rr_text, kind="float")
                    structured_input_checks.append((rr_ok, "RR value(s)", rr_detail))
                    if rr_ok and rr_value is not None:
                        exit_params_dict[rr_key] = rr_value
                    if is_atr_brackets:
                        stop_dist_default = _value_to_csv_text(
                            exit_params_dict.get(
                                "sldist_atr_mult",
                                exit_params_dict.get("sldist_atr", 1.0),
                            )
                        )
                        _sync_widget_from_source("limited_structured_sldist_atr_mult", stop_dist_default)
                        stop_dist_text = st.text_input(
                            "Stop-distance ATR multiple(s)",
                            key="limited_structured_sldist_atr_mult",
                            help="Enter a single stop-distance ATR multiple or comma-separated values to sweep multiple settings.",
                        )
                        stop_dist_ok, stop_dist_value, stop_dist_detail = _parse_csv_numeric_text(stop_dist_text, kind="float")
                        structured_input_checks.append((stop_dist_ok, "Stop-distance ATR multiple(s)", stop_dist_detail))
                        if stop_dist_ok and stop_dist_value is not None:
                            exit_params_dict["sldist_atr_mult"] = stop_dist_value
                            exit_params_dict.pop("sldist_atr", None)
                    st.session_state["limited_exit_params_json"] = _json_pretty(exit_params_dict)
                    exit_params = st.session_state.get("limited_exit_params_json", _json_pretty(exit_params_dict))
                    if is_atr_brackets:
                        st.caption("RR and stop-distance ATR multiplier are shown here. Other exit parameters are preserved; use Advanced to edit plugin-specific fields like ATR period.")
                    else:
                        st.caption("Other exit parameters are preserved. Use Advanced if you need to edit plugin-specific fields.")
                elif str(current_exit_name or "") == "time_exit":
                    show_exit_json_in_main = False
                    hold_bars_default = _value_to_csv_text(exit_params_dict.get("hold_bars", 1))
                    _sync_widget_from_source("limited_structured_hold_bars", hold_bars_default)
                    hold_bars_text = st.text_input(
                        "Hold bars",
                        key="limited_structured_hold_bars",
                        help="Enter a single hold-bar value or comma-separated integers to sweep multiple settings.",
                    )
                    hold_ok, hold_value, hold_detail = _parse_csv_numeric_text(hold_bars_text, kind="int")
                    structured_input_checks.append((hold_ok, "Hold bars", hold_detail))
                    if hold_ok and hold_value is not None:
                        exit_params_dict["hold_bars"] = hold_value
                        st.session_state["limited_exit_params_json"] = _json_pretty(exit_params_dict)
                    exit_params = st.session_state.get("limited_exit_params_json", _json_pretty(exit_params_dict))
                    st.caption("Use comma-separated integers to test several fixed holding periods without editing JSON.")
                elif str(current_exit_name or "") in monkey_exit_plugin_names:
                    show_exit_json_in_main = False
                    avg_hold_default = float(
                        exit_params_dict.get("avg_hold_bars", exit_params_dict.get("hold_bars", 10.0)) or 10.0
                    )
                    if not math.isfinite(avg_hold_default) or avg_hold_default <= 0:
                        avg_hold_default = 10.0
                    _sync_widget_from_source("limited_structured_avg_hold_bars", float(avg_hold_default))
                    avg_hold_bars = float(
                        st.number_input(
                            "Average hold bars",
                            min_value=1.0,
                            step=0.25,
                            key="limited_structured_avg_hold_bars",
                        )
                    )
                    exit_params_dict["avg_hold_bars"] = float(avg_hold_bars)
                    exit_params_dict.pop("hold_bars_values", None)
                    exit_params_dict.pop("hold_bars", None)
                    st.session_state["limited_exit_params_json"] = _json_pretty(exit_params_dict)
                    exit_params = st.session_state["limited_exit_params_json"]
                    st.caption("This controls the mean hold time for monkey exit sampling. Seed settings remain in the Monkey Sampling section.")
                else:
                    exit_params = st.text_area(
                        scenario_exit_label,
                        height=150,
                        key="limited_exit_params_json",
                    )

            if isinstance(selected_workbook, dict):
                st.subheader("4) Workbook Outputs + Pass Criteria")
                default_output_rows = _build_workbook_output_rows(list(selected_workbook.get("outputs", [])))
                default_rule_rows = _build_workbook_rule_rows(list(selected_workbook.get("pass_criteria", [])))
                review_header_cols = st.columns([0.72, 0.28])
                with review_header_cols[0]:
                    st.caption("Structured workbook review rows. Edit labels and thresholds directly instead of maintaining freeform text.")
                with review_header_cols[1]:
                    if st.button("Reset to workbook defaults", key="limited_workbook_review_reset"):
                        st.session_state["limited_workbook_output_rows"] = copy.deepcopy(default_output_rows)
                        st.session_state["limited_workbook_rule_rows"] = copy.deepcopy(default_rule_rows)
                        st.session_state.pop("limited_workbook_output_rows_editor", None)
                        st.session_state.pop("limited_workbook_rule_rows_editor", None)
                        st.rerun()

                output_seed_rows = _clean_workbook_output_rows(list(st.session_state.get("limited_workbook_output_rows", [])))
                rule_seed_rows = _clean_workbook_rule_rows(list(st.session_state.get("limited_workbook_rule_rows", [])))
                if not output_seed_rows and default_output_rows:
                    output_seed_rows = copy.deepcopy(default_output_rows)
                if not rule_seed_rows and default_rule_rows:
                    rule_seed_rows = copy.deepcopy(default_rule_rows)

                review_cols = st.columns(2, gap="large")
                with review_cols[0]:
                    st.markdown("**Outputs**")
                    output_editor_df = st.data_editor(
                        pd.DataFrame(output_seed_rows, columns=["output_label", "mapped_metric", "notes"]),
                        key="limited_workbook_output_rows_editor",
                        use_container_width=True,
                        hide_index=True,
                        num_rows="dynamic",
                        column_config={
                            "output_label": st.column_config.TextColumn("Output label", width="large"),
                            "mapped_metric": st.column_config.SelectboxColumn(
                                "Mapped metric",
                                options=WORKBOOK_REVIEW_METRIC_OPTIONS,
                                width="medium",
                            ),
                            "notes": st.column_config.TextColumn("Notes", width="medium"),
                        },
                    )
                with review_cols[1]:
                    st.markdown("**Pass Rules**")
                    rule_editor_df = st.data_editor(
                        pd.DataFrame(rule_seed_rows, columns=["rule_label", "mapped_metric", "operator", "threshold", "unit"]),
                        key="limited_workbook_rule_rows_editor",
                        use_container_width=True,
                        hide_index=True,
                        num_rows="dynamic",
                        column_config={
                            "rule_label": st.column_config.TextColumn("Rule label", width="large"),
                            "mapped_metric": st.column_config.SelectboxColumn(
                                "Mapped metric",
                                options=WORKBOOK_REVIEW_METRIC_OPTIONS,
                                width="medium",
                            ),
                            "operator": st.column_config.SelectboxColumn(
                                "Op",
                                options=WORKBOOK_REVIEW_OPERATOR_OPTIONS,
                                width="small",
                            ),
                            "threshold": st.column_config.NumberColumn("Threshold", width="small", format="%.4f"),
                            "unit": st.column_config.SelectboxColumn(
                                "Unit",
                                options=WORKBOOK_REVIEW_UNIT_OPTIONS,
                                width="small",
                            ),
                        },
                    )

                workbook_output_rows = _clean_workbook_output_rows(output_editor_df.to_dict("records"))
                workbook_rule_rows = _clean_workbook_rule_rows(rule_editor_df.to_dict("records"))
                st.session_state["limited_workbook_output_rows"] = workbook_output_rows
                st.session_state["limited_workbook_rule_rows"] = workbook_rule_rows

                workbook_review_modified = (
                    _workbook_rows_fingerprint(workbook_output_rows) != _workbook_rows_fingerprint(default_output_rows)
                    or _workbook_rows_fingerprint(workbook_rule_rows) != _workbook_rows_fingerprint(default_rule_rows)
                )
                review_state = "Modified" if workbook_review_modified else "Workbook default"
                st.caption(
                    f"{review_state}. Reviewing {len(workbook_output_rows)} outputs. "
                    f"PASS when all {len(workbook_rule_rows)} workbook rules are satisfied. "
                    "Execution-specific thresholds remain in Advanced."
                )

            active_seed_mode = "none"
            if str(current_entry_name or "") in monkey_entry_plugin_names:
                active_seed_mode = "entry"
            if str(current_exit_name or "") in monkey_exit_plugin_names:
                active_seed_mode = "exit" if active_seed_mode == "none" else "entry"
            if selected_monkey_mode == "Monkey Entry + Exit":
                active_seed_mode = "entry"

            seed_count = ""
            seed_start = ""
            exit_seed_count = ""
            exit_seed_start = ""
            if active_seed_mode != "none":
                st.subheader("5) Monkey Sampling")
                st.caption("Seed controls are shown only when the active scenario uses monkey entry or monkey exit logic.")

            if active_seed_mode == "entry" and selected_monkey_mode == "Monkey Entry":
                seed_cols = st.columns(2)
                seed_count = seed_cols[0].text_input("Seed count (entry, optional)", key="limited_seed_count")
                seed_start = seed_cols[1].text_input("Seed start (entry, optional)", key="limited_seed_start")
                st.caption("Monkey Entry uses entry seeds only. Exit seeds are ignored for this mode.")
            elif active_seed_mode == "exit" and selected_monkey_mode == "Monkey Exit":
                seed_cols = st.columns(2)
                exit_seed_count = seed_cols[0].text_input("Exit seed count (optional)", key="limited_exit_seed_count")
                exit_seed_start = seed_cols[1].text_input("Exit seed start (optional)", key="limited_exit_seed_start")
                st.caption("Monkey Exit uses exit seeds only. Entry seeds are ignored for this mode.")
            elif active_seed_mode == "entry" and selected_monkey_mode == "Monkey Entry + Exit":
                seed_cols = st.columns(2)
                seed_count = seed_cols[0].text_input("Seed count (entry, optional)", key="limited_seed_count")
                seed_start = seed_cols[1].text_input("Seed start (entry, optional)", key="limited_seed_start")
                st.caption(
                    "Baseline helper seeds entry only (recommended) to avoid entry-seed x exit-seed Cartesian blow-up. "
                    "Exit seed fields remain hidden for this mode."
                )
            elif active_seed_mode == "entry":
                seed_cols = st.columns(2)
                seed_count = seed_cols[0].text_input("Seed count (entry)", key="limited_seed_count")
                seed_start = seed_cols[1].text_input("Seed start (entry)", key="limited_seed_start")
            elif active_seed_mode == "exit":
                seed_cols = st.columns(2)
                exit_seed_count = seed_cols[0].text_input("Exit seed count", key="limited_exit_seed_count")
                exit_seed_start = seed_cols[1].text_input("Exit seed start", key="limited_exit_seed_start")

            with st.expander("Advanced / technical overrides", expanded=selected_workbook is None):
                st.caption(
                    "Use this section when you need to override the scenario defaults, change plugins, or edit the engine-specific pass logic."
                )
                plugin_cols = st.columns(3)
                entry_plugin = plugin_cols[0].selectbox("Entry plugin", [default_entry, *entry_plugins], key="limited_entry_plugin_select")
                exit_plugin = plugin_cols[1].selectbox("Exit plugin", [default_exit, *exit_plugins], key="limited_exit_plugin_select")
                sizing_plugin = plugin_cols[2].selectbox("Sizing plugin", [default_sizing, *sizing_plugins], key="limited_sizing_plugin_select")

                selected_entry_name = entry_default_name if entry_plugin == default_entry else entry_plugin
                selected_exit_name = exit_default_name if exit_plugin == default_exit else exit_plugin
                selected_sizing_name = sizing_default_name if sizing_plugin == default_sizing else sizing_plugin

                raw_param_cols = st.columns(2)
                if not show_entry_json_in_main:
                    if selected_entry_name and selected_entry_name == entry_default_name and strategy_entry_contract.get("has_policy", False):
                        raw_param_cols[0].caption(
                            "Entry params JSON is read-only here because this strategy locks non-optimisable baseline entry fields."
                        )
                        raw_param_cols[0].code(st.session_state.get("limited_entry_params_json", ""), language="json")
                        entry_params = str(st.session_state.get("limited_entry_params_json", "") or "")
                    else:
                        entry_params = raw_param_cols[0].text_area(
                            "Entry params JSON (advanced)",
                            height=130,
                            key="limited_entry_params_json",
                        )
                if not show_exit_json_in_main:
                    if selected_exit_name and selected_exit_name == exit_default_name and strategy_exit_contract.get("has_policy", False):
                        raw_param_cols[1].caption(
                            "Exit params JSON is read-only here because this strategy locks non-optimisable baseline exit fields."
                        )
                        raw_param_cols[1].code(st.session_state.get("limited_exit_params_json", ""), language="json")
                        exit_params = str(st.session_state.get("limited_exit_params_json", "") or "")
                    else:
                        exit_params = raw_param_cols[1].text_area(
                            "Exit params JSON (advanced)",
                            height=130,
                            key="limited_exit_params_json",
                        )

                sizing_params = st.text_area("Sizing params JSON", height=110, key="limited_sizing_params_json")

                eval_cols = st.columns(4)
                pass_threshold = eval_cols[0].text_input("Pass threshold % (optional)", key="limited_pass_threshold")
                min_trades = eval_cols[1].text_input("Min trades (optional)", key="limited_min_trades")
                pip_size = eval_cols[2].text_input("Pip size", key="limited_pip_size")
                progress_every = int(eval_cols[3].number_input("Progress every", min_value=1, step=1, key="limited_progress_every"))

                favourable_criteria = st.text_area(
                    "Favourable criteria JSON (optional)",
                    height=86,
                    key="limited_favourable_criteria",
                )

                c6, c7 = st.columns(2)
                run_base = c6.text_input("Run base (optional)", key="limited_run_base")
                test_name = c7.text_input("Test name (optional)", key="limited_test_name")
                extra_args = st.text_area("Extra args (optional)", height=72, key="limited_extra_args")

            if selected_monkey_mode:
                st.markdown("#### Monkey Test (Baseline-Matched)")
                st.caption(
                    "Select a baseline limited run and apply a baseline-matched monkey setup. "
                    "This auto-fills target trades/long ratio/avg hold, dominance criteria, and seed defaults. "
                    "You can still edit the fields above after applying."
                )
                unlock_cols = st.columns([1.25, 2.75])
                unlock_overrides = unlock_cols[0].checkbox(
                    "Unlock overrides",
                    key="limited_monkey_unlock_overrides",
                    help="When off, baseline-derived monkey fields stay locked to the applied baseline setup.",
                )
                if unlock_overrides:
                    unlock_cols[1].caption("Manual edits are allowed. Re-apply baseline setup to re-lock/refresh derived values.")
                else:
                    unlock_cols[1].caption("Baseline-derived monkey fields are locked after apply. Turn on Unlock overrides to edit them.")
                monkey_base_opts = [p.as_posix() for p in limited_runs]
                monkey_base_default = st.session_state.get("limited_monkey_baseline_run", "") or (monkey_base_opts[0] if monkey_base_opts else "")
                monkey_base_run_raw = _select_path_from_options(
                    "Baseline limited run",
                    monkey_base_opts,
                    monkey_base_default,
                    key_prefix="limited_monkey_baseline",
                )
                if monkey_base_run_raw.strip():
                    st.session_state["limited_monkey_baseline_run"] = monkey_base_run_raw.strip()
                    loaded_base = _load_limited_run_artifacts(_abs_path(monkey_base_run_raw.strip()))
                    monkey_baseline_loaded_ok = loaded_base is not None
                else:
                    loaded_base = None
                    monkey_baseline_loaded_ok = None

                selected_base_row: pd.Series | None = None
                selected_base_metrics: dict[str, float] | None = None
                if loaded_base is None and monkey_base_run_raw.strip():
                    st.error("Could not load selected baseline limited run.")
                elif loaded_base is not None:
                    base_results, base_meta, _base_pass = loaded_base
                    base_iter_options: list[int] = []
                    if "iter" in base_results.columns:
                        base_iter_options = (
                            pd.to_numeric(base_results["iter"], errors="coerce")
                            .dropna()
                            .astype(int)
                            .drop_duplicates()
                            .sort_values()
                            .tolist()
                        )
                    selected_base_iter: int | None = None
                    if len(base_iter_options) > 1:
                        selected_base_iter = st.selectbox(
                            "Baseline iteration",
                            base_iter_options,
                            index=0,
                            key="limited_monkey_baseline_iter",
                        )
                    elif len(base_results) > 1:
                        st.warning("Baseline run has multiple rows; defaulting to first row. Select an iteration if available.")

                    selected_base_row = _pick_limited_row(base_results, iter_id=selected_base_iter)
                    if selected_base_row is not None:
                        base_initial_eq = _infer_initial_equity_from_spec(base_meta.get("spec", {}) if isinstance(base_meta, dict) else {})
                        selected_base_metrics = _limited_baseline_metrics_from_row(selected_base_row, initial_equity=base_initial_eq)
                        _render_metric_row(
                            [
                                ("Base Trades", selected_base_metrics.get("trades", float("nan")), "{:.0f}"),
                                ("Base Long %", selected_base_metrics.get("long_trade_pct", float("nan")), "{:.2f}"),
                                ("Base Avg Hold", selected_base_metrics.get("avg_bars_held", float("nan")), "{:.2f}"),
                                ("Base Return %", selected_base_metrics.get("total_return_%", float("nan")), "{:.2f}"),
                            ]
                        )
                        _render_metric_row(
                            [
                                ("Base MaxDD %", selected_base_metrics.get("max_drawdown_abs_%", float("nan")), "{:.2f}"),
                                ("Monkey Seeds", 8000, "{:d}"),
                                ("Davey PASS Floor %", 90, "{:d}"),
                                ("Constraint Note", "filter in Results tab", "{}"),
                            ]
                        )

                        def _apply_baseline_monkey_preset(mode_label: str) -> None:
                            assert selected_base_metrics is not None
                            base_trades = selected_base_metrics.get("trades", float("nan"))
                            base_long_pct = selected_base_metrics.get("long_trade_pct", float("nan"))
                            base_avg_hold = selected_base_metrics.get("avg_bars_held", float("nan"))
                            base_ret = selected_base_metrics.get("total_return_%", float("nan"))
                            base_dd = selected_base_metrics.get("max_drawdown_abs_%", float("nan"))
                            if not (np.isfinite(base_trades) and base_trades > 0 and np.isfinite(base_ret) and np.isfinite(base_dd)):
                                raise ValueError("Baseline run row is missing required metrics (trades, return, max drawdown).")

                            target_entries = max(1, int(round(float(base_trades))))
                            long_ratio = 0.5
                            if np.isfinite(base_long_pct):
                                long_ratio = max(0.0, min(1.0, float(base_long_pct) / 100.0))
                            avg_hold = 10.0
                            if np.isfinite(base_avg_hold) and float(base_avg_hold) > 0:
                                avg_hold = float(base_avg_hold)

                            criteria = _build_monkey_dominance_criteria(
                                baseline_return_pct=float(base_ret),
                                baseline_max_dd_pct=float(base_dd),
                            )
                            pending_updates: dict[str, Any] = {}
                            if mode_label in {"Monkey Entry", "Monkey Entry + Exit"}:
                                pending_updates["limited_entry_plugin_select"] = "monkey_entry"
                                pending_updates["limited_entry_params_json"] = _json_pretty(
                                    {"target_entries": target_entries, "side": "both", "long_ratio": long_ratio}
                                )
                                pending_updates["limited_seed_count"] = "8000"
                                pending_updates["limited_seed_start"] = "1"
                            if mode_label == "Monkey Exit":
                                pending_updates["limited_entry_plugin_select"] = default_entry
                                pending_updates["limited_seed_count"] = ""
                                pending_updates["limited_seed_start"] = ""

                            if mode_label in {"Monkey Exit", "Monkey Entry + Exit"}:
                                pending_updates["limited_exit_plugin_select"] = "monkey_exit"
                                pending_updates["limited_exit_params_json"] = _json_pretty({"avg_hold_bars": avg_hold})
                                if mode_label == "Monkey Exit":
                                    pending_updates["limited_exit_seed_count"] = "8000"
                                    pending_updates["limited_exit_seed_start"] = "1"
                                else:
                                    # Avoid Cartesian explosion (entry seeds x exit seeds).
                                    pending_updates["limited_exit_seed_count"] = ""
                                    pending_updates["limited_exit_seed_start"] = ""
                            if mode_label == "Monkey Entry":
                                pending_updates["limited_exit_plugin_select"] = default_exit
                                pending_updates["limited_exit_seed_count"] = ""
                                pending_updates["limited_exit_seed_start"] = ""

                            pending_updates["limited_favourable_criteria"] = json.dumps(criteria, separators=(",", ":"))
                            pending_updates["limited_pass_threshold"] = "90"
                            # Monkey tests should not fail due to a separate trade-count gate.
                            # Trade-count matching is handled by the monkey prefilter (for supported modes).
                            pending_updates["limited_min_trades"] = "0"
                            pending_updates["limited_test_name"] = (
                                "monkey_entry_exit_test__baseline_matched"
                                if mode_label == "Monkey Entry + Exit"
                                else ("monkey_entry_test__baseline_matched" if mode_label == "Monkey Entry" else "monkey_exit_test__baseline_matched")
                            )
                            monkey_prefilter_args_parts: list[str] = []
                            monkey_runtime_args_parts: list[str] = ["--monkey-davey-style"]
                            if mode_label in {"Monkey Exit", "Monkey Entry + Exit"}:
                                monkey_prefilter_args_parts.extend(
                                    [
                                        "--monkey-match-prefilter",
                                        "--monkey-match-target-trades",
                                        str(target_entries),
                                        "--monkey-match-trade-tol-pct",
                                        "5",
                                    ]
                                )
                                if np.isfinite(base_long_pct):
                                    monkey_prefilter_args_parts.extend(
                                        ["--monkey-match-target-long-pct", f"{float(base_long_pct):.6g}", "--monkey-match-long-tol-pp", "5"]
                                    )
                                if np.isfinite(avg_hold) and float(avg_hold) > 0:
                                    monkey_prefilter_args_parts.extend(
                                        ["--monkey-match-target-avg-hold", f"{float(avg_hold):.6g}", "--monkey-match-hold-tol-pct", "5"]
                                    )
                                monkey_runtime_args_parts.append("--monkey-fast-summary")
                            pending_updates["limited_extra_args"] = " ".join(
                                shlex.quote(str(x))
                                for x in [*monkey_prefilter_args_parts, *monkey_runtime_args_parts]
                            )
                            pending_updates["limited_monkey_unlock_overrides"] = False

                            derived_fields_payload: dict[str, Any] = {
                                "mode": mode_label,
                                "limited_entry_plugin_select": str(
                                    pending_updates.get("limited_entry_plugin_select", st.session_state.get("limited_entry_plugin_select", ""))
                                ),
                                "limited_exit_plugin_select": str(
                                    pending_updates.get("limited_exit_plugin_select", st.session_state.get("limited_exit_plugin_select", ""))
                                ),
                                "limited_entry_params_json": str(
                                    pending_updates.get("limited_entry_params_json", st.session_state.get("limited_entry_params_json", ""))
                                ),
                                "limited_exit_params_json": str(
                                    pending_updates.get("limited_exit_params_json", st.session_state.get("limited_exit_params_json", ""))
                                ),
                                "limited_seed_count": str(pending_updates.get("limited_seed_count", st.session_state.get("limited_seed_count", ""))),
                                "limited_seed_start": str(pending_updates.get("limited_seed_start", st.session_state.get("limited_seed_start", ""))),
                                "limited_exit_seed_count": str(
                                    pending_updates.get("limited_exit_seed_count", st.session_state.get("limited_exit_seed_count", ""))
                                ),
                                "limited_exit_seed_start": str(
                                    pending_updates.get("limited_exit_seed_start", st.session_state.get("limited_exit_seed_start", ""))
                                ),
                                "limited_favourable_criteria": str(
                                    pending_updates.get("limited_favourable_criteria", st.session_state.get("limited_favourable_criteria", ""))
                                ),
                                "limited_pass_threshold": str(
                                    pending_updates.get("limited_pass_threshold", st.session_state.get("limited_pass_threshold", ""))
                                ),
                                "limited_min_trades": str(pending_updates.get("limited_min_trades", st.session_state.get("limited_min_trades", ""))),
                                "limited_test_name": str(pending_updates.get("limited_test_name", st.session_state.get("limited_test_name", ""))),
                                "limited_extra_args": str(pending_updates.get("limited_extra_args", st.session_state.get("limited_extra_args", ""))),
                                "baseline_run": str(st.session_state.get("limited_monkey_baseline_run", "")),
                                "baseline_iter": selected_base_iter,
                            }
                            pending_updates["limited_monkey_derived_fields"] = derived_fields_payload
                            st.session_state["_limited_pending_form_updates"] = pending_updates
                            st.session_state["_limited_prefill_lock"] = True
                            st.rerun()

                        helper_key_suffix = selected_monkey_mode.lower().replace(" ", "_").replace("+", "plus")
                        if st.button(
                            f"Apply Baseline-Matched Setup ({selected_monkey_mode})",
                            key=f"limited_apply_{helper_key_suffix}_from_baseline",
                            type="secondary",
                        ):
                            try:
                                _apply_baseline_monkey_preset(selected_monkey_mode)
                            except Exception as e:
                                st.error(f"Unable to apply baseline-matched monkey setup: {e}")
                        if selected_monkey_mode == "Monkey Entry + Exit":
                            st.caption(
                                "For `Monkey Entry + Exit`, the helper seeds entry only (8,000) to avoid entry-seed x exit-seed Cartesian blow-up. "
                                "Use Results > Limited tests > Constrained Dominance to apply the ±5% filters against your baseline."
                            )

            def _json_check(label: str, raw_text: str) -> tuple[bool, str, str]:
                text = raw_text.strip()
                if not text:
                    return True, label, "blank (optional)"
                try:
                    json.loads(text)
                    return True, label, "valid JSON"
                except Exception as e:
                    return False, label, f"invalid JSON: {e}"

            def _numeric_check(label: str, raw_text: str, *, required: bool = False) -> tuple[bool, str, str]:
                text = raw_text.strip()
                if not text:
                    if required:
                        return False, label, "required"
                    return True, label, "blank (optional)"
                try:
                    float(text)
                    return True, label, "valid number"
                except Exception:
                    return False, label, "invalid number"

            checklist_rows: list[dict[str, Any]] = []
            for ok, label, detail in (
                _json_check("Entry params JSON", entry_params),
                _json_check("Exit params JSON", exit_params),
                _json_check("Sizing params JSON", sizing_params),
                _numeric_check("Slippage / spread (pips)", spread_pips, required=True),
                _numeric_check("Commission RT", commission_rt, required=True),
                _numeric_check("Pip size", pip_size),
            ):
                checklist_rows.append({"Check": label, "Status": "OK" if ok else "Fix", "Details": detail, "_ok": ok})
            for ok, label, detail in structured_input_checks:
                checklist_rows.append({"Check": label, "Status": "OK" if ok else "Fix", "Details": detail, "_ok": ok})

            if selected_monkey_mode:
                entry_is_monkey = str(selected_entry_name or "") == "monkey_entry"
                exit_is_monkey = str(selected_exit_name or "") == "monkey_exit"
                if selected_monkey_mode == "Monkey Entry":
                    mode_ok = entry_is_monkey and (not exit_is_monkey)
                    mode_detail = "Expect monkey entry + non-monkey exit."
                    seeds_ok = bool(seed_count.strip())
                    seeds_detail = "Entry seed count required." if not seeds_ok else f"Entry seeds={seed_count.strip()}."
                elif selected_monkey_mode == "Monkey Exit":
                    mode_ok = (not entry_is_monkey) and exit_is_monkey
                    mode_detail = "Expect non-monkey entry + monkey exit."
                    seeds_ok = bool(exit_seed_count.strip())
                    seeds_detail = "Exit seed count required." if not seeds_ok else f"Exit seeds={exit_seed_count.strip()}."
                else:
                    mode_ok = entry_is_monkey and exit_is_monkey
                    mode_detail = "Expect monkey entry + monkey exit."
                    seeds_ok = bool(seed_count.strip())
                    seeds_detail = "Entry seed count required (helper uses entry-only seeds)." if not seeds_ok else f"Entry seeds={seed_count.strip()}."

                criteria_ok = bool(favourable_criteria.strip()) and bool(pass_threshold.strip()) and bool(min_trades.strip())
                criteria_detail = (
                    "Need favourable criteria + pass threshold + min trades."
                    if not criteria_ok
                    else "Criteria/threshold/min-trades set."
                )
                baseline_selected = bool(monkey_base_run_raw.strip())
                baseline_ok = bool(monkey_baseline_loaded_ok) if baseline_selected else False
                baseline_detail = (
                    "Select a baseline limited run."
                    if not baseline_selected
                    else ("Loaded baseline run." if baseline_ok else "Selected baseline could not be loaded.")
                )
                lock_payload = st.session_state.get("limited_monkey_derived_fields")
                lock_matches = (
                    isinstance(lock_payload, dict)
                    and str(lock_payload.get("mode", "")) == str(selected_monkey_mode)
                    and str(lock_payload.get("baseline_run", "")).strip() == monkey_base_run_raw.strip()
                )
                lock_detail = (
                    "Derived fields locked to applied baseline."
                    if (not bool(st.session_state.get("limited_monkey_unlock_overrides", False)) and lock_matches)
                    else (
                        "Overrides unlocked (manual edits allowed)."
                        if bool(st.session_state.get("limited_monkey_unlock_overrides", False))
                        else "Apply baseline-matched setup to lock derived fields."
                    )
                )
                lock_ok = bool(st.session_state.get("limited_monkey_unlock_overrides", False)) or lock_matches

                checklist_rows.extend(
                    [
                        {"Check": "Monkey mode plugins", "Status": "OK" if mode_ok else "Fix", "Details": mode_detail, "_ok": mode_ok},
                        {"Check": "Monkey seeds", "Status": "OK" if seeds_ok else "Fix", "Details": seeds_detail, "_ok": seeds_ok},
                        {"Check": "Baseline run", "Status": "OK" if baseline_ok else "Fix", "Details": baseline_detail, "_ok": baseline_ok},
                        {"Check": "Dominance criteria", "Status": "OK" if criteria_ok else "Fix", "Details": criteria_detail, "_ok": criteria_ok},
                        {"Check": "Baseline-derived fields", "Status": "OK" if lock_ok else "Fix", "Details": lock_detail, "_ok": lock_ok},
                    ]
                )

            estimate_entry_raw = str(entry_params or "").strip()
            estimate_exit_raw = str(exit_params or "").strip()
            if selected_entry_name and selected_entry_name == entry_default_name and strategy_entry_contract.get("has_policy", False):
                estimate_entry_raw = json.dumps(
                    _sanitize_strategy_limited_params(
                        _json_object_or_empty(estimate_entry_raw),
                        _strategy_default_entry_params(),
                        strategy_entry_contract,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            if selected_exit_name and selected_exit_name == exit_default_name and strategy_exit_contract.get("has_policy", False):
                estimate_exit_raw = json.dumps(
                    _sanitize_strategy_limited_params(
                        _json_object_or_empty(estimate_exit_raw),
                        _strategy_default_exit_params(),
                        strategy_exit_contract,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            iteration_estimate = _estimate_limited_iterations(
                entry_plugin_name=selected_entry_name,
                entry_params_raw=estimate_entry_raw,
                exit_plugin_name=selected_exit_name,
                exit_params_raw=estimate_exit_raw,
                seed_count_raw=seed_count,
                seed_start_raw=seed_start,
                exit_seed_count_raw=exit_seed_count,
                exit_seed_start_raw=exit_seed_start,
            )
            estimate_ok = bool(iteration_estimate.get("ok", False))
            estimate_detail = (
                f"entry={int(iteration_estimate.get('entry_variants', 0)):,} x "
                f"exit={int(iteration_estimate.get('exit_variants', 0)):,} = "
                f"{int(iteration_estimate.get('total', 0)):,} iterations"
                if estimate_ok
                else str(iteration_estimate.get("error", "unable to estimate"))
            )
            checklist_rows.append(
                {
                    "Check": "Iteration estimate",
                    "Status": "OK" if estimate_ok else "Fix",
                    "Details": estimate_detail,
                    "_ok": estimate_ok,
                }
            )

            ready_ok = all(bool(r.get("_ok", False)) for r in checklist_rows)
            st.subheader("6) Run")
            if estimate_ok:
                _render_metric_row(
                    [
                        ("Estimated iterations", int(iteration_estimate.get("total", 0)), "{:,d}"),
                        ("Entry variants", int(iteration_estimate.get("entry_variants", 0)), "{:,d}"),
                        ("Exit variants", int(iteration_estimate.get("exit_variants", 0)), "{:,d}"),
                        ("Skipped invalid entry sets", int(iteration_estimate.get("skipped_entry_variants", 0)), "{:,d}"),
                    ]
                )
                caveats = [str(x).strip() for x in list(iteration_estimate.get("caveats", [])) if str(x).strip()]
                if caveats:
                    st.caption("Estimate notes: " + " | ".join(caveats))
                st.caption(
                    "Estimate is based on the current entry/exit parameter grids and seed settings before execution. "
                    "Actual produced runs can still differ if runtime options stop early."
                )
            else:
                st.warning(f"Iteration estimate unavailable: {iteration_estimate.get('error', 'unknown error')}")
            run_cols = st.columns([0.28, 0.72], gap="large")
            with run_cols[0]:
                _render_status_badge("Run Readiness", ready_ok)
            with run_cols[1]:
                st.caption(
                    "Validation is computed from the current form state. Review the detailed checks if you are running a monkey workflow or custom JSON."
                )
                auto_run_after_prefill = bool(st.session_state.pop("_limited_auto_run_after_prefill", False))
                run_limited_clicked = st.button("Run limited tests", type="primary")
            st.caption(
                f"Run target: scenario=`{selected_workbook_template}` | "
                f"entry=`{selected_entry_name or 'none'}` | "
                f"exit=`{selected_exit_name or 'none'}` | "
                f"spread={spread_pips or 'missing'} pips"
            )
            checklist_df = pd.DataFrame([{k: v for k, v in r.items() if not str(k).startswith("_")} for r in checklist_rows])
            with st.expander("Validation details", expanded=not ready_ok):
                st.dataframe(checklist_df, use_container_width=True, hide_index=True)

            if run_limited_clicked or auto_run_after_prefill:
                try:
                    if auto_run_after_prefill and not run_limited_clicked:
                        st.info("Auto-running selected workbook scenario with prefilled fields.")
                    cmd = [
                        SCRIPT_PYTHON,
                        str(SCRIPTS_DIR / "run_limited_tests.py"),
                        "--strategy",
                        strategy_module,
                        "--data",
                        data_path,
                        "--progress-every",
                        str(progress_every),
                    ]
                    if run_base.strip():
                        cmd.extend(["--run-base", run_base.strip()])
                    if test_name.strip():
                        cmd.extend(["--test-name", test_name.strip()])

                    entry_params_raw = entry_params.strip()
                    exit_params_raw = exit_params.strip()
                    sizing_params_raw = sizing_params.strip()
                    spread_pips_raw = spread_pips.strip()
                    commission_rt_raw = commission_rt.strip()
                    seed_count_raw = seed_count.strip()
                    seed_start_raw = seed_start.strip()
                    exit_seed_count_raw = exit_seed_count.strip()
                    exit_seed_start_raw = exit_seed_start.strip()

                    if not spread_pips_raw:
                        raise ValueError("Slippage / spread (pips) is required.")
                    if not commission_rt_raw:
                        raise ValueError("Commission RT is required.")

                    if selected_entry_name and selected_entry_name == entry_default_name and strategy_entry_contract.get("has_policy", False):
                        entry_params_obj = _json_object_or_empty(entry_params_raw)
                        entry_params_raw = json.dumps(
                            _sanitize_strategy_limited_params(
                                entry_params_obj,
                                _strategy_default_entry_params(),
                                strategy_entry_contract,
                            ),
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        )
                    if selected_exit_name and selected_exit_name == exit_default_name and strategy_exit_contract.get("has_policy", False):
                        exit_params_obj = _json_object_or_empty(exit_params_raw)
                        exit_params_raw = json.dumps(
                            _sanitize_strategy_limited_params(
                                exit_params_obj,
                                _strategy_default_exit_params(),
                                strategy_exit_contract,
                            ),
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        )

                    use_entry_plugin = (
                        entry_plugin != default_entry
                        or bool(entry_params_raw)
                        or bool(seed_count_raw)
                    )
                    use_exit_plugin = (
                        exit_plugin != default_exit
                        or bool(exit_params_raw)
                        or bool(exit_seed_count_raw)
                    )
                    use_sizing_plugin = (
                        sizing_plugin != default_sizing
                        or bool(sizing_params_raw)
                    )

                    if use_entry_plugin:
                        if not selected_entry_name:
                            raise ValueError("Could not resolve entry plugin. Select one explicitly.")
                        cmd.extend(["--entry-plugin", str(selected_entry_name)])
                    if entry_params_raw:
                        cmd.extend(["--entry-params", entry_params_raw])

                    if use_exit_plugin:
                        if not selected_exit_name:
                            raise ValueError("Could not resolve exit plugin. Select one explicitly.")
                        cmd.extend(["--exit-plugin", str(selected_exit_name)])
                    if exit_params_raw:
                        cmd.extend(["--exit-params", exit_params_raw])

                    if use_sizing_plugin:
                        if not selected_sizing_name:
                            raise ValueError("Could not resolve sizing plugin. Select one explicitly.")
                        cmd.extend(["--sizing-plugin", str(selected_sizing_name)])
                    if sizing_params_raw:
                        cmd.extend(["--sizing-params", sizing_params_raw])

                    cmd.extend(["--spread-pips", spread_pips_raw])
                    cmd.extend(["--commission-rt", commission_rt_raw])
                    if pip_size.strip():
                        cmd.extend(["--pip-size", pip_size.strip()])
                    if seed_count_raw:
                        cmd.extend(["--seed-count", seed_count_raw])
                    if seed_start_raw:
                        cmd.extend(["--seed-start", seed_start_raw])
                    if exit_seed_count_raw:
                        cmd.extend(["--exit-seed-count", exit_seed_count_raw])
                    if exit_seed_start_raw:
                        cmd.extend(["--exit-seed-start", exit_seed_start_raw])
                    if favourable_criteria.strip():
                        cmd.extend(["--favourable-criteria", favourable_criteria.strip()])
                    if pass_threshold.strip():
                        cmd.extend(["--pass-threshold", pass_threshold.strip()])
                    if min_trades.strip():
                        cmd.extend(["--min-trades", min_trades.strip()])

                    extra_tokens = _parse_extra_args(extra_args)
                    monkey_plugins = {"monkey_entry", "monkey_exit", "random", "random_time_exit"}
                    uses_monkey_plugin = (
                        str(selected_entry_name or "").strip() in monkey_plugins
                        or str(selected_exit_name or "").strip() in monkey_plugins
                    )
                    if uses_monkey_plugin:
                        if not _cli_flag_present(extra_tokens, "--monkey-davey-style"):
                            extra_tokens.append("--monkey-davey-style")
                        if not _cli_flag_present(extra_tokens, "--no-save-trades"):
                            cmd.append("--no-save-trades")
                        if not _cli_flag_present(extra_tokens, "--signal-cache-max"):
                            cmd.extend(["--signal-cache-max", "0"])

                    cmd.extend(extra_tokens)
                except ValueError as e:
                    st.error(f"Invalid arguments: {e}")
                else:
                    strategy_label = _strategy_display_name(strategy_short, strategy_catalog)
                    st.info(
                        f"Running `{selected_workbook_template}` on `{strategy_label}` using `{Path(data_path).name}`. "
                        "Progress updates stream below."
                    )
                    rc, out = _run_cli_live(cmd, workflow="limited")
                    if rc == 0:
                        saved_csv = _extract_saved_path(out, "Saved:")
                        if saved_csv is not None:
                            st.session_state["last_limited_run"] = saved_csv.parent.as_posix()
                            st.success(f"Saved run: {_rel(saved_csv.parent)}")
                            ok, msg = _auto_generate_interactive_html(run_kind="limited", run_path=saved_csv.parent)
                            if ok:
                                st.info(f"Auto-generated interactive HTML: `{msg}`")
                            else:
                                st.warning(f"Limited-test HTML auto-generation failed: {msg}")

            last_limited = st.session_state.get("last_limited_run", "")
            if last_limited:
                st.markdown("---")
                st.subheader("Last limited test output")
                _render_limited_workbook_summary(_abs_path(last_limited), strategy_catalog=strategy_catalog)


    elif page == "results":
        st.header("Results")
        results_action_notice = str(st.session_state.pop("_results_action_notice", "") or "").strip()
        if results_action_notice:
            st.success(results_action_notice)
        if _render_focused_results_page(strategy_catalog=strategy_catalog):
            return
        t_browser, t_live = st.tabs(["Run browser", "Live pilot"])

        with t_browser:
            _render_hierarchical_results_browser(
                strategy_run_rows=strategy_run_rows,
                strategy_catalog=strategy_catalog,
            )

        with t_live:
            _render_live_pilot_browser(
                mc_runs=mc_runs,
                live_trade_files=live_trade_files,
            )

        with st.expander("Clean up run outputs"):
            st.markdown(
                "Use this to delete old run directories under `runs/`. By default this is a dry run to avoid accidental data loss."
            )
            keep_runs = st.number_input(
                "Keep most recent run directories per workflow", min_value=0, value=1, help="Set to 0 to delete all run directories for each workflow."
            )
            dry_run = st.checkbox(
                "Dry run (no deletions)",
                value=True,
                help="Preview what would be deleted. Uncheck to actually delete folders (requires explicit confirmation).",
            )

            if st.button("Run cleanup"):
                cmd = [
                    SCRIPT_PYTHON,
                    str(SCRIPTS_DIR / "clean_runs.py"),
                    "--keep",
                    str(keep_runs),
                ]
                if not dry_run:
                    cmd.append("--force")

                rc, out = _run_cli_live(cmd, workflow="cleanup")
                if rc == 0:
                    st.success("Cleanup completed.")
                else:
                    st.error("Cleanup command failed. See output above.")

    else:
        _render_reference_page(
            strategy_catalog,
            plugin_catalog,
            strategy_run_rows,
            qp_get=_qp_get,
            build_query_url=_build_query_url,
            json_pretty=_json_pretty,
        )


if __name__ == "__main__":
    main()
