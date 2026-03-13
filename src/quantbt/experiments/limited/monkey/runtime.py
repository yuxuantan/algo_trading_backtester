from __future__ import annotations

import math

from .constants import MONKEY_ENTRY_PLUGIN_NAMES, MONKEY_TIME_EXIT_PLUGIN_NAMES


def load_monkey_runtime_cfg(spec: dict) -> dict:
    test_cfg = spec.get("test", {}) if isinstance(spec, dict) else {}
    raw = test_cfg.get("monkey_runtime")
    if not isinstance(raw, dict):
        return {"fast_summary": False, "sequential_stop": {"enabled": False}}
    seq = raw.get("sequential_stop")
    seq_cfg = seq if isinstance(seq, dict) else {}
    return {
        "fast_summary": bool(raw.get("fast_summary", False)),
        "sequential_stop": {
            "enabled": bool(seq_cfg.get("enabled", False)),
            "min_accepted": max(1, int(seq_cfg.get("min_accepted", 1000))),
            "check_every": max(1, int(seq_cfg.get("check_every", 200))),
            "fail_threshold_pct": float(seq_cfg.get("fail_threshold_pct", 75.0)),
            "z": float(seq_cfg.get("z", 1.96)),
        },
    }


def _is_monkey_test_spec(spec: dict) -> bool:
    if not isinstance(spec, dict):
        return False
    strategy = spec.get("strategy")
    if not isinstance(strategy, dict):
        return False
    entry_cfg = strategy.get("entry")
    exit_cfg = strategy.get("exit")
    rules = entry_cfg.get("rules", []) if isinstance(entry_cfg, dict) else []
    entry_names = {
        str(r.get("name", "")).strip()
        for r in rules
        if isinstance(r, dict)
    }
    exit_name = str(exit_cfg.get("name", "")).strip() if isinstance(exit_cfg, dict) else ""
    return bool(
        entry_names.intersection(MONKEY_ENTRY_PLUGIN_NAMES)
        or exit_name in MONKEY_TIME_EXIT_PLUGIN_NAMES
    )


def _extract_monkey_baselines_from_criteria(criteria: dict) -> dict | None:
    if not isinstance(criteria, dict):
        return None
    rules = criteria.get("rules", [])
    if not isinstance(rules, list):
        return None

    baseline_return = None
    baseline_dd = None
    dd_metric = "max_drawdown_abs_%"
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        metric = str(rule.get("metric", "")).strip()
        op = str(rule.get("op", "")).strip()
        try:
            value = float(rule.get("value"))
        except Exception:
            continue
        if metric == "total_return_%" and op == "<":
            baseline_return = value
        elif metric in {"max_drawdown_abs_%", "max_drawdown_%"} and op == ">":
            baseline_dd = value
            dd_metric = metric

    if baseline_return is None or baseline_dd is None:
        return None
    if not math.isfinite(float(baseline_return)) or not math.isfinite(float(baseline_dd)):
        return None
    return {
        "baseline_return_%": float(baseline_return),
        "baseline_max_dd_%": float(baseline_dd),
        "dd_metric": str(dd_metric),
    }


def load_monkey_davey_cfg(spec: dict, *, criteria: dict, pass_threshold: float) -> dict:
    test_cfg = spec.get("test", {}) if isinstance(spec, dict) else {}
    raw = test_cfg.get("monkey_davey")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {"enabled": False}
    if not _is_monkey_test_spec(spec):
        return {"enabled": False}

    baseline = _extract_monkey_baselines_from_criteria(criteria) or {}
    ret_raw = raw.get("baseline_return_%", raw.get("baseline_return_pct"))
    dd_raw = raw.get("baseline_max_dd_%", raw.get("baseline_max_dd_pct"))
    dd_metric = str(raw.get("dd_metric", baseline.get("dd_metric", "max_drawdown_abs_%"))).strip() or "max_drawdown_abs_%"

    baseline_return = baseline.get("baseline_return_%") if ret_raw is None else ret_raw
    baseline_dd = baseline.get("baseline_max_dd_%") if dd_raw is None else dd_raw
    try:
        baseline_return = float(baseline_return)
        baseline_dd = float(baseline_dd)
    except Exception:
        return {
            "enabled": False,
            "error": (
                "monkey_davey requires baseline return/maxDD thresholds; "
                "supply them in favourable criteria or monkey_davey config."
            ),
        }
    if not math.isfinite(baseline_return) or not math.isfinite(baseline_dd):
        return {
            "enabled": False,
            "error": "monkey_davey baseline thresholds must be finite numbers.",
        }

    threshold_raw = raw.get("pass_threshold_%", raw.get("pass_threshold_pct", pass_threshold))
    try:
        threshold = float(threshold_raw)
    except Exception:
        threshold = float(pass_threshold)
    if not math.isfinite(threshold):
        threshold = float(pass_threshold)

    return {
        "enabled": True,
        "baseline_return_%": float(baseline_return),
        "baseline_max_dd_%": float(baseline_dd),
        "dd_metric": str(dd_metric),
        "pass_threshold_%": float(threshold),
    }


def wilson_interval(successes: int, n: int, *, z: float) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = float(successes) / float(n)
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (float(lo), float(hi))
