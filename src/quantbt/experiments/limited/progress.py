from __future__ import annotations

import math


def _fmt_value(v):
    if v is None:
        return "nan"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.2f}"
    return str(v)


def _criteria_status(criteria: dict, summary: dict) -> str:
    parts = []
    for rule in criteria.get("rules", []):
        metric = rule["metric"]
        op = rule["op"]
        target = rule["value"]
        got = summary.get(metric)
        parts.append(f"{metric}{op}{_fmt_value(target)} (got {_fmt_value(got)})")
    mode = criteria.get("mode", "all")
    joined = "; ".join(parts) if parts else "none"
    return f"{mode}: {joined}"


def print_progress(i, total, elapsed, last_summary, *, pass_pct, criteria):
    pct = 100 * i / total if total else 100.0
    rate = elapsed / i if i > 0 else 0.0
    eta = rate * (total - i) if total else 0.0
    print(
        f"[{i:>4}/{total}] "
        f"{pct:6.2f}% | "
        f"elapsed {elapsed:6.1f}s | "
        f"ETA {eta:6.1f}s | "
        f"last_ret {last_summary.get('total_return_%', float('nan')):6.2f}% | "
        f"pass {pass_pct:6.2f}% | "
        f"criteria {_criteria_status(criteria, last_summary)}",
        flush=True,
    )
