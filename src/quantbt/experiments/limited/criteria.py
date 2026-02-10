from __future__ import annotations

import json
import math
from pathlib import Path


def load_json_arg(value: str):
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def parse_favourable_criteria(value):
    if value is None:
        return {"mode": "all", "rules": [{"metric": "avg_profit_per_trade", "op": ">", "value": 0.0}]}

    if isinstance(value, (dict, list)):
        data = value
    else:
        data = load_json_arg(value)
    if isinstance(data, dict):
        if "rules" in data:
            mode = data.get("mode", "all")
            rules = data["rules"]
        else:
            rules = []
            for metric, spec in data.items():
                if isinstance(spec, dict):
                    if len(spec) != 1:
                        raise ValueError(f"criteria for {metric} must have exactly one operator")
                    op, val = next(iter(spec.items()))
                else:
                    op, val = ">=", spec
                rules.append({"metric": metric, "op": op, "value": val})
            mode = "all"
    elif isinstance(data, list):
        rules = data
        mode = "all"
    else:
        raise ValueError("favourable-criteria must be JSON dict or list")

    for r in rules:
        if "metric" not in r or "op" not in r or "value" not in r:
            raise ValueError("each rule must include metric, op, value")
    if mode not in ("all", "any"):
        raise ValueError("criteria mode must be 'all' or 'any'")
    return {"mode": mode, "rules": rules}


def criteria_pass(summary: dict, criteria: dict) -> bool:
    ops = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b,
           "<=": lambda a, b: a <= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}

    def _rule_ok(rule):
        metric = rule["metric"]
        op = rule["op"]
        value = rule["value"]
        if op not in ops:
            raise ValueError(f"unsupported operator: {op}")
        v = summary.get(metric)
        if v is None:
            return False
        if isinstance(v, float) and math.isnan(v):
            return False
        return ops[op](v, value)

    results = [_rule_ok(r) for r in criteria["rules"]]
    return all(results) if criteria["mode"] == "all" else any(results)
