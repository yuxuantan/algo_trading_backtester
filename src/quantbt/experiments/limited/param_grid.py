from __future__ import annotations

import itertools

from quantbt.plugins import get_entry


def expand_params(params: dict) -> list[dict]:
    base = {}
    grid_keys = []
    grid_values = []
    for k, v in params.items():
        if isinstance(v, list) and not k.endswith("_values"):
            grid_keys.append(k)
            grid_values.append(v)
        else:
            base[k] = v

    if not grid_keys:
        return [dict(base)]

    combos = []
    for values in itertools.product(*grid_values):
        item = dict(base)
        item.update(dict(zip(grid_keys, values)))
        combos.append(item)
    return combos


def build_entry_variants(rules: list[dict]) -> tuple[list[list[dict]], list[tuple[str, dict]]]:
    if not rules:
        raise ValueError("strategy.entry.rules must be non-empty")

    entry_variants: list[list[dict]] = []
    skipped: list[tuple[str, dict]] = []
    for rule in rules:
        name = rule["name"]
        plugin = get_entry(name)
        params_list = expand_params(rule.get("params", {}))
        valid = []
        for p in params_list:
            validator = getattr(plugin, "validate", None)
            if callable(validator) and not validator(p):
                skipped.append((name, p))
                continue
            valid.append({"name": name, "plugin": plugin, "params": p})
        if not valid:
            raise ValueError(f"no valid params for entry rule '{name}'")
        entry_variants.append(valid)

    return entry_variants, skipped


def build_exit_param_space(exit_spec: dict) -> list[dict]:
    exit_param_space = expand_params(exit_spec.get("params", {}))
    if not exit_param_space:
        return [{}]
    return exit_param_space


def total_iterations(entry_variants: list[list[dict]], exit_param_space: list[dict]) -> int:
    total = 1
    for variants in entry_variants:
        total *= len(variants)
    total *= len(exit_param_space)
    return total
