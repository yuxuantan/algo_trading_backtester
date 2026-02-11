from __future__ import annotations

MONKEY_ENTRY_PLUGINS = {"monkey_entry", "random"}
MONKEY_EXIT_PLUGINS = {"monkey_exit", "random_time_exit"}


def _entry_plugin_names(entry_spec: dict) -> tuple[str, ...]:
    rules = entry_spec.get("rules", [])
    names = [str(rule.get("name", "")).strip() for rule in rules if str(rule.get("name", "")).strip()]
    return tuple(sorted(names))


def _exit_plugin_name(strategy_spec: dict) -> str:
    return str(strategy_spec.get("exit", {}).get("name", "")).strip()


def classify_test_focus(strategy_spec: dict, base_strategy_spec: dict) -> str:
    current_entry = _entry_plugin_names(strategy_spec.get("entry", {}))
    base_entry = _entry_plugin_names(base_strategy_spec.get("entry", {}))

    current_exit = _exit_plugin_name(strategy_spec)
    base_exit = _exit_plugin_name(base_strategy_spec)

    entry_same = bool(current_entry) and current_entry == base_entry
    exit_same = bool(current_exit) and current_exit == base_exit

    if entry_same and exit_same:
        return "core_system_test"
    if entry_same and not exit_same:
        return "entry_test"
    if exit_same and not entry_same:
        return "exit_test"
    if set(current_entry).issubset(MONKEY_ENTRY_PLUGINS) and current_exit in MONKEY_EXIT_PLUGINS:
        return "monkey_entry_exit_test"

    raise ValueError(
        "Both entry and exit plugins differ from the base strategy. "
        "This run is blocked because it is not testing the original strategy "
        "(only monkey_entry+monkey_exit is allowed as an exception)."
    )


def infer_test_name(strategy_spec: dict, *, test_focus: str) -> str:
    entry = strategy_spec.get("entry", {})
    exit_ = strategy_spec.get("exit", {})
    rules = entry.get("rules", [])

    entry_names = [str(r.get("name", "entry")) for r in rules] or ["entry"]
    entry_tag = "+".join(entry_names)
    exit_name = str(exit_.get("name", "exit"))

    if test_focus not in {"core_system_test", "entry_test", "exit_test", "monkey_entry_exit_test"}:
        raise ValueError(f"unsupported test_focus: {test_focus}")

    exit_style_map = {
        "atr_brackets": "fixed_atr_exit",
        "time_exit": "time_exit",
        "random_time_exit": "random_exit",
        "monkey_exit": "monkey_exit",
    }
    exit_tag = exit_style_map.get(exit_name, f"{exit_name}_exit")
    return f"{test_focus}__{entry_tag}__{exit_tag}"

