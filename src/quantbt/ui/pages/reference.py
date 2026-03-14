from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st


def _strategy_default_entry_params(strategy_cfg: dict[str, Any]) -> dict[str, Any]:
    entry_cfg = strategy_cfg.get("entry", {}) if isinstance(strategy_cfg, dict) else {}
    rules = entry_cfg.get("rules", []) if isinstance(entry_cfg, dict) else []
    if isinstance(rules, list) and rules and isinstance(rules[0], dict):
        return dict(rules[0].get("params", {}) or {})
    return {}


def _strategy_default_exit_params(strategy_cfg: dict[str, Any]) -> dict[str, Any]:
    exit_cfg = strategy_cfg.get("exit", {}) if isinstance(strategy_cfg, dict) else {}
    return dict(exit_cfg.get("params", {}) or {}) if isinstance(exit_cfg, dict) else {}


def _optimizable_param_rows(section_cfg: dict[str, Any], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    raw = section_cfg.get("optimizable", {}) if isinstance(section_cfg, dict) else {}
    if isinstance(raw, list):
        raw = {str(key): {} for key in raw}
    if not isinstance(raw, dict):
        return []

    rows: list[dict[str, Any]] = []
    for key, spec in raw.items():
        key = str(key or "").strip()
        if not key:
            continue
        spec = spec if isinstance(spec, dict) else {}
        start = spec.get("start")
        end = spec.get("end", spec.get("stop"))
        step = spec.get("step")
        rows.append(
            {
                "Param": key,
                "Label": str(spec.get("label", key.replace("_", " "))),
                "Default": spec.get("default", defaults.get(key)),
                "Start": start,
                "End": end,
                "Step": step,
                "Integer": bool(spec.get("integer", False)),
            }
        )
    return rows


def _fixed_param_rows(section_cfg: dict[str, Any], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    raw = section_cfg.get("non_optimizable", []) if isinstance(section_cfg, dict) else []
    if not isinstance(raw, list):
        return []

    rows: list[dict[str, Any]] = []
    for key in raw:
        key = str(key or "").strip()
        if not key:
            continue
        rows.append({"Param": key, "Value": defaults.get(key)})
    return rows


def _run_history_frame(rows: list[dict[str, Any]], *, workflow: str | None = None) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if workflow and str(row.get("workflow", "")).strip().lower() != str(workflow).strip().lower():
            continue
        selected.append(
            {
                "Workflow": row.get("workflow", ""),
                "Category": row.get("category_label", row.get("category", "")),
                "Result": row.get("result", ""),
                "Dataset": row.get("dataset_slug", ""),
                "Date Window": row.get("date_window", ""),
                "Created": row.get("created_at", ""),
                "Run ID": row.get("run_id", ""),
                "Open result": row.get("result_url", ""),
                "Path": row.get("path", ""),
            }
        )
    if not selected:
        return pd.DataFrame()
    df = pd.DataFrame(selected)
    return df.sort_values(["Created", "Workflow"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _render_run_history_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No runs found.")
        return
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Open result": st.column_config.LinkColumn(
                "Open result",
                display_text="Open",
            )
        },
    )


def _reference_link(label: str, *, ref_type: str, ref_name: str, build_query_url: Callable[..., str]) -> str:
    return f"[{label}]({build_query_url(page='reference', ref_type=ref_type, ref_name=ref_name)})"


def _render_logic_block(
    title: str,
    *,
    name: str,
    params: dict[str, Any],
    ref_type: str,
    extra_lines: list[str] | None = None,
    json_pretty: Callable[[Any], str],
    build_query_url: Callable[..., str],
) -> None:
    st.markdown(f"**{title}**")
    if name:
        st.markdown(
            f"Plugin: {_reference_link(name, ref_type=ref_type, ref_name=name, build_query_url=build_query_url)}"
        )
    else:
        st.markdown("Plugin: `n/a`")
    for line in extra_lines or []:
        st.caption(line)
    st.code(json_pretty(params or {}), language="json")


def render_reference_page(
    strategy_catalog: dict[str, dict[str, Any]],
    plugin_catalog: dict[str, dict[str, dict[str, Any]]],
    strategy_run_rows: list[dict[str, Any]],
    *,
    qp_get: Callable[[str, str], str],
    build_query_url: Callable[..., str],
    json_pretty: Callable[[Any], str],
) -> None:
    st.header("Strategies")

    req_type = qp_get("ref_type", "strategy")
    req_name = qp_get("ref_name", "")
    section_labels = {
        "strategy": "Strategies",
        "entry": "Entry Plugins",
        "exit": "Exit Plugins",
        "sizing": "Sizing Plugins",
    }
    section_order = list(section_labels.keys())
    initial_section = req_type if req_type in section_labels else "strategy"
    section = st.radio(
        "Reference section",
        options=section_order,
        index=section_order.index(initial_section),
        format_func=lambda key: section_labels[key],
        horizontal=True,
        key="ref_section",
    )

    if section == "strategy":
        names = sorted(strategy_catalog.keys())
        if not names:
            st.info("No strategies discovered.")
        else:
            idx = names.index(req_name) if req_type == "strategy" and req_name in names else 0
            selected = st.selectbox("Strategy", names, index=idx, key="ref_strategy_select")
            info = strategy_catalog[selected]
            strategy_cfg = info.get("strategy_config", {}) if isinstance(info.get("strategy_config"), dict) else {}
            entry_cfg = strategy_cfg.get("entry", {}) if isinstance(strategy_cfg, dict) else {}
            exit_cfg = strategy_cfg.get("exit", {}) if isinstance(strategy_cfg, dict) else {}
            sizing_cfg = strategy_cfg.get("sizing", {}) if isinstance(strategy_cfg, dict) else {}
            limited_cfg = strategy_cfg.get("limited_test", {}) if isinstance(strategy_cfg, dict) else {}
            entry_defaults = _strategy_default_entry_params(strategy_cfg)
            exit_defaults = _strategy_default_exit_params(strategy_cfg)
            strategy_runs = [
                row
                for row in strategy_run_rows
                if str(row.get("strategy", "")).strip() == str(selected).strip()
            ]

            top_cols = st.columns(4)
            top_cols[0].markdown(f"**Module**\n\n`{info['module']}`")
            top_cols[1].markdown(f"**File**\n\n`{info['file']}`")
            top_cols[2].markdown(f"**Entry**\n\n`{', '.join(str(rule.get('name', '')) for rule in entry_cfg.get('rules', []) if isinstance(rule, dict)) or 'n/a'}`")
            top_cols[3].markdown(f"**Exit**\n\n`{exit_cfg.get('name', 'n/a')}`")
            if info.get("error"):
                st.error(f"Import error: {info['error']}")
            else:
                st.subheader("Strategy Logic")
                logic_cols = st.columns(3)
                with logic_cols[0]:
                    extra_lines = []
                    if entry_cfg.get("mode"):
                        extra_lines.append(f"Mode: `{entry_cfg.get('mode')}`")
                    if entry_cfg.get("vote_k") is not None:
                        extra_lines.append(f"Vote K: `{entry_cfg.get('vote_k')}`")
                    entry_rules = entry_cfg.get("rules", []) if isinstance(entry_cfg, dict) else []
                    if isinstance(entry_rules, list) and len(entry_rules) > 1:
                        extra_lines.append(f"Rules: `{len(entry_rules)}`")
                    primary_entry = entry_rules[0] if isinstance(entry_rules, list) and entry_rules and isinstance(entry_rules[0], dict) else {}
                    _render_logic_block(
                        "Entry",
                        name=str(primary_entry.get("name", "")),
                        params=dict(primary_entry.get("params", {}) or {}),
                        ref_type="entry",
                        extra_lines=extra_lines,
                        json_pretty=json_pretty,
                        build_query_url=build_query_url,
                    )
                with logic_cols[1]:
                    _render_logic_block(
                        "Exit",
                        name=str(exit_cfg.get("name", "")),
                        params=dict(exit_cfg.get("params", {}) or {}),
                        ref_type="exit",
                        json_pretty=json_pretty,
                        build_query_url=build_query_url,
                    )
                with logic_cols[2]:
                    _render_logic_block(
                        "Sizing",
                        name=str(sizing_cfg.get("name", "")),
                        params=dict(sizing_cfg.get("params", {}) or {}),
                        ref_type="sizing",
                        json_pretty=json_pretty,
                        build_query_url=build_query_url,
                    )

                entry_rules = entry_cfg.get("rules", []) if isinstance(entry_cfg, dict) else []
                if isinstance(entry_rules, list) and len(entry_rules) > 1:
                    st.markdown("**Additional Entry Rules**")
                    for idx_rule, rule in enumerate(entry_rules[1:], start=2):
                        if not isinstance(rule, dict):
                            continue
                        with st.expander(f"Rule {idx_rule}: {rule.get('name', 'unnamed')}"):
                            st.code(json_pretty(dict(rule.get("params", {}) or {})), language="json")

                st.subheader("Limited-Test Parameter Policy")
                policy_tabs = st.tabs(["Entry Params", "Exit Params", "Strategy Defaults", "Raw Strategy Config"])
                with policy_tabs[0]:
                    entry_section = limited_cfg.get("entry", {}) if isinstance(limited_cfg, dict) else {}
                    opt_rows = _optimizable_param_rows(entry_section, entry_defaults)
                    fixed_rows = _fixed_param_rows(entry_section, entry_defaults)
                    st.markdown("**Optimisable entry params**")
                    if opt_rows:
                        st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No optimisable entry params configured.")
                    st.markdown("**Non-optimisable entry params**")
                    if fixed_rows:
                        st.dataframe(pd.DataFrame(fixed_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No non-optimisable entry params configured.")

                with policy_tabs[1]:
                    exit_section = limited_cfg.get("exit", {}) if isinstance(limited_cfg, dict) else {}
                    opt_rows = _optimizable_param_rows(exit_section, exit_defaults)
                    fixed_rows = _fixed_param_rows(exit_section, exit_defaults)
                    st.markdown("**Optimisable exit params**")
                    if opt_rows:
                        st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No optimisable exit params configured.")
                    st.markdown("**Non-optimisable exit params**")
                    if fixed_rows:
                        st.dataframe(pd.DataFrame(fixed_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No non-optimisable exit params configured.")

                with policy_tabs[2]:
                    c1, c2 = st.columns(2)
                    c1.markdown("**Params Dataclass Defaults**")
                    c1.code(json_pretty(info.get("params_defaults", {})), language="json")
                    c2.markdown("**Default PARAM_SPACE**")
                    c2.code(json_pretty(info.get("param_space", {})), language="json")
                    c3, c4 = st.columns(2)
                    c3.markdown("**Default Entry Params**")
                    c3.code(json_pretty(entry_defaults), language="json")
                    c4.markdown("**Default Exit Params**")
                    c4.code(json_pretty(exit_defaults), language="json")

                with policy_tabs[3]:
                    st.code(json_pretty(strategy_cfg), language="json")

                st.subheader("Runs For This Strategy")
                if not strategy_runs:
                    st.info("No runs discovered yet for this strategy.")
                else:
                    wf_counts: dict[str, int] = {}
                    for row in strategy_runs:
                        workflow = str(row.get("workflow", "")).strip().lower() or "unknown"
                        wf_counts[workflow] = wf_counts.get(workflow, 0) + 1
                    count_cols = st.columns(max(1, min(4, len(wf_counts))))
                    for idx_count, (workflow, count) in enumerate(sorted(wf_counts.items())):
                        count_cols[idx_count].metric(workflow.replace("_", " ").title(), int(count))

                    run_tabs = st.tabs(["All", "Limited", "Walk-forward", "Monte Carlo", "Optimize"])
                    with run_tabs[0]:
                        df = _run_history_frame(strategy_runs)
                        _render_run_history_table(df)
                    with run_tabs[1]:
                        df = _run_history_frame(strategy_runs, workflow="limited")
                        if df.empty:
                            st.info("No limited-test runs for this strategy.")
                        else:
                            _render_run_history_table(df)
                    with run_tabs[2]:
                        df = _run_history_frame(strategy_runs, workflow="walkforward")
                        if df.empty:
                            st.info("No walk-forward runs for this strategy.")
                        else:
                            _render_run_history_table(df)
                    with run_tabs[3]:
                        df = _run_history_frame(strategy_runs, workflow="monte_carlo")
                        if df.empty:
                            st.info("No Monte Carlo runs for this strategy.")
                        else:
                            _render_run_history_table(df)
                    with run_tabs[4]:
                        df = _run_history_frame(strategy_runs, workflow="optimize")
                        if df.empty:
                            st.info("No optimize runs for this strategy.")
                        else:
                            _render_run_history_table(df)

            with st.expander("View strategy source code"):
                st.code(info.get("source", ""), language="python")

    def _render_plugin_tab(kind: str, title: str) -> None:
        items = plugin_catalog.get(kind, {})
        names = sorted(items.keys())
        if not names:
            st.info(f"No {title.lower()} discovered.")
            return

        idx = names.index(req_name) if req_type == kind and req_name in names else 0
        selected = st.selectbox(title, names, index=idx, key=f"ref_{kind}_select")
        info = items[selected]

        st.markdown(f"**Plugin:** `{selected}`")
        st.markdown(f"**File:** `{info.get('file', '')}`")
        st.markdown(f"**Signature:** `{selected}{info.get('signature', '')}`")
        if info.get("doc"):
            st.markdown(info["doc"])

        with st.expander("View plugin source code"):
            st.code(info.get("source", ""), language="python")

    if section == "entry":
        _render_plugin_tab("entry", "Entry Plugin")

    if section == "exit":
        _render_plugin_tab("exit", "Exit Plugin")

    if section == "sizing":
        _render_plugin_tab("sizing", "Sizing Plugin")
