from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from datetime import date
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - fallback used when plotly is unavailable
    px = None

REPO_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = REPO_ROOT / "runs"
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantbt.experiments.walkforward.fitness import build_initial_review_report


def _resolve_script_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


SCRIPT_PYTHON = _resolve_script_python()


COMMON_WF_OBJECTIVES = [
    "return_on_account",
    "total_return_%",
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


def _parse_extra_args(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return shlex.split(raw)


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


@st.cache_data(show_spinner=False)
def _discover_strategy_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    strategies_dir = SRC_DIR / "quantbt" / "strategies"

    for py in sorted(strategies_dir.glob("*.py")):
        if py.name == "__init__.py":
            continue

        short = py.stem
        module_path = f"quantbt.strategies.{short}"
        entry: dict[str, Any] = {
            "module": module_path,
            "file": _rel(py),
            "param_space": {},
            "strategy_config": {},
            "params_defaults": {},
            "source": py.read_text(encoding="utf-8"),
            "error": None,
        }

        try:
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
        except Exception as e:
            entry["error"] = str(e)

        catalog[short] = entry

    return catalog


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


def _discover_walkforward_runs() -> list[Path]:
    root = RUNS_ROOT / "walkforward"
    if not root.exists():
        return []
    out = [p.parent for p in root.rglob("summary.json")]
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def _discover_mc_runs() -> list[Path]:
    root = RUNS_ROOT / "walkforward"
    if not root.exists():
        return []
    out = [p.parent for p in root.rglob("mc_summary.json")]
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


def _discover_limited_runs() -> list[Path]:
    root = RUNS_ROOT / "limited"
    if not root.exists():
        return []
    out = [p.parent for p in root.rglob("limited_results.csv")]
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


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


def _status_label(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _criteria_table(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, bool]:
    df = pd.DataFrame(rows)
    if df.empty:
        return df, False

    df["Status"] = df["Passed"].map(_status_label)
    overall = bool(df["Passed"].all())
    return df[["Criterion", "Target", "Actual", "Status"]], overall


def _render_histogram(series: pd.Series, *, title: str, bins: int = 60) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        st.info(f"No numeric data for: {title}")
        return

    if px is not None:
        fig = px.histogram(x=s, nbins=bins, title=title)
        st.plotly_chart(fig, use_container_width=True)
        return

    bucketed = pd.cut(s, bins=bins, include_lowest=True)
    counts = bucketed.value_counts(sort=False)
    hist_df = pd.DataFrame({"bin": counts.index.astype(str), "count": counts.values}).set_index("bin")
    st.markdown(f"**{title}**")
    st.bar_chart(hist_df)


def _safe_key_from_path(prefix: str, path: Path) -> str:
    raw = f"{prefix}_{_rel(path)}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw)


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
        out_file = run_path / "oos_equity_interactive.html"
    elif run_kind == "monte_carlo":
        cmd = [SCRIPT_PYTHON, str(SCRIPTS_DIR / "plot_monte_carlo.py"), "--mc-run-dir", run_path.as_posix()]
        out_file = run_path / "mc_interactive.html"
    elif run_kind == "limited":
        cmd = [SCRIPT_PYTHON, str(SCRIPTS_DIR / "plot_limited_results.py"), "--run-dir", run_path.as_posix()]
        out_file = run_path / "limited_interactive.html"
    else:
        return False, f"Unsupported run_kind: {run_kind}"

    rc, out = _run_cli(cmd)
    if rc == 0 and out_file.exists():
        return True, _rel(out_file)
    return False, out.strip() or f"HTML generation failed for {run_kind}"


def _render_metric_row(metrics: list[tuple[str, Any, str]]) -> None:
    cols = st.columns(len(metrics)) if metrics else []
    for c, (label, value, fmt) in zip(cols, metrics):
        if isinstance(value, (int, float)) and fmt:
            try:
                c.metric(label, fmt.format(value))
                continue
            except Exception:
                pass
        c.metric(label, value)


def _render_walkforward_results(run_dir: Path) -> None:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    if not summary_path.exists() or not config_path.exists():
        st.error("Missing `summary.json` or `config.json` in selected walk-forward run.")
        return

    summary = _read_json(summary_path)
    config = _read_json(config_path)
    agg = summary.get("aggregated_oos_summary", {}) or {}
    objective = str(summary.get("objective", "return_on_account"))
    optimization_mode = str(summary.get("optimization_mode", config.get("optimization_mode", "peak")))
    selection_mode = str(summary.get("selection_mode", config.get("selection_mode", "peak")))
    wf_objective = _as_float(summary.get("comparison", {}).get("walkforward_objective"))
    baseline_objective = _as_float(summary.get("comparison", {}).get("baseline_objective"))
    wfe_summary = summary.get("wfe", {}) if isinstance(summary.get("wfe"), dict) else {}

    strategy_raw = str(config.get("strategy", ""))
    strategy_short = strategy_raw.rsplit(".", 1)[-1] if strategy_raw else ""

    st.subheader("Walk-Forward Result")
    if strategy_short:
        st.markdown(_reference_link(f"Strategy: {strategy_short}", "strategy", strategy_short))
        strategy_info = _discover_strategy_catalog().get(strategy_short, {})
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

    equity_path = run_dir / "oos_equity_curve.csv"
    trades_path = run_dir / "oos_trades.csv"

    eq = pd.read_csv(equity_path) if equity_path.exists() else pd.DataFrame()
    if "time" in eq.columns:
        eq["time"] = pd.to_datetime(eq["time"], errors="coerce")
    eq_for_metrics = eq
    if {"time", "equity"}.issubset(eq.columns):
        eq_for_metrics = eq.dropna(subset=["time"]).set_index("time")

    trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    initial_review = summary.get("initial_review", {})
    if not isinstance(initial_review, dict) or "metrics" not in initial_review:
        cfg_bt = config.get("backtest_config", {})
        initial_review = build_initial_review_report(
            equity_df=eq_for_metrics if isinstance(eq_for_metrics, pd.DataFrame) else pd.DataFrame(),
            trades_df=trades,
            aggregated_summary=agg,
            initial_equity=float(cfg_bt.get("initial_equity", 100000.0)),
            commission_per_round_trip=float(cfg_bt.get("commission_per_round_trip", 0.0)),
            spread_pips=float(cfg_bt.get("spread_pips", 0.0)),
        )

    ir_metrics = initial_review.get("metrics", {}) if isinstance(initial_review, dict) else {}
    ir_thresholds = initial_review.get("thresholds", {}) if isinstance(initial_review, dict) else {}
    ir_checks = initial_review.get("checks", {}) if isinstance(initial_review, dict) else {}
    ir_all_pass = bool(ir_checks.get("all_passed", False))

    _render_metric_row(
        [
            ("Trades", int(agg.get("trades", 0)), "{:d}"),
            ("Total Return %", _as_float(agg.get("total_return_%", 0.0)), "{:.2f}"),
            ("Max DD %", _as_float(agg.get("max_drawdown_abs_%", 0.0)), "{:.2f}"),
            ("Initial Review", _status_label(ir_all_pass), "{}"),
        ]
    )
    st.caption(f"Optimization mode: `{optimization_mode}` | Selection mode: `{selection_mode}`")

    def _money(v: Any) -> str:
        x = _as_float(v)
        return "n/a" if pd.isna(x) else f"${x:,.2f}"

    def _pct(v: Any) -> str:
        x = _as_float(v)
        return "n/a" if pd.isna(x) else f"{x:.2f}%"

    def _num(v: Any, nd: int = 3) -> str:
        x = _as_float(v)
        return "n/a" if pd.isna(x) else f"{x:.{nd}f}"

    def _int(v: Any) -> int:
        x = _as_float(v)
        return 0 if pd.isna(x) else int(x)

    max_top_trade_share_cfg = _as_float(config.get("max_top_trade_share", 1.0))
    top_trade_share_actual = _as_float(ir_metrics.get("top_trade_share", agg.get("top_trade_share")))

    st.markdown("**Initial Review (WFA)**")
    initial_rows: list[dict[str, Any]] = [
        {
            "Criterion": "Total net profit",
            "Target": f">= {_money(ir_thresholds.get('annualized_net_profit_abs_min'))}/year",
            "Actual": (
                f"{_money(ir_metrics.get('total_net_profit_abs'))} total | "
                f"{_money(ir_metrics.get('annualized_net_profit_abs'))}/year"
            ),
            "Passed": bool(ir_checks.get("annualized_net_profit_ok", False)),
        },
        {
            "Criterion": "Profit factor",
            "Target": (
                f">= {_num(ir_thresholds.get('profit_factor_min'), 2)} "
                f"(ideal >= {_num(ir_thresholds.get('profit_factor_ideal'), 2)})"
            ),
            "Actual": _num(ir_metrics.get("profit_factor"), 3),
            "Passed": bool(ir_checks.get("profit_factor_ok", False)),
        },
        {
            "Criterion": "Average trade net profit",
            "Target": f">= {_money(ir_thresholds.get('avg_trade_net_profit_abs_min'))}",
            "Actual": _money(ir_metrics.get("avg_trade_net_profit_abs")),
            "Passed": bool(ir_checks.get("avg_trade_net_profit_ok", False)),
        },
        {
            "Criterion": "Outlier concentration",
            "Target": f"top trade share <= {_num(max_top_trade_share_cfg, 2)}",
            "Actual": _num(top_trade_share_actual, 3),
            "Passed": bool(np.isfinite(top_trade_share_actual) and top_trade_share_actual <= max_top_trade_share_cfg),
        },
        {
            "Criterion": "Tharp expectancy",
            "Target": f">= {_num(ir_thresholds.get('tharp_expectancy_min'), 2)}",
            "Actual": _num(ir_metrics.get("tharp_expectancy"), 3),
            "Passed": bool(ir_checks.get("tharp_expectancy_ok", False)),
        },
        {
            "Criterion": "Slippage + commission realism",
            "Target": "non-zero costs (commission and spread)",
            "Actual": (
                f"commission/RT={_money(ir_metrics.get('commission_per_round_trip'))}, "
                f"spread={_num(ir_metrics.get('spread_pips'), 3)} pips"
            ),
            "Passed": bool(ir_checks.get("commission_model_ok", False) and ir_checks.get("spread_model_ok", False)),
        },
        {
            "Criterion": "Maximum drawdown vs net profit",
            "Target": f"DD/NetProfit < {_num(ir_thresholds.get('drawdown_to_net_profit_ratio_max'), 2)}",
            "Actual": (
                f"DD={_money(ir_metrics.get('max_drawdown_abs'))} "
                f"({_pct(ir_metrics.get('max_drawdown_abs_%'))}), "
                f"ratio={_num(ir_metrics.get('drawdown_to_net_profit_ratio'), 3)}"
            ),
            "Passed": bool(ir_checks.get("drawdown_vs_profit_ok", False)),
        },
        {
            "Criterion": "Equity curve slope",
            "Target": "positive slope (45-degree ideal)",
            "Actual": (
                f"slope/bar={_num(ir_metrics.get('equity_slope_per_bar'), 4)}, "
                f"angle={_num(ir_metrics.get('equity_slope_angle_deg'), 2)} deg"
            ),
            "Passed": bool(ir_checks.get("equity_slope_ok", False)),
        },
        {
            "Criterion": "Equity curve flat periods",
            "Target": f"flat bars ratio <= {_num(ir_thresholds.get('flat_bars_ratio_max'), 2)}",
            "Actual": (
                f"flat ratio={_num(ir_metrics.get('flat_bars_ratio'), 3)}, "
                f"longest flat run={_int(ir_metrics.get('longest_flat_run_bars'))} bars"
            ),
            "Passed": bool(ir_checks.get("equity_flat_period_ok", False)),
        },
        {
            "Criterion": "Equity drawdown depth + duration",
            "Target": (
                f"max DD duration ratio <= "
                f"{_num(ir_thresholds.get('max_drawdown_duration_ratio_max'), 2)}"
            ),
            "Actual": (
                f"max DD duration={_int(ir_metrics.get('max_drawdown_duration_bars'))} bars "
                f"(ratio={_num(ir_metrics.get('max_drawdown_duration_ratio'), 3)})"
            ),
            "Passed": bool(ir_checks.get("equity_drawdown_duration_ok", False)),
        },
        {
            "Criterion": "Equity curve fuzziness",
            "Target": (
                f"fuzziness <= {_num(ir_thresholds.get('equity_fuzziness_max'), 2)} "
                f"and R^2 >= {_num(ir_thresholds.get('equity_linearity_r2_min'), 2)}"
            ),
            "Actual": (
                f"fuzziness={_num(ir_metrics.get('equity_fuzziness'), 3)}, "
                f"R^2={_num(ir_metrics.get('equity_linearity_r2'), 3)}"
            ),
            "Passed": bool(ir_checks.get("equity_fuzziness_ok", False) and ir_checks.get("equity_linearity_ok", False)),
        },
    ]
    initial_df, initial_ok = _criteria_table(initial_rows)
    st.markdown(f"**Initial Review Status: {_status_label(initial_ok)}**")
    st.dataframe(initial_df, use_container_width=True, hide_index=True)

    is_bars = int(config.get("is_bars", 0))
    oos_bars = int(config.get("oos_bars", 0))
    ratio_pct = (100.0 * oos_bars / is_bars) if is_bars > 0 else float("nan")
    oos_under_min = int(summary.get("oos_under_min_trades_count", 0))
    fold_count = int(summary.get("fold_count", 0))
    wfe_metric = str(wfe_summary.get("metric", config.get("wfe_metric", "total_return_%")))
    wfe_min_pct = _as_float(wfe_summary.get("min_pct", config.get("wfe_min_pct", 0.0)))
    wfe_valid_count = int(wfe_summary.get("valid_fold_count", 0))
    wfe_pass_count = int(wfe_summary.get("pass_fold_count", 0))
    wfe_pass_rate = _as_float(wfe_summary.get("pass_rate_pct"))
    wfe_all_valid_passed = bool(wfe_summary.get("all_valid_passed", False))
    has_wfe = bool(wfe_summary)

    execution_rows: list[dict[str, Any]] = [
        {
            "Criterion": "Run status is ok",
            "Target": "status == ok",
            "Actual": str(summary.get("status")),
            "Passed": str(summary.get("status")) == "ok",
        },
        {
            "Criterion": "OOS/IS ratio constraint",
            "Target": "10% <= OOS/IS <= 50%",
            "Actual": f"{ratio_pct:.2f}%",
            "Passed": bool(10.0 <= ratio_pct <= 50.0),
        },
        {
            "Criterion": "Per-fold min trades",
            "Target": "all folds meet min trades",
            "Actual": f"{oos_under_min} failing fold(s)",
            "Passed": oos_under_min == 0,
        },
        {
            "Criterion": "Walk-forward efficiency",
            "Target": f"WFE({wfe_metric}) >= {wfe_min_pct:.2f}% on all valid folds",
            "Actual": (
                f"{wfe_pass_count}/{wfe_valid_count} valid folds pass ({_pct(wfe_pass_rate)})"
                if has_wfe
                else "not tracked in this run"
            ),
            "Passed": wfe_all_valid_passed if has_wfe else True,
        },
        {
            "Criterion": "Fold count",
            "Target": "> 0",
            "Actual": fold_count,
            "Passed": fold_count > 0,
        },
    ]
    execution_df, execution_ok = _criteria_table(execution_rows)
    st.markdown(f"**Execution Checks Status: {_status_label(execution_ok)}**")
    st.dataframe(execution_df, use_container_width=True, hide_index=True)

    if bool(summary.get("baseline_included")) and pd.notna(wf_objective) and pd.notna(baseline_objective):
        delta = float(wf_objective - baseline_objective)
        ratio = float(wf_objective / baseline_objective) if baseline_objective != 0 else float("nan")
        baseline_df = pd.DataFrame(
            [
                {
                    "Objective": objective,
                    "Walk-forward": wf_objective,
                    "Baseline": baseline_objective,
                    "Delta (WF - Baseline)": delta,
                    "WF / Baseline": ratio,
                }
            ]
        )
        st.caption(
            "Baseline comparison is informational only. Walk-forward is expected to be lower than a full-data baseline."
        )
        st.dataframe(baseline_df, use_container_width=True, hide_index=True)

    if {"time", "equity"}.issubset(eq.columns):
        st.subheader("OOS Equity Curve")
        if px is not None:
            fig = px.line(eq, x="time", y="equity", title="OOS Equity Curve")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(eq.set_index("time")["equity"])

    folds_path = run_dir / "folds.csv"
    if folds_path.exists():
        folds = pd.read_csv(folds_path)
        st.subheader("Fold Summary")
        if {"fold", "oos_total_return_%"}.issubset(folds.columns):
            if px is not None:
                fig = px.bar(folds, x="fold", y="oos_total_return_%", title="OOS Return % by Fold")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(folds.set_index("fold")["oos_total_return_%"])

        view_cols = [
            "fold",
            "oos_trades",
            "oos_total_return_%",
            "oos_max_drawdown_abs_%",
            "oos_return_on_account",
            "wfe_pct",
            "wfe_pass",
            "oos_min_trades_met",
        ]
        existing = [c for c in view_cols if c in folds.columns]
        st.dataframe(folds[existing] if existing else folds, use_container_width=True)

    html_path = run_dir / "oos_equity_interactive.html"
    if html_path.exists():
        _render_saved_interactive_html(html_path)


def _render_mc_results(mc_run_dir: Path) -> None:
    summary_path = mc_run_dir / "mc_summary.json"
    if not summary_path.exists():
        st.error("Missing `mc_summary.json` in selected Monte Carlo run.")
        return

    summary = _read_json(summary_path)
    metrics = summary.get("metrics", {})
    thresholds = summary.get("thresholds", {})
    checks = summary.get("threshold_checks", {})

    st.subheader("Monte Carlo Result")
    st.markdown("**Detailed Review (Monte Carlo)**")
    _render_metric_row(
        [
            ("Risk of Ruin %", _as_float(metrics.get("risk_of_ruin_pct", 0.0)), "{:.2f}"),
            ("Median Return %", _as_float(metrics.get("median_return_%", 0.0)), "{:.2f}"),
            ("Median Max DD %", _as_float(metrics.get("median_max_drawdown_%", 0.0)), "{:.2f}"),
            (
                "Return/DD Ratio",
                _as_float(metrics.get("return_drawdown_ratio_ratio_of_medians", 0.0)),
                "{:.3f}",
            ),
        ]
    )

    criteria_rows = [
        {
            "Criterion": "Risk of ruin",
            "Target": f"< {thresholds.get('risk_of_ruin_pct_max', 'n/a')}%",
            "Actual": f"{_as_float(metrics.get('risk_of_ruin_pct')):.2f}%",
            "Passed": bool(checks.get("risk_of_ruin_ok", False)),
        },
        {
            "Criterion": "Median max drawdown",
            "Target": f"< {thresholds.get('median_max_drawdown_pct_max', 'n/a')}%",
            "Actual": f"{_as_float(metrics.get('median_max_drawdown_%')):.2f}%",
            "Passed": bool(checks.get("median_max_drawdown_ok", False)),
        },
        {
            "Criterion": "Median return",
            "Target": f"> {thresholds.get('median_return_pct_min', 'n/a')}%",
            "Actual": f"{_as_float(metrics.get('median_return_%')):.2f}%",
            "Passed": bool(checks.get("median_return_ok", False)),
        },
        {
            "Criterion": "Return/Drawdown ratio",
            "Target": f"> {thresholds.get('return_drawdown_ratio_min', 'n/a')}",
            "Actual": f"{_as_float(metrics.get('return_drawdown_ratio_ratio_of_medians')):.3f}",
            "Passed": bool(checks.get("return_drawdown_ratio_ok", False)),
        },
    ]

    criteria_df, criteria_ok = _criteria_table(criteria_rows)
    st.markdown(f"**Overall Status: {_status_label(criteria_ok)}**")
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)

    sims_path = mc_run_dir / "mc_simulations.csv"
    if sims_path.exists():
        sims = pd.read_csv(sims_path)
        st.subheader("Distribution")
        if "return_%" in sims.columns:
            _render_histogram(sims["return_%"], title="Return % Distribution", bins=70)
        if "max_drawdown_%" in sims.columns:
            _render_histogram(sims["max_drawdown_%"], title="Max Drawdown % Distribution", bins=70)

    q_path = mc_run_dir / "mc_paths_quantiles.csv"
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

    html_path = mc_run_dir / "mc_interactive.html"
    if html_path.exists():
        _render_saved_interactive_html(html_path)


def _render_limited_results(run_dir: Path) -> None:
    results_path = run_dir / "limited_results.csv"
    trades_path = run_dir / "limited_trades.csv"
    pass_path = run_dir / "pass_summary.json"
    meta_path = run_dir / "run_meta.json"

    if not results_path.exists() or not pass_path.exists() or not meta_path.exists():
        st.error("Missing one of `limited_results.csv`, `pass_summary.json`, `run_meta.json`.")
        return

    results = pd.read_csv(results_path)
    pass_summary = _read_json(pass_path)
    run_meta = _read_json(meta_path)

    favourable_pct = _as_float(pass_summary.get("favourable_pct", 0.0))
    pass_threshold = _as_float(pass_summary.get("pass_threshold_%", 0.0))
    min_trades = int(pass_summary.get("min_trades", run_meta.get("min_trades", 0)))

    st.subheader("Limited Test Result")
    _render_metric_row(
        [
            ("Favourable %", favourable_pct, "{:.2f}"),
            ("Pass Threshold %", pass_threshold, "{:.2f}"),
            ("Iterations", int(pass_summary.get("total_iters", len(results))), "{:d}"),
            ("Passed", _status_label(bool(pass_summary.get("passed", False))), "{}"),
        ]
    )

    criteria_cfg = run_meta.get("criteria", {"mode": "all", "rules": []})
    mode = str(criteria_cfg.get("mode", "all"))
    rules = criteria_cfg.get("rules", [])

    criteria_rows: list[dict[str, Any]] = [
        {
            "Criterion": "Combined favourable rate",
            "Target": f">= {pass_threshold:.2f}%",
            "Actual": f"{favourable_pct:.2f}%",
            "Passed": favourable_pct >= pass_threshold,
        }
    ]

    if "trades" in results.columns:
        trades_met_pct = float((pd.to_numeric(results["trades"], errors="coerce") >= min_trades).mean() * 100.0)
        criteria_rows.append(
            {
                "Criterion": "Min trades constraint hit rate",
                "Target": f"trades >= {min_trades}",
                "Actual": f"{trades_met_pct:.2f}% iterations",
                "Passed": trades_met_pct > 0.0,
            }
        )

    for idx, rule in enumerate(rules, start=1):
        metric = str(rule.get("metric", ""))
        op = str(rule.get("op", ""))
        value = rule.get("value")

        if metric in results.columns:
            pass_mask = _evaluate_series_rule(results[metric], op, value)
            pass_pct = float(pass_mask.mean() * 100.0)
            actual_txt = f"{pass_pct:.2f}% iterations pass"
            passed = pass_pct >= pass_threshold
        else:
            actual_txt = "metric missing in limited_results.csv"
            passed = False

        criteria_rows.append(
            {
                "Criterion": f"Rule {idx} ({mode})",
                "Target": f"{metric} {op} {value}",
                "Actual": actual_txt,
                "Passed": passed,
            }
        )

    criteria_df, criteria_ok = _criteria_table(criteria_rows)
    st.markdown(f"**Overall Status: {_status_label(criteria_ok)}**")
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)

    st.subheader("Distribution")
    if "total_return_%" in results.columns:
        _render_histogram(results["total_return_%"], title="total_return_% distribution", bins=60)
    if "max_drawdown_abs_%" in results.columns:
        _render_histogram(results["max_drawdown_abs_%"], title="max_drawdown_abs_% distribution", bins=60)

    st.subheader("Iterations")
    iter_cols = [
        c
        for c in [
            "iter",
            "favourable",
            "trades",
            "total_return_%",
            "max_drawdown_abs_%",
            "profit_factor",
            "win_rate_%",
            "entry_params",
            "exit_params",
        ]
        if c in results.columns
    ]
    st.dataframe(results[iter_cols] if iter_cols else results, use_container_width=True)

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
                            ("Trades", trades_n, "{:d}"),
                            ("Return %", float(r.get("total_return_%", float("nan"))), "{:.2f}"),
                            ("Max DD %", float(r.get("max_drawdown_abs_%", float("nan"))), "{:.2f}"),
                        ]
                    )

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
                    st.dataframe(trades_iter[trade_cols] if trade_cols else trades_iter, use_container_width=True)
            else:
                st.info("No iteration ids available in results.")
        else:
            st.info("`limited_trades.csv` found, but missing `iter` column for drill-down.")
    else:
        st.info("Per-iteration trade details are not available for this run. Re-run limited tests to generate `limited_trades.csv`.")

    html_path = run_dir / "limited_interactive.html"
    if html_path.exists():
        _render_saved_interactive_html(html_path)


def _render_reference_page(strategy_catalog: dict[str, dict[str, Any]], plugin_catalog: dict[str, dict[str, dict[str, Any]]]) -> None:
    st.header("Strategy & Plugin Reference")

    req_type = _qp_get("ref_type", "strategy")
    req_name = _qp_get("ref_name", "")

    t_strategy, t_entry, t_exit, t_sizing = st.tabs(["Strategies", "Entry Plugins", "Exit Plugins", "Sizing Plugins"])

    with t_strategy:
        names = sorted(strategy_catalog.keys())
        if not names:
            st.info("No strategies discovered.")
        else:
            idx = names.index(req_name) if req_type == "strategy" and req_name in names else 0
            selected = st.selectbox("Strategy", names, index=idx, key="ref_strategy_select")
            info = strategy_catalog[selected]

            st.markdown(f"**Module:** `{info['module']}`")
            st.markdown(f"**File:** `{info['file']}`")
            if info.get("error"):
                st.error(f"Import error: {info['error']}")
            else:
                c1, c2 = st.columns(2)
                c1.markdown("**Default Param Dataclass Values**")
                c1.code(_json_pretty(info.get("params_defaults", {})), language="json")
                c2.markdown("**Default PARAM_SPACE**")
                c2.code(_json_pretty(info.get("param_space", {})), language="json")

                st.markdown("**Default STRATEGY Config**")
                st.code(_json_pretty(info.get("strategy_config", {})), language="json")

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

    with t_entry:
        _render_plugin_tab("entry", "Entry Plugin")

    with t_exit:
        _render_plugin_tab("exit", "Exit Plugin")

    with t_sizing:
        _render_plugin_tab("sizing", "Sizing Plugin")


def main() -> None:
    st.set_page_config(page_title="QuantBT Streamlit Frontend", layout="wide")
    st.title("QuantBT Streamlit Frontend")
    st.caption(f"Script interpreter: `{SCRIPT_PYTHON}`")

    if st.sidebar.button("Refresh discovered metadata"):
        st.cache_data.clear()
        st.rerun()

    for k in (
        "last_walkforward_run",
        "last_mc_run",
        "last_limited_run",
        "last_downloaded_dataset",
    ):
        st.session_state.setdefault(k, "")

    strategy_catalog = _discover_strategy_catalog()
    plugin_catalog = _discover_plugin_catalog()
    data_files = _list_csv_data_files()
    walk_runs = _discover_walkforward_runs()
    mc_runs = _discover_mc_runs()
    limited_runs = _discover_limited_runs()
    download_symbols = _discover_download_symbols()
    download_timeframes = _discover_download_timeframes()

    fallback_dataset = "data/processed/eurusd_1h_20100101_20260209_dukascopy_python.csv"
    if st.session_state["last_downloaded_dataset"]:
        default_dataset = _rel(_abs_path(st.session_state["last_downloaded_dataset"]))
    elif data_files:
        default_dataset = _rel(data_files[0])
    else:
        default_dataset = fallback_dataset

    page_options = {
        "run_tests": "Run Tests",
        "download_data": "Download Data",
        "results": "Results",
        "reference": "Reference",
    }
    requested_page = _qp_get("page", "run_tests")
    if requested_page == "html_view":
        _render_html_view_page()
        return
    keys = list(page_options.keys())
    if requested_page not in keys:
        requested_page = "run_tests"

    nav_key = "nav_page"
    last_qp_key = "_last_nav_qp_page"

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

    if page == "run_tests":
        st.header("Run Tests")
        t_wf, t_mc, t_limited = st.tabs(["Walk-forward", "Monte Carlo", "Limited tests"])

        with t_wf:
            strategy_names = sorted(strategy_catalog.keys())
            default_strategy_idx = strategy_names.index("sma_cross_test_strat") if "sma_cross_test_strat" in strategy_names else 0
            strategy = st.selectbox("Strategy", strategy_names, index=default_strategy_idx, key="wf_strategy")
            info = strategy_catalog.get(strategy, {})
            st.markdown(_reference_link("View strategy config", "strategy", strategy))

            dataset_options = [_rel(p) for p in data_files]
            dataset = _select_path_from_options("Dataset CSV", dataset_options, default_dataset, key_prefix="wf_dataset")

            c1, c2, c3, c4 = st.columns(4)
            optimizer = c1.selectbox("Optimizer", ("grid", "optuna"))
            optimization_mode = c2.selectbox(
                "Optimization mode",
                ("peak", "stability_robustness"),
                help="`stability_robustness` emphasizes parameter stability, WFE, and anti-outlier filters.",
            )
            selection_mode = c3.selectbox("Selection mode", ("peak", "plateau"))
            direction = c4.selectbox("Direction", ("maximize", "minimize"))

            c5, c6 = st.columns(2)
            objective_choice = c5.selectbox("Objective", [*COMMON_WF_OBJECTIVES, "profit_factor", "custom..."])
            if objective_choice == "custom...":
                objective = c5.text_input("Custom objective", value="return_on_account", key="wf_custom_objective")
            else:
                objective = objective_choice

            wfe_metric_choice = c6.selectbox(
                "WFE metric",
                [*COMMON_WF_OBJECTIVES, "profit_factor", "custom..."],
                index=1 if "total_return_%" in COMMON_WF_OBJECTIVES else 0,
            )
            if wfe_metric_choice == "custom...":
                wfe_metric = c6.text_input("Custom WFE metric", value="total_return_%", key="wf_custom_wfe_metric")
            else:
                wfe_metric = wfe_metric_choice

            c7, c8, c9 = st.columns(3)
            is_bars = int(c7.number_input("IS bars", min_value=1, value=18000, step=100))
            oos_bars = int(c8.number_input("OOS bars", min_value=1, value=6000, step=100))
            step_bars = int(c9.number_input("Step bars (0 => default)", min_value=0, value=6000, step=100))

            c10, c11, c12 = st.columns(3)
            min_trades = int(c10.number_input("Min trades (fallback)", min_value=0, value=30, step=1, key="wf_min_trades"))
            min_is_trades = int(c11.number_input("Min IS trades", min_value=0, value=30, step=1, key="wf_min_is_trades"))
            min_oos_trades = int(c12.number_input("Min OOS trades", min_value=0, value=30, step=1, key="wf_min_oos_trades"))

            c13, c14, c15 = st.columns(3)
            wfe_min_pct = float(c13.number_input("WFE min %", min_value=0.0, value=0.0, step=1.0, format="%.1f"))
            max_top_trade_share = float(
                c14.number_input("Max top trade share", min_value=0.01, max_value=1.0, value=1.0, step=0.01, format="%.2f")
            )
            plateau_min_neighbors = int(c15.number_input("Plateau min neighbors", min_value=0, value=3, step=1))

            c16, c17, c18 = st.columns(3)
            plateau_stability_penalty = float(
                c16.number_input("Plateau stability penalty", min_value=0.0, value=0.5, step=0.1, format="%.2f")
            )
            progress_every = int(c17.number_input("Progress every", min_value=1, value=20, step=1, key="wf_progress_every"))
            run_base = c18.text_input("Run base", value="runs")

            c19, c20, c21 = st.columns(3)
            n_trials = int(c19.number_input("Optuna n-trials", min_value=1, value=200, step=1))
            timeout_s = int(c20.number_input("Optuna timeout (sec)", min_value=0, value=0, step=10))
            sampler = c21.selectbox("Sampler", ("tpe", "random"))

            c22, c23, c24 = st.columns(3)
            seed = int(c22.number_input("Seed", min_value=0, value=42, step=1, key="wf_seed"))
            anchored_mode = c23.selectbox("Window mode", ("Anchored", "Unanchored"))
            ts_col = c24.text_input("Timestamp column", value="timestamp")

            c25, c26, c27 = st.columns(3)
            initial_equity = float(c25.number_input("Initial equity", min_value=0.0, value=100000.0, step=1000.0))
            risk_pct = float(c26.number_input("Risk %", min_value=0.0, value=0.01, step=0.001, format="%.4f"))
            spread_pips = float(c27.number_input("Spread pips", min_value=0.0, value=0.2, step=0.1))

            c28, c29, c30 = st.columns(3)
            pip_size = float(c28.number_input("Pip size", min_value=0.0, value=0.0001, format="%.6f"))
            margin_rate = float(c29.number_input("Margin rate", min_value=0.0, value=0.0, step=0.001, format="%.4f"))
            lot_size = float(c30.number_input("Lot size", min_value=1.0, value=100000.0, step=1000.0))

            c31, c32, c33 = st.columns(3)
            commission_rt = float(c31.number_input("Commission RT", value=5.0, step=0.5))
            required_margin_abs = c32.text_input("Required margin abs (optional)", value="")
            extra_args = c33.text_input("Extra args (optional)", value="", key="wf_extra_args")

            c34, c35, c36 = st.columns(3)
            baseline_full_data = c34.checkbox("Run baseline full-data compare", value=True)
            compound_oos = c35.checkbox("Compound OOS equity", value=True)
            conservative_same_bar = c36.checkbox("Conservative same-bar handling", value=False)
            if optimization_mode == "stability_robustness":
                st.caption(
                    "stability_robustness mode enforces stricter floors: "
                    "min IS/OOS trades >= 50, WFE >= 50%, top-trade-share <= 0.30."
                )

            default_param_space = info.get("param_space", {})
            use_param_space_override = st.checkbox("Use PARAM_SPACE override (--param-space)", value=False)
            param_space_text = st.text_area(
                "Param space JSON (prepopulated from strategy)",
                value=_json_pretty(default_param_space),
                height=160,
                key=f"wf_param_space_{strategy}",
            )

            if st.button("Run walk-forward", type="primary"):
                try:
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
                    if anchored_mode == "Unanchored":
                        cmd.append("--unanchored")
                    if step_bars > 0:
                        cmd.extend(["--step-bars", str(step_bars)])
                    if required_margin_abs.strip():
                        cmd.extend(["--required-margin-abs", required_margin_abs.strip()])
                    if use_param_space_override:
                        cmd.extend(["--param-space", param_space_text.strip()])
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

        with t_mc:
            walk_run_options = [p.as_posix() for p in walk_runs]
            default_walk_run = st.session_state["last_walkforward_run"] or (walk_run_options[0] if walk_run_options else "runs/walkforward/...")
            run_dir = _select_path_from_options(
                "Walk-forward run directory",
                walk_run_options,
                default_walk_run,
                key_prefix="mc_run_dir",
            )

            c1, c2, c3 = st.columns(3)
            n_sims = int(c1.number_input("Number of simulations", min_value=1, value=8000, step=100))
            ruin_equity = float(c2.number_input("Ruin equity", min_value=0.0, value=70000.0, step=500.0))
            seed = int(c3.number_input("Seed", min_value=0, value=42, step=1, key="mc_seed"))

            c4, c5, c6 = st.columns(3)
            n_trades = c4.text_input("Trades per simulation (optional)", value="")
            sample_with_replacement = c5.checkbox("Sample with replacement", value=True)
            stop_at_ruin = c6.checkbox("Stop path at ruin", value=True)

            c7, c8, c9 = st.columns(3)
            pnl_mode = c7.selectbox("PnL mode", ("actual", "fixed_risk"))
            fixed_risk_dollars = c8.text_input("Fixed risk dollars (required if fixed_risk)", value="")
            progress_every = int(c9.number_input("Progress every", min_value=1, value=200, step=10, key="mc_progress_every"))

            c10, c11, c12 = st.columns(3)
            sample_paths = int(c10.number_input("Saved sample paths", min_value=0, value=120, step=10))
            save_quantiles = c11.checkbox("Save quantile paths", value=True)
            extra_args = c12.text_input("Extra args (optional)", value="", key="mc_extra_args")

            c13, c14, c15, c16 = st.columns(4)
            tor_max = float(c13.number_input("Max risk of ruin %", value=10.0, step=0.5))
            mdd_max = float(c14.number_input("Max median DD %", value=40.0, step=0.5))
            ret_min = float(c15.number_input("Min median return %", value=40.0, step=0.5))
            ratio_min = float(c16.number_input("Min return/DD ratio", value=2.0, step=0.1))

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
                    if n_trades.strip():
                        cmd.extend(["--n-trades", n_trades.strip()])
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

        with t_limited:
            strategy_names = sorted(strategy_catalog.keys())
            default_strategy_idx = strategy_names.index("sma_cross_test_strat") if "sma_cross_test_strat" in strategy_names else 0
            strategy_short = st.selectbox("Strategy", strategy_names, index=default_strategy_idx, key="limited_strategy")
            strategy_module = f"quantbt.strategies.{strategy_short}"
            strategy_info = strategy_catalog.get(strategy_short, {})
            strategy_cfg = strategy_info.get("strategy_config", {})

            st.markdown(_reference_link("View strategy config", "strategy", strategy_short))

            dataset_options = [_rel(p) for p in data_files]
            data_path = _select_path_from_options(
                "Dataset CSV",
                dataset_options,
                default_dataset,
                key_prefix="limited_dataset",
            )

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

            preset_defs: dict[str, dict[str, Any]] = {
                "Core": {
                    "entry_plugin": default_entry,
                    "entry_params": _strategy_default_entry_params(),
                    "exit_plugin": default_exit,
                    "exit_params": _strategy_default_exit_params(),
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
                    "entry_plugin": "sma_cross",
                    "entry_params": {"fast": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "slow": [100, 125, 150, 175, 200, 225, 250, 275, 300, 325]},
                    "exit_plugin": "atr_brackets",
                    "exit_params": {"rr": 2.0, "sldist_atr_mult": 1.5, "atr_period": 14},
                    "seed_count": "",
                    "seed_start": "",
                    "exit_seed_count": "",
                    "exit_seed_start": "",
                    "favourable_criteria": "",
                    "pass_threshold": "",
                },
                "Fixed Bar Exit": {
                    "entry_plugin": "sma_cross",
                    "entry_params": {"fast": [20, 30, 40, 50, 60, 70, 80], "slow": [125, 150, 175, 200, 225, 250, 275, 300, 325, 350]},
                    "exit_plugin": "time_exit",
                    "exit_params": {"hold_bars": [1]},
                    "seed_count": "",
                    "seed_start": "",
                    "exit_seed_count": "",
                    "exit_seed_start": "",
                    "favourable_criteria": "",
                    "pass_threshold": "",
                },
                "Similar Entry": {
                    "entry_plugin": "donchian_breakout",
                    "entry_params": {"lookback": [20]},
                    "exit_plugin": "atr_brackets",
                    "exit_params": {"rr": [1.0, 1.5, 2.0, 2.5, 3.0], "sldist_atr_mult": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], "atr_period": 14},
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
                    "exit_params": _strategy_default_exit_params(),
                    "seed_count": "8000",
                    "seed_start": "1",
                    "exit_seed_count": "",
                    "exit_seed_start": "",
                    "favourable_criteria": json.dumps(monkey_criteria, separators=(",", ":")),
                    "pass_threshold": "90",
                },
                "Monkey Exit": {
                    "entry_plugin": default_entry,
                    "entry_params": _strategy_default_entry_params(),
                    "exit_plugin": "monkey_exit",
                    "exit_params": {"avg_hold_bars": 15.75},
                    "seed_count": "",
                    "seed_start": "",
                    "exit_seed_count": "8000",
                    "exit_seed_start": "1",
                    "favourable_criteria": json.dumps(monkey_criteria, separators=(",", ":")),
                    "pass_threshold": "90",
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
                },
            }

            preset_label = st.selectbox("Limited test template", list(preset_defs.keys()), key="limited_test_template")

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
                st.session_state.setdefault("limited_commission_rt", "5")

            preset_track_key = "_limited_last_applied_template"
            strategy_track_key = "_limited_last_template_strategy"
            template_changed = st.session_state.get(preset_track_key) != preset_label
            strategy_changed = st.session_state.get(strategy_track_key) != strategy_short
            if template_changed or strategy_changed:
                _apply_limited_template(preset_label)
                st.session_state[preset_track_key] = preset_label
                st.session_state[strategy_track_key] = strategy_short

            st.session_state.setdefault("limited_entry_plugin_select", default_entry)
            st.session_state.setdefault("limited_exit_plugin_select", default_exit)
            st.session_state.setdefault("limited_sizing_plugin_select", default_sizing)
            st.session_state.setdefault("limited_entry_params_json", _json_pretty(_strategy_default_entry_params()))
            st.session_state.setdefault("limited_exit_params_json", _json_pretty(_strategy_default_exit_params()))
            st.session_state.setdefault("limited_sizing_params_json", _json_pretty(_strategy_default_sizing_params()))
            st.session_state.setdefault("limited_run_base", "")
            st.session_state.setdefault("limited_test_name", "")
            st.session_state.setdefault("limited_progress_every", 10)
            st.session_state.setdefault("limited_commission_rt", "5")
            st.session_state.setdefault("limited_pass_threshold", "")
            st.session_state.setdefault("limited_min_trades", "")
            st.session_state.setdefault("limited_favourable_criteria", "")
            st.session_state.setdefault("limited_extra_args", "")
            st.session_state.setdefault("limited_seed_count", "")
            st.session_state.setdefault("limited_seed_start", "")
            st.session_state.setdefault("limited_exit_seed_count", "")
            st.session_state.setdefault("limited_exit_seed_start", "")

            c1, c2, c3 = st.columns(3)
            entry_plugin = c1.selectbox("Entry plugin", [default_entry, *entry_plugins], key="limited_entry_plugin_select")
            exit_plugin = c2.selectbox("Exit plugin", [default_exit, *exit_plugins], key="limited_exit_plugin_select")
            sizing_plugin = c3.selectbox("Sizing plugin", [default_sizing, *sizing_plugins], key="limited_sizing_plugin_select")

            selected_entry_name = entry_default_name if entry_plugin == default_entry else entry_plugin
            selected_exit_name = exit_default_name if exit_plugin == default_exit else exit_plugin
            selected_sizing_name = sizing_default_name if sizing_plugin == default_sizing else sizing_plugin

            link_cols = st.columns(3)
            if selected_entry_name:
                link_cols[0].markdown(_reference_link("View entry plugin", "entry", str(selected_entry_name)))
            if selected_exit_name:
                link_cols[1].markdown(_reference_link("View exit plugin", "exit", str(selected_exit_name)))
            if selected_sizing_name:
                link_cols[2].markdown(_reference_link("View sizing plugin", "sizing", str(selected_sizing_name)))

            c4, c5 = st.columns(2)
            entry_params = c4.text_area("Entry params JSON", height=140, key="limited_entry_params_json")
            exit_params = c5.text_area("Exit params JSON", height=140, key="limited_exit_params_json")

            sizing_params = st.text_area("Sizing params JSON", height=110, key="limited_sizing_params_json")

            c6, c7, c8 = st.columns(3)
            run_base = c6.text_input("Run base (optional)", key="limited_run_base")
            test_name = c7.text_input("Test name (optional)", key="limited_test_name")
            progress_every = int(c8.number_input("Progress every", min_value=1, step=1, key="limited_progress_every"))

            c9, c10, c11 = st.columns(3)
            commission_rt = c9.text_input("Commission RT (optional)", key="limited_commission_rt")
            pass_threshold = c10.text_input("Pass threshold % (optional)", key="limited_pass_threshold")
            min_trades = c11.text_input("Min trades (optional)", key="limited_min_trades")

            c12, c13 = st.columns(2)
            favourable_criteria = c12.text_input("Favourable criteria JSON (optional)", key="limited_favourable_criteria")
            extra_args = c13.text_input("Extra args (optional)", key="limited_extra_args")

            c14, c15, c16, c17 = st.columns(4)
            seed_count = c14.text_input("Seed count (optional)", key="limited_seed_count")
            seed_start = c15.text_input("Seed start (optional)", key="limited_seed_start")
            exit_seed_count = c16.text_input("Exit seed count (optional)", key="limited_exit_seed_count")
            exit_seed_start = c17.text_input("Exit seed start (optional)", key="limited_exit_seed_start")

            if st.button("Run limited tests", type="primary"):
                try:
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
                    seed_count_raw = seed_count.strip()
                    seed_start_raw = seed_start.strip()
                    exit_seed_count_raw = exit_seed_count.strip()
                    exit_seed_start_raw = exit_seed_start.strip()

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

                    if commission_rt.strip():
                        cmd.extend(["--commission-rt", commission_rt.strip()])
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

                    cmd.extend(_parse_extra_args(extra_args))
                except ValueError as e:
                    st.error(f"Invalid arguments: {e}")
                else:
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

    elif page == "download_data":
        st.header("Download Data")

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

        c_date_1, c_date_2 = st.columns(2)
        start_date = c_date_1.date_input("Start date", value=date(2010, 1, 1), key="download_start_date")
        end_date = c_date_2.date_input("End date", value=date.today(), key="download_end_date")

        if provider == "dukascopy":
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

    elif page == "results":
        st.header("Results")
        t_wf, t_mc, t_limited = st.tabs(["Walk-forward", "Monte Carlo", "Limited tests"])

        with t_wf:
            options = [p.as_posix() for p in walk_runs]
            default = st.session_state["last_walkforward_run"] or (options[0] if options else "")
            run_dir_raw = _select_path_from_options("Walk-forward run", options, default, key_prefix="res_wf")
            if run_dir_raw.strip():
                _render_walkforward_results(_abs_path(run_dir_raw.strip()))
            else:
                st.info("No walk-forward run selected.")

        with t_mc:
            options = [p.as_posix() for p in mc_runs]
            default = st.session_state["last_mc_run"] or (options[0] if options else "")
            run_dir_raw = _select_path_from_options("Monte Carlo run", options, default, key_prefix="res_mc")
            if run_dir_raw.strip():
                _render_mc_results(_abs_path(run_dir_raw.strip()))
            else:
                st.info("No Monte Carlo run selected.")

        with t_limited:
            options = [p.as_posix() for p in limited_runs]
            default = st.session_state["last_limited_run"] or (options[0] if options else "")
            run_dir_raw = _select_path_from_options("Limited run", options, default, key_prefix="res_limited")
            if run_dir_raw.strip():
                _render_limited_results(_abs_path(run_dir_raw.strip()))
            else:
                st.info("No limited run selected.")

    else:
        _render_reference_page(strategy_catalog, plugin_catalog)


if __name__ == "__main__":
    main()
