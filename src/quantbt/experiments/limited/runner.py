from __future__ import annotations

from collections import OrderedDict
import itertools
import json
import math
import time
from pathlib import Path

import pandas as pd

from quantbt.artifacts import (
    infer_limited_scenario_slug,
    limited_iterations_path,
    limited_trades_path,
    spec_path,
    summary_path,
    tables_dir,
    write_manifest,
)
from quantbt.core.engine import BacktestConfig
from quantbt.core.engine_limited import run_backtest_limited
from quantbt.io.datasets import read_dataset_meta, dataset_tag_for_runs
from quantbt.plugins import get_exit, get_sizing, load_default_plugins
from quantbt.plugins.combiner import combine_signals

from .base import limited_test_pass_rate
from .criteria import criteria_pass, parse_favourable_criteria
from .data_prep import build_signal_frame, compute_atr, iter_entries_from_signals, load_price_frame
from .monkey import (
    MONKEY_ENTRY_PLUGIN_NAMES,
    MONKEY_TIME_EXIT_PLUGIN_NAMES,
    build_exact_monkey_entries_for_time_exit,
    build_precomputed_time_exit_wrapper,
    load_monkey_davey_cfg,
    load_monkey_match_prefilter_cfg,
    load_monkey_runtime_cfg,
    prefilter_schedule_matches,
    run_backtest_limited_time_exit_fast_summary,
    simulate_flat_only_time_exit_schedule,
    time_exit_prefilter_is_supported,
    wilson_interval,
)
from .param_grid import build_entry_variants, build_exit_param_space, total_iterations
from .progress import print_progress
from .runlog import make_limited_run_dir, write_json
from .types import EntryEvent, ScheduleMetrics


DEFAULT_SIGNAL_CACHE_MAX = 128


def _load_signal_cache_max(spec: dict) -> int:
    test_cfg = spec.get("test", {}) if isinstance(spec, dict) else {}
    raw = test_cfg.get("signal_cache_max")
    if raw is None:
        return int(DEFAULT_SIGNAL_CACHE_MAX)
    try:
        val = int(raw)
    except Exception:
        return int(DEFAULT_SIGNAL_CACHE_MAX)
    return max(0, val)


def _get_full_system_runner(
    *,
    test_focus: str,
    entry_mode: str,
    entry_combo: tuple[dict, ...],
):
    if str(test_focus).strip() != "entry_test":
        return None
    if str(entry_mode).strip() != "all":
        return None
    if len(entry_combo) != 1 or not isinstance(entry_combo[0], dict):
        return None
    plugin = entry_combo[0].get("plugin")
    runner = getattr(plugin, "run_full_system", None)
    return runner if callable(runner) else None


def run_spec(spec: dict, *, progress_every: int = 10):
    load_default_plugins()

    if "data" not in spec:
        raise ValueError("spec must include 'data'")

    data_path = Path(spec["data"])
    ts_col = spec.get("ts_col", "timestamp")

    df = load_price_frame(data_path, ts_col=ts_col)

    cfg_dict = spec.get("config", {})
    cfg = BacktestConfig(**cfg_dict)

    strat = spec["strategy"]
    entry_spec = strat["entry"]
    exit_spec = strat["exit"]
    sizing_spec = strat.get("sizing", {"name": "fixed_risk", "params": {}})

    entry_mode = entry_spec.get("mode", "all")
    vote_k = entry_spec.get("vote_k")
    rules = entry_spec.get("rules", [])

    entry_variants, skipped = build_entry_variants(rules)
    if skipped:
        print(f"skipped {len(skipped)} invalid entry param sets")

    exit_plugin = get_exit(exit_spec["name"])
    exit_param_space = build_exit_param_space(exit_spec)

    sizing_plugin = get_sizing(sizing_spec.get("name", "fixed_risk"))
    sizing_params = sizing_spec.get("params", {})

    criteria = parse_favourable_criteria(spec.get("test", {}).get("favourable_criteria"))
    min_trades = int(spec.get("test", {}).get("min_trades", 30))
    pass_threshold = float(spec.get("test", {}).get("pass_threshold_pct", 70.0))
    monkey_davey_cfg = load_monkey_davey_cfg(spec, criteria=criteria, pass_threshold=pass_threshold)
    monkey_davey_enabled = bool(monkey_davey_cfg.get("enabled", False))
    if monkey_davey_enabled:
        pass_threshold = float(monkey_davey_cfg.get("pass_threshold_%", pass_threshold))
        print(
            "[INFO] strict Davey monkey scoring enabled "
            f"(baseline return={monkey_davey_cfg['baseline_return_%']:.6g}%, "
            f"baseline maxDD={monkey_davey_cfg['baseline_max_dd_%']:.6g}%, "
            f"dd metric={monkey_davey_cfg['dd_metric']}, "
            f"pass threshold={pass_threshold:.2f}%).",
            flush=True,
        )
    elif monkey_davey_cfg.get("error"):
        print(f"[INFO] monkey_davey disabled: {monkey_davey_cfg['error']}", flush=True)

    dataset_meta = read_dataset_meta(data_path)
    dataset_tag = dataset_tag_for_runs(data_path, dataset_meta)
    strategy_tag = spec.get("test", {}).get("strategy_tag", entry_spec.get("tag", "spec"))
    scenario_slug = infer_limited_scenario_slug(
        strat,
        test_focus=str(spec.get("test", {}).get("test_focus", "")).strip(),
        test_name=str(spec.get("test", {}).get("test_name", "limited_test")),
    )

    run_dir = make_limited_run_dir(
        base=spec.get("test", {}).get("run_base", "runs"),
        strategy=strategy_tag,
        dataset_tag=dataset_tag,
        test_name=spec.get("test", {}).get("test_name", "limited_test"),
        scenario_slug=scenario_slug,
    )
    tables_dir(run_dir).mkdir(parents=True, exist_ok=True)

    run_meta = {
        "spec": spec,
        "dataset_meta": dataset_meta,
        "criteria": criteria,
        "pass_threshold_%": pass_threshold,
        "min_trades": min_trades,
    }
    if monkey_davey_enabled:
        run_meta["monkey_davey"] = monkey_davey_cfg
    write_json(spec_path(run_dir), run_meta)
    write_manifest(
        run_dir,
        workflow="limited",
        strategy=strategy_tag,
        category=scenario_slug,
        dataset_tag=dataset_tag,
        extras={
            "scenario_label": str(spec.get("test", {}).get("test_name", "limited_test")),
            "test_focus": str(spec.get("test", {}).get("test_focus", "")),
        },
    )

    requires_atr = bool(getattr(exit_plugin, "requires_atr", False))
    atr_period = int(exit_spec.get("params", {}).get("atr_period", 14))
    atr_series = compute_atr(df, atr_period) if requires_atr else None

    signal_cache_max = _load_signal_cache_max(spec)
    signals_cache: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
    test_focus = str(spec.get("test", {}).get("test_focus", "")).strip()

    save_trades = bool(spec.get("test", {}).get("save_trades", True))

    rows = []
    trade_rows: list[pd.DataFrame] | None = [] if save_trades else None
    total = total_iterations(entry_variants, exit_param_space)

    start_ts = time.time()
    iter_count = 0
    pass_count = 0
    attempt_count = 0
    prefilter_reject_count = 0
    early_stop_triggered = False
    early_stop_reason: str | None = None
    early_stop_ci_pct: tuple[float, float] | None = None
    early_stop_decision: str | None = None
    davey_return_worse_count = 0
    davey_dd_worse_count = 0

    monkey_prefilter_cfg = load_monkey_match_prefilter_cfg(spec)
    monkey_runtime_cfg = load_monkey_runtime_cfg(spec)
    monkey_fast_summary_requested = bool(monkey_runtime_cfg.get("fast_summary", False))
    monkey_seq_cfg = monkey_runtime_cfg.get("sequential_stop", {}) if isinstance(monkey_runtime_cfg, dict) else {}
    monkey_seq_enabled = bool(monkey_seq_cfg.get("enabled", False))
    if monkey_davey_enabled and monkey_seq_enabled:
        monkey_seq_enabled = False
        print("[INFO] monkey sequential stopping disabled in strict Davey mode (full run required).", flush=True)
    monkey_exact_scheduler_enabled = False
    prefilter_supported, prefilter_exit_supports_entry = time_exit_prefilter_is_supported(exit_spec["name"], exit_plugin)
    if monkey_prefilter_cfg and not prefilter_supported:
        print(
            "[INFO] monkey exact-match prefilter requested but disabled: "
            f"exit `{exit_spec['name']}` is not time-based (time_exit/monkey_exit/random_time_exit).",
            flush=True,
        )
        monkey_prefilter_cfg = None
    elif monkey_prefilter_cfg:
        print(
            "[INFO] monkey exact-match prefilter enabled "
            f"(target trades={monkey_prefilter_cfg['target_trades']:.2f}, "
            f"trade tol={monkey_prefilter_cfg['trade_tol_pct']:.2f}%, "
            f"long tol={monkey_prefilter_cfg['long_tol_pp']:.2f}pp, "
            f"hold tol={monkey_prefilter_cfg['hold_tol_pct']:.2f}%).",
            flush=True,
        )
        if str(exit_spec["name"]).strip() in MONKEY_TIME_EXIT_PLUGIN_NAMES:
            monkey_exact_scheduler_enabled = True
            print(
                "[INFO] exact monkey schedule mode active (flat-only generated schedule; "
                "trade count sampled within tolerance; per-trade hold bars randomized with baseline mean target).",
                flush=True,
            )
    monkey_fast_summary_active = bool(monkey_fast_summary_requested and prefilter_supported)
    if monkey_fast_summary_requested and not prefilter_supported:
        print("[INFO] monkey fast-summary requested but disabled: exit is not time-based.", flush=True)
    elif monkey_fast_summary_active:
        print(
            "[INFO] monkey fast-summary mode active (summary-only time-exit evaluator).",
            flush=True,
        )
    if monkey_seq_enabled:
        print(
            "[INFO] monkey sequential stopping enabled "
            f"(min accepted={int(monkey_seq_cfg.get('min_accepted', 1000))}, "
            f"check every={int(monkey_seq_cfg.get('check_every', 200))}, "
            f"fail<{float(monkey_seq_cfg.get('fail_threshold_pct', 75.0)):.1f}%, "
            f"pass>{pass_threshold:.1f}%, z={float(monkey_seq_cfg.get('z', 1.96)):.2f}).",
            flush=True,
        )
    if signal_cache_max <= 0:
        print("[INFO] signal cache disabled (signal_cache_max=0).", flush=True)
    elif signal_cache_max < DEFAULT_SIGNAL_CACHE_MAX:
        print(
            f"[INFO] signal cache bounded at {signal_cache_max} entries.",
            flush=True,
        )
    if not save_trades:
        print("[INFO] tables/trades.csv disabled (--no-save-trades).", flush=True)

    def size_fn(**kwargs):
        return sizing_plugin(**kwargs, params=sizing_params)

    for entry_combo in itertools.product(*entry_variants):
        full_system_runner = _get_full_system_runner(
            test_focus=test_focus,
            entry_mode=entry_mode,
            entry_combo=entry_combo,
        )
        if full_system_runner is None:
            signals_list = []
            for rule in entry_combo:
                key = (rule["name"], tuple(sorted(rule["params"].items())))
                sig = None
                if signal_cache_max > 0:
                    sig = signals_cache.get(key)
                    if sig is not None:
                        signals_cache.move_to_end(key)
                if sig is None:
                    sig = rule["plugin"](df, rule["params"])
                    if signal_cache_max > 0:
                        signals_cache[key] = sig
                        signals_cache.move_to_end(key)
                        if len(signals_cache) > signal_cache_max:
                            signals_cache.popitem(last=False)
                signals_list.append(sig)

            combined = combine_signals(signals_list, mode=entry_mode, vote_k=vote_k)
            df_sig = build_signal_frame(
                df,
                combined,
                atr_series=atr_series if requires_atr else None,
            )

            entry_iter_fn = lambda d: iter_entries_from_signals(d, use_atr=requires_atr)
            prefilter_entries_cache: list[EntryEvent] | None = None
            single_rule = entry_combo[0] if len(entry_combo) == 1 and isinstance(entry_combo[0], dict) else None
            single_rule_name = str(single_rule.get("name", "")).strip() if single_rule is not None else ""
            exact_monkey_schedule_usable = bool(
                monkey_exact_scheduler_enabled
                and monkey_prefilter_cfg is not None
                and single_rule_name in MONKEY_ENTRY_PLUGIN_NAMES
            )
            if (monkey_prefilter_cfg is not None or monkey_fast_summary_active) and not exact_monkey_schedule_usable:
                prefilter_entries_cache = list(entry_iter_fn(df_sig))
        else:
            df_sig = None
            entry_iter_fn = None
            prefilter_entries_cache = None
            single_rule = entry_combo[0]
            single_rule_name = str(single_rule.get("name", "")).strip()
            exact_monkey_schedule_usable = False

        for exit_params in exit_param_space:
            attempt_count += 1
            prefilter_metrics: ScheduleMetrics | None = None
            run_entry_iter_fn = entry_iter_fn
            run_exit_plugin = exit_plugin
            fast_summary_entries: list[EntryEvent] | None = None
            fast_summary_supports_entry = prefilter_exit_supports_entry

            if full_system_runner is not None:
                iter_count += 1
                _eq, _trades, summary = full_system_runner(
                    df,
                    entry_params=dict(single_rule.get("params", {})),
                    exit_plugin=run_exit_plugin,
                    exit_params=dict(exit_params),
                    cfg=cfg,
                    sizing_plugin=sizing_plugin,
                    sizing_params=sizing_params,
                )
            elif exact_monkey_schedule_usable and single_rule is not None:
                try:
                    exact_entries, exact_metrics = build_exact_monkey_entries_for_time_exit(
                        df_sig=df_sig,
                        entry_params=dict(single_rule.get("params", {})),
                        exit_params=dict(exit_params),
                        monkey_cfg=monkey_prefilter_cfg,
                    )
                    prefilter_metrics = exact_metrics
                    run_entry_iter_fn = (lambda _entries=exact_entries: (lambda _d: iter(_entries)))()
                    run_exit_plugin = build_precomputed_time_exit_wrapper(exit_plugin)
                    fast_summary_entries = exact_entries
                    fast_summary_supports_entry = True
                except Exception as exc:
                    prefilter_reject_count += 1
                    if progress_every and (attempt_count % progress_every == 0 or attempt_count == total):
                        elapsed = time.time() - start_ts
                        eta = (elapsed / attempt_count) * (total - attempt_count) if attempt_count else 0.0
                        print(
                            f"[{attempt_count:>4}/{total}] "
                            f"{(100*attempt_count/total if total else 100.0):6.2f}% | "
                            f"elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s | "
                            f"exact scheduler reject {prefilter_reject_count} "
                            f"({type(exc).__name__}: {exc}) | accepted {iter_count}",
                            flush=True,
                            )
                        continue
            elif monkey_prefilter_cfg is not None and prefilter_entries_cache is not None:
                prefilter_metrics = simulate_flat_only_time_exit_schedule(
                    entries=prefilter_entries_cache,
                    build_exit_fn=run_exit_plugin,
                    exit_params=exit_params,
                    n_bars=len(df_sig),
                    supports_entry_arg=prefilter_exit_supports_entry,
                )
                if prefilter_metrics is None:
                    # Fallback: if schedule prefilter can't evaluate this plugin instance, run normal backtest.
                    pass
                else:
                    prefilter_ok, _prefilter_reasons = prefilter_schedule_matches(prefilter_metrics, monkey_prefilter_cfg)
                    if not prefilter_ok:
                        prefilter_reject_count += 1
                        if progress_every and (attempt_count % progress_every == 0 or attempt_count == total):
                            elapsed = time.time() - start_ts
                            eta = (elapsed / attempt_count) * (total - attempt_count) if attempt_count else 0.0
                            print(
                                f"[{attempt_count:>4}/{total}] "
                                f"{(100*attempt_count/total if total else 100.0):6.2f}% | "
                                f"elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s | "
                                f"prefilter reject {prefilter_reject_count} | "
                                f"accepted {iter_count}",
                                flush=True,
                            )
                        continue
                fast_summary_entries = prefilter_entries_cache
            elif monkey_fast_summary_active and prefilter_entries_cache is not None:
                fast_summary_entries = prefilter_entries_cache

            if full_system_runner is None:
                iter_count += 1
                if monkey_fast_summary_active and fast_summary_entries is not None:
                    _eq, _trades, summary = run_backtest_limited_time_exit_fast_summary(
                        df_sig,
                        cfg=cfg,
                        entries=fast_summary_entries,
                        build_exit_fn=run_exit_plugin,
                        exit_params=exit_params,
                        size_fn=size_fn,
                        supports_entry_arg=bool(fast_summary_supports_entry),
                    )
                else:
                    _eq, _trades, summary = run_backtest_limited(
                        df_sig,
                        cfg=cfg,
                        entry_iter_fn=run_entry_iter_fn,
                        build_exit_fn=run_exit_plugin,
                        exit_params=exit_params,
                        size_fn=size_fn,
                    )

            if save_trades and _trades is not None and not _trades.empty and trade_rows is not None:
                tdf = _trades.copy()
                tdf["iter"] = iter_count
                tdf["entry_params"] = json.dumps([r["params"] for r in entry_combo], default=str)
                tdf["exit_params"] = json.dumps(exit_params, default=str)
                trade_rows.append(tdf)

            trades_ok = summary.get("trades", 0) >= min_trades
            if monkey_davey_enabled:
                ret = summary.get("total_return_%")
                dd_metric = str(monkey_davey_cfg.get("dd_metric", "max_drawdown_abs_%"))
                dd = summary.get(dd_metric)
                if dd is None and dd_metric == "max_drawdown_abs_%":
                    dd_alt = summary.get("max_drawdown_%")
                    if isinstance(dd_alt, (int, float)) and math.isfinite(float(dd_alt)):
                        dd = abs(float(dd_alt))
                return_worse = isinstance(ret, (int, float)) and math.isfinite(float(ret)) and float(ret) < float(monkey_davey_cfg["baseline_return_%"])
                dd_worse = isinstance(dd, (int, float)) and math.isfinite(float(dd)) and float(dd) > float(monkey_davey_cfg["baseline_max_dd_%"])
                ok = bool(trades_ok and return_worse and dd_worse)
                if trades_ok and return_worse:
                    davey_return_worse_count += 1
                if trades_ok and dd_worse:
                    davey_dd_worse_count += 1
            else:
                return_worse = None
                dd_worse = None
                ok = bool(trades_ok and criteria_pass(summary, criteria))
            if ok:
                pass_count += 1

            rows.append({
                "iter": iter_count,
                "attempt": attempt_count,
                "entry_params": [r["params"] for r in entry_combo],
                "exit_params": exit_params,
                **({
                    "prefilter_trades": prefilter_metrics.get("trades"),
                    "prefilter_long_trade_pct": prefilter_metrics.get("long_trade_pct"),
                    "prefilter_avg_bars_held": prefilter_metrics.get("avg_bars_held"),
                } if isinstance(prefilter_metrics, dict) else {}),
                **summary,
                **(
                    {
                        "davey_return_worse": bool(return_worse),
                        "davey_maxdd_worse": bool(dd_worse),
                        "davey_both_worse": bool(return_worse and dd_worse),
                    }
                    if monkey_davey_enabled
                    else {}
                ),
                "favourable": ok,
            })

            if progress_every and (attempt_count % progress_every == 0 or attempt_count == total):
                elapsed = time.time() - start_ts
                pass_pct = (pass_count / iter_count) * 100 if iter_count else 0.0
                print_progress(attempt_count, total, elapsed, summary, pass_pct=pass_pct, criteria=criteria)

            if monkey_seq_enabled and iter_count > 0:
                min_accepted = int(monkey_seq_cfg.get("min_accepted", 1000))
                check_every = int(monkey_seq_cfg.get("check_every", 200))
                if iter_count >= min_accepted and (iter_count % check_every == 0):
                    z = float(monkey_seq_cfg.get("z", 1.96))
                    lo, hi = wilson_interval(pass_count, iter_count, z=z)
                    lo_pct = lo * 100.0
                    hi_pct = hi * 100.0
                    fail_threshold = float(monkey_seq_cfg.get("fail_threshold_pct", 75.0))
                    if lo_pct > pass_threshold:
                        early_stop_triggered = True
                        early_stop_decision = "PASS"
                        early_stop_ci_pct = (float(lo_pct), float(hi_pct))
                        early_stop_reason = (
                            f"Wilson CI lower bound {lo_pct:.2f}% > pass threshold {pass_threshold:.2f}% "
                            f"after {iter_count} accepted runs"
                        )
                    elif hi_pct < fail_threshold:
                        early_stop_triggered = True
                        early_stop_decision = "FAIL"
                        early_stop_ci_pct = (float(lo_pct), float(hi_pct))
                        early_stop_reason = (
                            f"Wilson CI upper bound {hi_pct:.2f}% < fail threshold {fail_threshold:.2f}% "
                            f"after {iter_count} accepted runs"
                        )
                    if early_stop_triggered:
                        print(f"[INFO] monkey sequential stop {early_stop_decision}: {early_stop_reason}", flush=True)
                        break

        if early_stop_triggered:
            break

    res_df = pd.DataFrame(rows)
    res_df.to_csv(limited_iterations_path(run_dir), index=False)
    if trade_rows:
        trades_df = pd.concat(trade_rows, ignore_index=True)
        trades_df.to_csv(limited_trades_path(run_dir), index=False)

    pass_rate = limited_test_pass_rate(res_df)
    davey_return_worse_pct = (
        float((100.0 * davey_return_worse_count / len(res_df)))
        if monkey_davey_enabled and len(res_df) > 0
        else None
    )
    davey_dd_worse_pct = (
        float((100.0 * davey_dd_worse_count / len(res_df)))
        if monkey_davey_enabled and len(res_df) > 0
        else None
    )
    pass_decision = (
        bool(
            davey_return_worse_pct is not None
            and davey_dd_worse_pct is not None
            and davey_return_worse_pct >= pass_threshold
            and davey_dd_worse_pct >= pass_threshold
        )
        if monkey_davey_enabled
        else bool(pass_rate >= pass_threshold)
    )
    pass_summary = {
        "favourable_pct": pass_rate,
        "pass_threshold_%": pass_threshold,
        "passed": pass_decision,
        "total_iters": int(len(res_df)),
        "attempted_iters": int(attempt_count) if attempt_count else int(total),
        "prefilter_rejects": int(prefilter_reject_count),
        "min_trades": min_trades,
    }
    if monkey_davey_enabled:
        pass_summary["davey_style"] = {
            "enabled": True,
            "baseline_return_%": float(monkey_davey_cfg["baseline_return_%"]),
            "baseline_max_dd_%": float(monkey_davey_cfg["baseline_max_dd_%"]),
            "dd_metric": str(monkey_davey_cfg["dd_metric"]),
            "return_worse_pct": float(davey_return_worse_pct) if davey_return_worse_pct is not None else float("nan"),
            "maxdd_worse_pct": float(davey_dd_worse_pct) if davey_dd_worse_pct is not None else float("nan"),
            "both_worse_pct": float(pass_rate),
            "pass_rule": "return_worse_pct >= pass_threshold AND maxdd_worse_pct >= pass_threshold",
        }

    if monkey_seq_enabled:
        pass_summary["sequential_stop"] = {
            "triggered": bool(early_stop_triggered),
            "decision": early_stop_decision,
            "reason": early_stop_reason,
            "accepted_iters": int(len(res_df)),
            "attempted_iters": int(attempt_count) if attempt_count else int(total),
            "wilson_ci_pct": (
                [float(early_stop_ci_pct[0]), float(early_stop_ci_pct[1])]
                if isinstance(early_stop_ci_pct, tuple) and len(early_stop_ci_pct) == 2
                else None
            ),
        }
    write_json(summary_path(run_dir), pass_summary)

    if monkey_prefilter_cfg is not None:
        accepted = int(len(res_df))
        attempts = int(attempt_count) if attempt_count else int(total)
        acc_pct = (100.0 * accepted / attempts) if attempts else 0.0
        print(
            f"Prefilter acceptance: {accepted}/{attempts} ({acc_pct:.1f}%) "
            f"| rejected before backtest: {prefilter_reject_count}",
            flush=True,
        )
    if early_stop_triggered and early_stop_reason:
        print(f"Sequential stop: {early_stop_decision} | {early_stop_reason}", flush=True)
    if monkey_davey_enabled:
        ret_pct_print = float("nan") if davey_return_worse_pct is None else float(davey_return_worse_pct)
        dd_pct_print = float("nan") if davey_dd_worse_pct is None else float(davey_dd_worse_pct)
        print(
            "Davey metrics: "
            f"return_worse%={ret_pct_print:.1f} | "
            f"maxdd_worse%={dd_pct_print:.1f} | "
            f"both_worse%={pass_rate:.1f}",
            flush=True,
        )
    print(f"Favourable%: {pass_rate:.1f}% -> {'PASS' if pass_decision else 'FAIL'}")
    print(f"Overall Result: {'PASS' if bool(pass_summary.get('passed', False)) else 'FAIL'}")
    print(f"Saved: {limited_iterations_path(run_dir)}")
    if trade_rows:
        print(f"Saved: {limited_trades_path(run_dir)}")
