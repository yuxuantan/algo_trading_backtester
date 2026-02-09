from __future__ import annotations

from typing import Any, Callable
import pandas as pd
import numpy as np

try:
    import optuna
except ImportError as e:
    raise ImportError(
        "Optuna is not installed. Install with: pip install optuna\n"
        "Or add it to pyproject.toml dependencies."
    ) from e



def optuna_search(
    *,
    study_name: str,
    storage_url: str,
    sampler: str,
    direction: str,
    n_trials: int,
    timeout_s: int | None,
    seed: int | None,
    suggest_fn: Callable[[optuna.trial.Trial], dict[str, Any]],
    run_once_fn: Callable[[dict[str, Any]], dict[str, Any] | None],
    objective_key: str,
    constraints_fn: Callable[[dict[str, Any]], bool] | None = None,
    prune_fn: Callable[[dict[str, Any]], bool] | None = None,
    progress_fn: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:

    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be 'maximize' or 'minimize'")

    sampler_l = sampler.lower()
    if sampler_l == "tpe":
        opt_sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_l == "random":
        opt_sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError("sampler must be 'tpe' or 'random'")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction=direction,
        sampler=opt_sampler,
        pruner=optuna.pruners.NopPruner(),  # keep stable; pruning handled by prune_fn
    )
    
    def _safe_best_value(study):
        try:
            return study.best_value
        except Exception:
            return None


    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_fn(trial)

        if constraints_fn is not None and not constraints_fn(params):
            raise optuna.TrialPruned("Rejected by constraints_fn")

        summary = run_once_fn(params)
        if summary is None:
            raise optuna.TrialPruned("run_once_fn returned None (rejected)")

        if objective_key not in summary or summary[objective_key] is None:
            raise optuna.TrialPruned(f"Missing objective '{objective_key}' in summary")

        if prune_fn is not None and prune_fn(summary):
            raise optuna.TrialPruned("Pruned by prune_fn")

        obj_val = float(summary[objective_key])

        # store summary metrics into trial attrs (for export)
        trial.set_user_attr("_objective_key", objective_key)
        trial.set_user_attr("_objective_value", obj_val)

        for k, v in summary.items():
            if v is None:
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                trial.set_user_attr(k, float(v))
            else:
                trial.set_user_attr(k, str(v))

        if progress_fn is not None:
            progress_fn({
                "trial_number": trial.number,
                "params": params,
                "objective": obj_val,
                "summary": summary,
                "best_value": _safe_best_value(study),
            })

        return obj_val

    timeout = None if (timeout_s is None or timeout_s <= 0) else timeout_s
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)

    rows = []
    for t in study.get_trials(deepcopy=False):
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {}
        row.update(t.params)
        row["trial_number"] = t.number
        row["value"] = t.value
        row.update(t.user_attrs)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_col = objective_key if objective_key in df.columns else "value"
    df = df.sort_values(sort_col, ascending=(direction == "minimize")).reset_index(drop=True)
    return df
