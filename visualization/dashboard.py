"""Streamlit dashboard to launch and monitor optimizations.

Features:
- Configuration editor for optimization parameters.
- Real-time status monitoring with dynamic ETA and progress bar.
- Visual reporting of best trial results.
- Overfitting check launcher.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
import yaml

# --- Configuration du chemin de projet ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
except NameError:
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover - optional dependency for timer refresh
    st_autorefresh = None

from optimization.config_overrides import apply_overrides
from optimization.dashboard_runner import (
    OptimizationJobConfig,
    build_job_config,
    generate_best_trial_report,
    get_optimization_status,
    locate_overfitting_index,
    reset_job_state,
    start_optimization_job,
    start_overfitting_checks,
)
from scripts.run_optimization import load_config


def load_available_optimization_configs() -> Dict[str, Path]:
    """Return a mapping of friendly names to optimization YAML files."""

    config_dir = PROJECT_ROOT / "config"
    mapping: Dict[str, Path] = {}
    for path in sorted(config_dir.glob("optimization_*.yaml")):
        label = path.stem.replace("optimization_", "")
        mapping[label] = path
    return mapping


def _human_eta(seconds: Optional[float]) -> str:
    """Format a duration in seconds into a human-readable string (e.g., '1h 30m')."""
    if seconds is None:
        return "N/A"
    if seconds <= 0:
        return "completed"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {sec:02d}s"


def _display_config_details(config: Dict[str, Any]) -> None:
    opt_cfg = config.get("optimization", config) or {}
    strategy_cfg = opt_cfg.get("strategy", {}) or {}
    data_cfg = opt_cfg.get("data", {}) or {}
    study_cfg = opt_cfg.get("study", {}) or {}

    st.subheader("Configuration")
    cols = st.columns(3)
    cols[0].metric("Strategy", strategy_cfg.get("name") or "N/A")
    cols[1].metric("Module", strategy_cfg.get("module") or "N/A")
    cols[2].metric("Class", strategy_cfg.get("class_name") or "N/A")

    st.markdown("**Data**")
    tickers = data_cfg.get("tickers") or data_cfg.get("ticker") or "N/A"
    st.write(
        f"- Tickers: {tickers}\n"
        f"- Interval: {data_cfg.get('interval', 'N/A')} | "
        f"{data_cfg.get('start_date', 'N/A')} -> {data_cfg.get('end_date', 'N/A')}"
    )

    st.markdown("**Optuna Study**")
    st.write(
        f"- Study: {study_cfg.get('study_name', 'N/A')}\n"
        f"- Storage: {study_cfg.get('storage', 'N/A')}\n"
        f"- n_trials: {study_cfg.get('n_trials', 'N/A')} | "
        f"timeout: {study_cfg.get('timeout', 'N/A')} | "
        f"n_jobs: {study_cfg.get('n_jobs', 'N/A')}"
    )


def _render_status(status_cfg: OptimizationJobConfig) -> Optional[Dict[str, Any]]:
    """Display the current optimization status, including progress bar and ETA.

    Args:
        status_cfg: Configuration for the job to monitor.

    Returns:
        A dictionary containing status details (status, best_params, best_value) if successful, else None.
    """
    try:
        status = get_optimization_status(status_cfg)
    except Exception as exc:
        st.error(f"Unable to read job status: {exc}")
        return None

    st.subheader("Current Status")
    cols = st.columns(4)
    cols[0].metric("State", status.status)
    cols[1].metric("Completed", str(status.n_trials_completed))
    cols[2].metric("Planned", status.n_trials_planned or "N/A")
    cols[3].metric("ETA", _human_eta(status.eta_seconds))

    progress = 0.0
    if status.n_trials_planned:
        progress = min(status.n_trials_completed / status.n_trials_planned, 1.0)
    
    st.progress(progress, text=f"Progress: {int(progress * 100)}%")

    if status.best_value is not None:
        st.info(f"Best value: {status.best_value}")
    if status.best_params:
        st.json(status.best_params)
    if status.error_message:
        st.warning(status.error_message)

    return {
        "status": status.status,
        "best_params": status.best_params,
        "best_value": status.best_value,
    }


def _parse_tickers(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    else:
        return []
    return [v for v in items if v]


def _build_data_overrides(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    default_tickers = _parse_tickers(
        data_cfg.get("tickers") or data_cfg.get("ticker") or ""
    )
    tickers_text = st.text_input(
        "Tickers (comma-separated)", value=", ".join(default_tickers)
    )
    tickers = _parse_tickers(tickers_text)
    if tickers:
        if len(tickers) == 1:
            overrides["ticker"] = tickers[0]
            overrides.pop("tickers", None)
        else:
            overrides["tickers"] = tickers
            overrides.pop("ticker", None)

    start_text = st.text_input(
        "Start date (YYYY-MM-DD)", value=str(data_cfg.get("start_date") or "")
    )
    end_text = st.text_input(
        "End date (YYYY-MM-DD)", value=str(data_cfg.get("end_date") or "")
    )
    if start_text:
        overrides["start_date"] = start_text
    if end_text:
        overrides["end_date"] = end_text

    interval_options = ["1d", "4h", "1h", "15m"]
    current_interval = str(data_cfg.get("interval") or interval_options[0])
    if current_interval not in interval_options:
        interval_options = [current_interval] + interval_options
    interval_choice = st.selectbox("Interval", interval_options, index=0)
    if interval_choice:
        overrides["interval"] = interval_choice

    return {k: v for k, v in overrides.items() if v not in (None, "", [])}


def _edit_param_spec(name: str, spec: Any) -> Any:
    # Dict with explicit type
    if isinstance(spec, dict) and "type" in spec:
        ptype = str(spec.get("type", "")).lower()
        if ptype in ("int", "integer"):
            low = int(spec.get("low", 0))
            high = int(spec.get("high", low + 1))
            step = int(spec.get("step", 1) or 1)
            low = st.number_input(f"{name} low", value=low, step=1)
            high = st.number_input(f"{name} high", value=high, step=1)
            step = st.number_input(f"{name} step", value=step, step=1, min_value=1)
            return {"type": "int", "low": low, "high": high, "step": step}

            # Unreachable, but keep structure consistent
        if ptype in ("float", "uniform"):
            low = float(spec.get("low", 0.0))
            high = float(spec.get("high", low + 1.0))
            step_val = spec.get("step")
            log_scale = bool(spec.get("log") or (str(spec.get("scale", "")).lower() == "log"))
            low = st.number_input(f"{name} low", value=low)
            high = st.number_input(f"{name} high", value=high)
            step_input = st.number_input(
                f"{name} step (0 = auto)", value=float(step_val or 0.0)
            )
            log_scale = st.checkbox(f"{name} log scale", value=log_scale)
            payload: Dict[str, Any] = {"type": "float", "low": low, "high": high}
            if step_input > 0:
                payload["step"] = step_input
            if log_scale:
                payload["log"] = True
            return payload

        if ptype in ("categorical", "choice"):
            choices = spec.get("choices") or []
            default_str = ", ".join(map(str, choices))
            text = st.text_input(f"{name} choices (comma-separated)", value=default_str)
            new_choices = [c.strip() for c in text.split(",") if c.strip()]
            return {"type": "categorical", "choices": new_choices or choices}

        st.info(f"{name}: unsupported spec type '{ptype}', kept as-is.")
        return spec

    # Numeric range in list/tuple [low, high, step?]
    if isinstance(spec, (list, tuple)) and len(spec) in (2, 3) and all(
        isinstance(v, (int, float)) for v in spec[:2]
    ):
        low = float(spec[0])
        high = float(spec[1])
        step_val = float(spec[2]) if len(spec) == 3 else 0.0
        low = st.number_input(f"{name} low", value=low)
        high = st.number_input(f"{name} high", value=high)
        step_input = st.number_input(f"{name} step (0 = auto)", value=step_val)
        if step_input > 0:
            return [low, high, step_input]
        return [low, high]

    # List of values -> categorical
    if isinstance(spec, (list, tuple)):
        default_str = ", ".join(map(str, spec))
        text = st.text_input(f"{name} values (comma-separated)", value=default_str)
        values = [v.strip() for v in text.split(",") if v.strip()]
        return values or spec

    # Scalar -> fixed value
    if isinstance(spec, (int, float)):
        val = st.number_input(f"{name} (fixed)", value=float(spec))
        return type(spec)(val)

    text = st.text_input(f"{name} (fixed)", value=str(spec))
    return text or spec


def _build_param_space_overrides(strategy_cfg: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    param_space = strategy_cfg.get("param_space", {}) or {}
    if not param_space:
        st.info("No param_space defined in strategy config.")
        return overrides

    st.subheader("Hyperparameters (auto-discovered)")
    for name, spec in param_space.items():
        with st.expander(name, expanded=False):
            new_spec = _edit_param_spec(name, spec)
            if new_spec != spec:
                overrides[name] = new_spec
    return overrides


def _count_param_options(spec: Any) -> Optional[int]:
    if isinstance(spec, dict) and "type" in spec:
        ptype = str(spec.get("type", "")).lower()
        if ptype in ("int", "integer"):
            low = spec.get("low")
            high = spec.get("high")
            step = spec.get("step", 1)
            if low is None or high is None or not step:
                return None
            try:
                count = int(math.floor((float(high) - float(low)) / float(step))) + 1
                return count if count > 0 else None
            except Exception:
                return None
        if ptype in ("float", "uniform"):
            low = spec.get("low")
            high = spec.get("high")
            step = spec.get("step")
            if low is None or high is None or step in (None, 0, 0.0):
                return None
            try:
                count = int(math.floor((float(high) - float(low)) / float(step))) + 1
                return count if count > 0 else None
            except Exception:
                return None
        if ptype in ("categorical", "choice"):
            choices = spec.get("choices") or []
            return len(list(choices))
        return None

    if isinstance(spec, (list, tuple)) and len(spec) in (2, 3) and all(
        isinstance(v, (int, float)) for v in spec[:2]
    ):
        low, high = spec[0], spec[1]
        step = spec[2] if len(spec) == 3 else 1
        if step in (None, 0, 0.0):
            return None
        try:
            count = int(math.floor((float(high) - float(low)) / float(step))) + 1
            return count if count > 0 else None
        except Exception:
            return None

    if isinstance(spec, (list, tuple)):
        return len(list(spec))

    return 1


def _compute_param_grid_size(
    param_space: Dict[str, Any], overrides: Dict[str, Any]
) -> tuple[Optional[int], Dict[str, Optional[int]]]:
    combined = dict(param_space)
    combined.update(overrides)
    per_param: Dict[str, Optional[int]] = {}
    total: Optional[int] = 1
    for name, spec in combined.items():
        count = _count_param_options(spec)
        per_param[name] = count
        if count is None:
            total = None
        elif total is not None:
            total *= count
    return total, per_param


def _dump_temp_config(config: Dict[str, Any]) -> Path:
    tmp_dir = PROJECT_ROOT / "tmp-output"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"dashboard_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


def _reset_state() -> None:
    reset_job_state()
    st.session_state.pop("job_cfg", None)


def main() -> None:
    st.set_page_config(page_title="Optimization Dashboard", layout="wide")
    st.title("Optimization Launcher")

    config_map = load_available_optimization_configs()
    if not config_map:
        st.error("No optimization configs found in config/optimization_*.yaml.")
        return

    selected_label = st.selectbox("Config file", list(config_map.keys()))
    config_path = config_map[selected_label]
    config_data = load_config(str(config_path))
    _display_config_details(config_data)

    overrides_col1, overrides_col2, overrides_col3 = st.columns(3)
    opt_cfg = config_data.get("optimization", config_data) or {}
    data_cfg = opt_cfg.get("data", {}) or {}
    strategy_cfg = opt_cfg.get("strategy", {}) or {}
    study_cfg = opt_cfg.get("study", {}) or {}

    n_trials_override = overrides_col1.number_input(
        "n_trials override (0 = YAML default)",
        min_value=0,
        value=study_cfg.get("n_trials") or 0,
        step=1,
        format="%d",
    )
    timeout_override = overrides_col2.number_input(
        "Timeout override (sec, 0 = YAML default)",
        min_value=0,
        value=study_cfg.get("timeout") or 0,
        step=60,
        format="%d",
    )
    n_jobs_override = overrides_col3.number_input(
        "n_jobs override (0 = YAML default)",
        min_value=-1,
        value=study_cfg.get("n_jobs") or 0,
        step=1,
        format="%d",
    )

    st.subheader("Data overrides")
    data_overrides = _build_data_overrides(data_cfg)

    param_space_overrides = _build_param_space_overrides(strategy_cfg)
    total_grid, per_param_grid = _compute_param_grid_size(
        strategy_cfg.get("param_space", {}) or {}, param_space_overrides
    )
    st.markdown("**Grid size (discrete combinations)**")
    if total_grid is None:
        st.info("Grid size: N/A (non-discrete parameters present).")
    else:
        st.metric("Total combinations", f"{total_grid:,}")
    if per_param_grid:
        with st.expander("Per-parameter cardinality"):
            st.json(per_param_grid)

    study_overrides: Dict[str, Any] = {}
    if n_trials_override > 0:
        study_overrides["n_trials"] = n_trials_override
    if timeout_override > 0:
        study_overrides["timeout"] = timeout_override
    if n_jobs_override not in {0, None}:
        study_overrides["n_jobs"] = n_jobs_override

    if st.button("Launch optimization", type="primary"):
        effective_config = apply_overrides(
            config_data,
            data_overrides=data_overrides or None,
            study_overrides=study_overrides or None,
            param_space_overrides=param_space_overrides or None,
        )
        temp_config_path = _dump_temp_config(effective_config)
        job_cfg = build_job_config(
            temp_config_path,
            n_trials=study_overrides.get("n_trials") if study_overrides else None,
            timeout=study_overrides.get("timeout") if study_overrides else None,
            n_jobs=study_overrides.get("n_jobs") if study_overrides else None,
        )
        start_optimization_job(job_cfg)
        st.session_state["job_cfg"] = job_cfg
        st.success("Optimization started.")

    st.button("Reset dashboard", on_click=_reset_state)

    current_cfg: OptimizationJobConfig = st.session_state.get(
        "job_cfg"
    ) or build_job_config(config_path)
    status_payload = _render_status(current_cfg)

    if status_payload and status_payload["status"] == "running" and st_autorefresh:
        st_autorefresh(interval=5000, limit=1000, key="status_refresh")

    if status_payload and status_payload["status"] == "done":
        st.subheader("Best Trial Backtest")
        if st.button("Generate best-trial report"):
            with st.spinner("Running backtest for best params..."):
                try:
                    report_path = generate_best_trial_report(
                        current_cfg.config_path,
                        status_payload["best_params"] or {},
                    )
                    abs_path = report_path.resolve()
                    st.success(f"Report generated at {abs_path}")
                    try:
                        st.markdown(f"[Open report]({abs_path.as_uri()})")
                    except ValueError:
                        st.info(f"Open this file in your browser: {abs_path}")
                    try:
                        html = abs_path.read_text(encoding="utf-8")
                        st.components.v1.html(html, height=800, scrolling=True)
                    except Exception:
                        st.warning("Unable to embed report preview; please open the file directly.")
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")

        st.subheader("Overfitting Checks")
        overfit_cols = st.columns(2)
        if overfit_cols[0].button("Run overfitting checks"):
            try:
                pid = start_overfitting_checks(current_cfg.config_path)
                overfit_cols[0].success(f"Started overfitting checks (pid {pid})")
            except Exception as exc:
                overfit_cols[0].error(f"Unable to start checks: {exc}")

        index_path = locate_overfitting_index(current_cfg.config_path)
        if index_path:
            overfit_cols[1].markdown(
                f"[View last overfitting report]({index_path.as_uri()})"
            )
        else:
            overfit_cols[1].info("No overfitting reports found yet.")


if __name__ == "__main__":
    main()



