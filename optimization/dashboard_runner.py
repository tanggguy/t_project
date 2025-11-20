"""Utilities to drive Optuna optimizations from the dashboard."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import optuna
import pandas as pd
import psutil

from backtesting.analyzers import performance as perf
from backtesting.engine import BacktestEngine
from backtesting.portfolio import (
    aggregate_weighted_returns,
    compute_portfolio_metrics,
    normalize_weights,
)
from scripts.run_backtest import (
    configure_position_sizing,
    get_strategy_defaults,
    merge_params,
)
from scripts.run_optimization import (
    discover_strategies,
    load_config,
    resolve_strategy,
)
from utils.config_loader import get_settings
from utils.data_manager import DataManager
from utils.logger import setup_logger

logger = setup_logger(__name__, log_file="logs/optimization/dashboard_runner.log")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
except Exception:
    PROJECT_ROOT = Path.cwd()

STATE_PATH = PROJECT_ROOT / "tmp-output" / "current_optimization.json"


@dataclass
class OptimizationJobConfig:
    """Configuration describing an optimization job to launch."""

    config_path: Path
    n_trials: Optional[int] = None
    timeout: Optional[int] = None
    n_jobs: Optional[int] = None
    study_name: Optional[str] = None
    storage_url: Optional[str] = None
    best_params_path: Optional[Path] = None


@dataclass
class OptimizationJobStatus:
    """Snapshot of an optimization execution.

    Attributes:
        status: Current state of the job (idle, running, done, failed).
        n_trials_planned: Total number of trials expected.
        n_trials_completed: Number of trials finished so far.
        avg_trial_duration: Average duration of recent trials (used for ETA).
        eta_seconds: Estimated time remaining in seconds.
        best_value: Best objective value found so far.
        best_params: Parameters corresponding to the best value.
        last_update: Timestamp of the last completed trial.
        error_message: Error details if the job failed.
    """

    status: Literal["idle", "running", "done", "failed"]
    n_trials_planned: Optional[int]
    n_trials_completed: int
    avg_trial_duration: Optional[float]
    eta_seconds: Optional[float]
    best_value: Optional[float | list[float]]
    best_params: Optional[Dict[str, Any]]
    last_update: Optional[datetime]
    error_message: Optional[str] = None


def build_job_config(
    config_path: Path,
    *,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> OptimizationJobConfig:
    """Build an OptimizationJobConfig by merging YAML defaults and overrides."""

    resolved_cfg = config_path.resolve()
    cfg = load_config(str(resolved_cfg))
    opt_cfg = cfg.get("optimization", cfg) or {}
    study_cfg = opt_cfg.get("study", {}) or {}
    output_cfg = opt_cfg.get("output", {}) or {}

    return OptimizationJobConfig(
        config_path=resolved_cfg,
        n_trials=n_trials if n_trials is not None else study_cfg.get("n_trials"),
        timeout=timeout if timeout is not None else study_cfg.get("timeout"),
        n_jobs=n_jobs if n_jobs is not None else study_cfg.get("n_jobs"),
        study_name=study_cfg.get("study_name"),
        storage_url=study_cfg.get("storage"),
        best_params_path=Path(output_cfg["best_params_path"])
        if output_cfg.get("best_params_path")
        else None,
    )


def _ensure_state_dir() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_lock(payload: Dict[str, Any]) -> None:
    _ensure_state_dir()
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_lock() -> Optional[Dict[str, Any]]:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Unable to read lock file, resetting it.")
        try:
            STATE_PATH.unlink()
        except Exception:
            logger.debug("Could not remove corrupt lock file.", exc_info=True)
        return None


def _process_is_running(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except Exception:
        logger.debug("Failed to inspect process %s", pid, exc_info=True)
        return False


def start_optimization_job(job_cfg: OptimizationJobConfig) -> None:
    """Launch the CLI in a detached process and record a lock file."""

    completed_at_start = 0
    if job_cfg.study_name and job_cfg.storage_url:
        try:
            existing = optuna.load_study(
                study_name=job_cfg.study_name,
                storage=job_cfg.storage_url,
            )
            completed_at_start = len([t for t in existing.trials if t.state.is_finished()])
        except Exception:
            logger.debug("No existing study to warm start counts.", exc_info=True)

    script_path = PROJECT_ROOT / "scripts" / "run_optimization.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(job_cfg.config_path),
    ]

    if job_cfg.n_trials is not None:
        cmd.extend(["--n-trials", str(job_cfg.n_trials)])
    if job_cfg.timeout is not None:
        cmd.extend(["--timeout", str(job_cfg.timeout)])
    if job_cfg.n_jobs is not None:
        cmd.extend(["--n-jobs", str(job_cfg.n_jobs)])

    logger.info("Starting optimization: %s", " ".join(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(PROJECT_ROOT),
    )

    payload = {
        "config_path": str(job_cfg.config_path),
        "pid": process.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "study_name": job_cfg.study_name,
        "storage_url": job_cfg.storage_url,
        "planned_n_trials": job_cfg.n_trials,
        "completed_at_start": completed_at_start,
    }
    _write_lock(payload)


def _load_study(job_cfg: OptimizationJobConfig, *, quiet_missing: bool = True) -> Optional[optuna.Study]:
    if not job_cfg.study_name or not job_cfg.storage_url:
        return None
    try:
        return optuna.load_study(
            study_name=job_cfg.study_name,
            storage=job_cfg.storage_url,
        )
    except Exception as exc:
        message = str(exc)
        if quiet_missing and ("Record does not exist" in message or "No such study" in message):
            logger.debug("Study not yet available (%s)", message)
            return None
        logger.error("Failed to load study %s: %s", job_cfg.study_name, message)
        return None


def _compute_eta(
    planned: Optional[int],
    completed: int,
    avg_duration: Optional[float],
) -> Optional[float]:
    """Estimate remaining time based on average duration of recent trials."""
    if planned is None or avg_duration is None:
        return None
    remaining = max(planned - completed, 0)
    if remaining == 0:
        return 0.0
    return remaining * avg_duration


def get_optimization_status(job_cfg: OptimizationJobConfig) -> OptimizationJobStatus:
    """Read the Optuna study and process lock to build a status snapshot.

    Calculates ETA using a moving average of the last 20 trials for better accuracy.
    """

    lock = _read_lock()
    process_running = bool(lock and lock.get("pid") and _process_is_running(int(lock["pid"])))

    study = _load_study(job_cfg)
    finished_trials = []
    best_value: Optional[float | list[float]] = None
    best_params: Optional[Dict[str, Any]] = None
    last_update: Optional[datetime] = None
    error_message: Optional[str] = None

    if study:
        finished_trials = [t for t in study.trials if t.state.is_finished()]
        try:
            values = study.best_trial.values if study.best_trial.values else None
            best_value = values if isinstance(values, list) else study.best_value
            best_params = dict(study.best_params) if study.best_params else None
        except Exception:
            try:
                first_best = study.best_trials[0]
                best_value = list(first_best.values)
                best_params = dict(first_best.params)
            except Exception:
                best_value = None
                best_params = None

        timestamps = [
            t.datetime_complete or t.datetime_start for t in finished_trials if t.datetime_complete
        ]
        if timestamps:
            last_update = max(timestamps)
    elif not process_running:
        error_message = "Study unavailable; check storage configuration."

    durations = []
    for trial in finished_trials:
        if trial.datetime_start and trial.datetime_complete:
            durations.append(
                (trial.datetime_complete - trial.datetime_start).total_seconds()
            )
    
    # Use moving average of last 20 trials for better accuracy
    recent_durations = durations[-20:] if durations else []
    avg_duration = float(sum(recent_durations) / len(recent_durations)) if recent_durations else None
    
    n_completed = len(finished_trials)
    planned_total = job_cfg.n_trials or (lock.get("planned_n_trials") if lock else None)
    completed_at_start = int(lock.get("completed_at_start", 0)) if lock else 0
    added_completed = max(n_completed - completed_at_start, 0)
    eta_seconds = _compute_eta(
        planned_total,
        added_completed if planned_total is not None else n_completed,
        avg_duration,
    )

    status: Literal["idle", "running", "done", "failed"] = "idle"
    if process_running:
        status = "running"
    elif planned_total is not None and added_completed >= planned_total:
        status = "done"
    elif n_completed > 0:
        status = "running" if process_running else "done"
    elif lock and not process_running:
        status = "failed"
        if error_message is None:
            error_message = "Process appears to have stopped unexpectedly."

    return OptimizationJobStatus(
        status=status,
        n_trials_planned=job_cfg.n_trials,
        n_trials_completed=n_completed,
        avg_trial_duration=avg_duration,
        eta_seconds=eta_seconds,
        best_value=best_value,
        best_params=best_params,
        last_update=last_update,
        error_message=error_message,
    )


def reset_job_state() -> None:
    """Remove the lock file and terminate the recorded process if still running."""

    lock = _read_lock()
    if lock and lock.get("pid"):
        pid = int(lock["pid"])
        if _process_is_running(pid):
            try:
                proc = psutil.Process(pid)
                cmdline = " ".join(proc.cmdline())
                if "run_optimization.py" in cmdline:
                    proc.terminate()
                else:
                    logger.warning("Skip terminating pid %s (unexpected command line)", pid)
            except Exception:
                logger.debug("Failed to terminate pid %s", pid, exc_info=True)
    try:
        if STATE_PATH.exists():
            STATE_PATH.unlink()
    except Exception:
        logger.debug("Failed to remove state file.", exc_info=True)


def _extract_time_returns(strat: Any) -> pd.Series:
    try:
        returns_dict = strat.analyzers.timereturns.get_analysis()
        series = pd.Series(returns_dict)
        series.index = pd.to_datetime(series.index)
        return series.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def _extract_trades(strat: Any) -> Optional[pd.DataFrame]:
    try:
        trade_list = strat.analyzers.tradelist.get_analysis()
        trades_df = pd.DataFrame(trade_list)
    except Exception:
        return None

    if trades_df.empty:
        return None

    for col in ("entry_dt", "exit_dt"):
        if col in trades_df.columns:
            trades_df[col] = pd.to_datetime(trades_df[col], errors="coerce")
    trades_df = trades_df.replace({pd.NA: None})
    return trades_df


def _build_engine(
    data_frame: pd.DataFrame,
    broker_cfg: Dict[str, Any],
    position_sizing_cfg: Dict[str, Any],
) -> tuple[BacktestEngine, float]:
    engine = BacktestEngine()
    initial_capital = float(broker_cfg.get("initial_capital", 10000.0))
    engine.cerebro.broker.setcash(initial_capital)

    if "commission_pct" in broker_cfg:
        engine.cerebro.broker.setcommission(
            commission=float(broker_cfg["commission_pct"])
        )
    elif "commission_fixed" in broker_cfg:
        engine.cerebro.broker.setcommission(
            commission=float(broker_cfg["commission_fixed"])
        )

    if "slippage_pct" in broker_cfg:
        slippage = float(broker_cfg["slippage_pct"])
        if slippage > 0:
            engine.cerebro.broker.set_slippage_perc(perc=slippage)

    engine.add_data(data_frame.copy())
    config_payload = {"backtest": {"position_sizing": position_sizing_cfg}}
    configure_position_sizing(engine, config_payload)

    return engine, initial_capital


def _load_data_frames(data_cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    tickers_cfg = data_cfg.get("tickers")
    if tickers_cfg:
        tickers = (
            [tickers_cfg] if isinstance(tickers_cfg, str) else list(tickers_cfg)
        )
    else:
        ticker = data_cfg.get("ticker")
        if not ticker:
            raise ValueError("No ticker provided in data configuration.")
        tickers = [ticker]

    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("No valid tickers provided.")

    data_manager = DataManager()
    frames: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = data_manager.get_data(
            ticker=ticker,
            start_date=data_cfg.get("start_date"),
            end_date=data_cfg.get("end_date"),
            interval=data_cfg.get("interval"),
            use_cache=data_cfg.get("use_cache", True),
        )
        if df is None or df.empty:
            logger.error("Empty dataset for %s; skipping.", ticker)
            continue
        frames[ticker] = df

    if not frames:
        raise ValueError("No data could be loaded for provided tickers.")
    return frames


def generate_best_trial_report(
    config_path: Path,
    best_params: Dict[str, Any],
) -> Path:
    """Run a backtest with best params and emit an HTML report."""

    cfg = load_config(str(config_path))
    opt_cfg = cfg.get("optimization", cfg) or {}
    data_cfg = opt_cfg.get("data", {})
    broker_cfg = opt_cfg.get("broker", {})
    sizing_cfg = opt_cfg.get("position_sizing", {})
    backtest_cfg = cfg.get("backtest") or opt_cfg.get("backtest") or {}
    output_cfg = backtest_cfg.get("output", {}) if isinstance(backtest_cfg, dict) else {}
    report_cfg = output_cfg.get("report", {}) if isinstance(output_cfg, dict) else {}

    available = discover_strategies()
    strategy_cfg = opt_cfg.get("strategy", {}) or {}
    strategy_class = resolve_strategy(strategy_cfg, available)
    defaults = get_strategy_defaults(strategy_class)
    fixed_params = strategy_cfg.get("fixed_params") or {}
    merged_params = merge_params(defaults, fixed_params)
    merged_params.update(best_params or {})

    frames = _load_data_frames(data_cfg)

    returns_map: Dict[str, pd.Series] = {}
    trades_df: Optional[pd.DataFrame] = None
    initial_capital = float(broker_cfg.get("initial_capital", 10000.0))

    for ticker, df in frames.items():
        engine, initial_capital = _build_engine(df, broker_cfg, sizing_cfg)
        engine.add_strategy(strategy_class, **merged_params)
        results = engine.run()
        if not results:
            logger.warning("No results returned for ticker %s", ticker)
            continue
        strat = results[0]
        returns_map[ticker] = _extract_time_returns(strat)
        ticker_trades = _extract_trades(strat)
        if ticker_trades is not None:
            ticker_trades["ticker"] = ticker
            trades_df = (
                ticker_trades if trades_df is None else pd.concat([trades_df, ticker_trades])
            )

    if not returns_map:
        raise ValueError("No returns available to build best-trial report.")

    weights = normalize_weights(list(returns_map.keys()), data_cfg.get("weights"))
    alignment = str(data_cfg.get("alignment", "intersection")).lower()
    portfolio_returns = aggregate_weighted_returns(returns_map, weights, alignment)
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns are empty; cannot build report.")

    analytics_settings = get_settings().get("analytics", {})
    metrics, equity, working_returns, underwater = compute_portfolio_metrics(
        portfolio_returns,
        initial_capital,
        analytics_settings,
    )

    if trades_df is not None and not trades_df.empty:
        trade_stats = perf.compute_trade_stats(trades_df)
        metrics.update(trade_stats)

    out_dir = Path(report_cfg.get("out_dir", "reports/generated"))
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = strategy_cfg.get("name") or strategy_class.__name__
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = (out_dir / f"{safe_name}_best_trial_{timestamp}.html").resolve()
    template = report_cfg.get("template", "default.html")

    meta = {
        "strategy_name": safe_name,
        "ticker": ", ".join(returns_map.keys()),
        "start_date": equity.index.min().date() if not equity.empty else None,
        "end_date": equity.index.max().date() if not equity.empty else None,
    }

    try:
        from reports.report_generator import generate_report
    except Exception as exc:
        logger.error("Cannot import generate_report: %s", exc)
        raise

    generate_report(
        meta=meta,
        metrics=metrics,
        equity=equity,
        underwater=underwater,
        trades=trades_df,
        out_path=str(out_file),
        template=template,
        returns=portfolio_returns,
        log_returns=working_returns,
        analytics_config={
            "periods_per_year": analytics_settings.get("periods_per_year", 252),
            "risk_free_rate": analytics_settings.get("risk_free_rate", 0.0),
            "rolling_window": analytics_settings.get("rolling_window", 63),
        },
    )
    return out_file


def start_overfitting_checks(config_path: Path, *, use_best_params: bool = True) -> Optional[int]:
    """Launch run_overfitting.py in a background process."""

    if use_best_params:
        cfg = load_config(str(config_path))
        opt_cfg = cfg.get("optimization", cfg) or {}
        output_cfg = opt_cfg.get("output", {}) or {}
        best_path = output_cfg.get("best_params_path")
        if not best_path or not Path(best_path).exists():
            raise FileNotFoundError(
                "best_params_path is missing; run optimization with dump_best_params enabled."
            )

    script_path = PROJECT_ROOT / "scripts" / "run_overfitting.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
    ]
    if use_best_params:
        cmd.append("--use-best-params")

    logger.info("Starting overfitting checks: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(PROJECT_ROOT),
    )
    return process.pid


def locate_overfitting_index(config_path: Path) -> Optional[Path]:
    """Locate the most recent overfitting index.html for a given config."""

    cfg = load_config(str(config_path))
    opt_cfg = cfg.get("optimization", cfg) or {}
    strategy_cfg = opt_cfg.get("strategy", {}) or {}
    overfit_cfg = cfg.get("overfitting") or opt_cfg.get("overfitting") or {}

    run_id = overfit_cfg.get("run_id") or strategy_cfg.get("name") or "overfitting"
    root = Path(overfit_cfg.get("output_dir") or "results/overfitting") / run_id
    if not root.exists():
        return None

    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        index_path = candidate / "index.html"
        if index_path.exists():
            return index_path
    return None

