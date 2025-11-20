import json
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

# Add project root to path to ensure imports work
sys.path.append("c:\\Users\\saill\\Desktop\\t_project")

from optimization.dashboard_runner import (
    OptimizationJobConfig,
    OptimizationJobStatus,
    build_job_config,
    _write_lock,
    _read_lock,
    _process_is_running,
    start_optimization_job,
    _compute_eta,
    get_optimization_status,
    reset_job_state,
    generate_best_trial_report,
    start_overfitting_checks,
    locate_overfitting_index,
    STATE_PATH
)

@pytest.fixture
def mock_config_path():
    return Path("config/test_config.yaml")

@pytest.fixture
def mock_config_data():
    return {
        "optimization": {
            "study": {
                "n_trials": 100,
                "timeout": 3600,
                "n_jobs": 4,
                "study_name": "test_study",
                "storage": "sqlite:///test.db"
            },
            "output": {
                "best_params_path": "results/best_params.json"
            },
            "data": {
                "ticker": "AAPL",
                "start_date": "2020-01-01",
                "end_date": "2021-01-01"
            },
            "strategy": {
                "name": "TestStrategy"
            }
        }
    }

@patch("optimization.dashboard_runner.load_config")
def test_build_job_config(mock_load_config, mock_config_path, mock_config_data):
    mock_load_config.return_value = mock_config_data
    
    config = build_job_config(mock_config_path)
    
    assert config.n_trials == 100
    assert config.study_name == "test_study"
    assert config.best_params_path == Path("results/best_params.json")
    
    # Test overrides
    config_override = build_job_config(mock_config_path, n_trials=50)
    assert config_override.n_trials == 50

@patch("optimization.dashboard_runner.STATE_PATH")
def test_write_lock(mock_state_path):
    mock_state_path.parent.mkdir = MagicMock()
    mock_state_path.write_text = MagicMock()
    
    payload = {"pid": 123}
    _write_lock(payload)
    
    mock_state_path.parent.mkdir.assert_called_once()
    mock_state_path.write_text.assert_called_once()
    args, kwargs = mock_state_path.write_text.call_args
    assert '"pid": 123' in args[0]

@patch("optimization.dashboard_runner.STATE_PATH")
def test_read_lock(mock_state_path):
    mock_state_path.exists.return_value = True
    mock_state_path.read_text.return_value = '{"pid": 123}'
    
    lock = _read_lock()
    assert lock == {"pid": 123}
    
    # Test missing file
    mock_state_path.exists.return_value = False
    assert _read_lock() is None
    
    # Test corrupt file
    mock_state_path.exists.return_value = True
    mock_state_path.read_text.return_value = '{invalid_json'
    mock_state_path.unlink = MagicMock()
    
    assert _read_lock() is None
    mock_state_path.unlink.assert_called_once()

@patch("psutil.Process")
def test_process_is_running(mock_process_cls):
    mock_proc = MagicMock()
    mock_process_cls.return_value = mock_proc
    
    # Running
    mock_proc.is_running.return_value = True
    mock_proc.status.return_value = "running"
    assert _process_is_running(123) is True
    
    # Zombie
    mock_proc.status.return_value = "zombie"
    assert _process_is_running(123) is False
    
    # NoSuchProcess
    import psutil
    mock_process_cls.side_effect = psutil.NoSuchProcess(123)
    assert _process_is_running(123) is False

@patch("optimization.dashboard_runner.subprocess.Popen")
@patch("optimization.dashboard_runner._write_lock")
@patch("optimization.dashboard_runner.optuna.load_study")
def test_start_optimization_job(mock_load_study, mock_write_lock, mock_popen, mock_config_path):
    job_cfg = OptimizationJobConfig(
        config_path=mock_config_path,
        n_trials=10,
        study_name="test",
        storage_url="sqlite:///test.db"
    )
    
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.state.is_finished.return_value = True
    mock_study.trials = [mock_trial] * 5
    mock_load_study.return_value = mock_study
    
    mock_process = MagicMock()
    mock_process.pid = 999
    mock_popen.return_value = mock_process
    
    start_optimization_job(job_cfg)
    
    mock_popen.assert_called_once()
    mock_write_lock.assert_called_once()
    payload = mock_write_lock.call_args[0][0]
    assert payload["pid"] == 999
    assert payload["completed_at_start"] == 5

def test_compute_eta():
    assert _compute_eta(None, 10, 1.0) is None
    assert _compute_eta(100, 10, None) is None
    assert _compute_eta(100, 10, 2.0) == 180.0 # (100-10)*2
    assert _compute_eta(100, 110, 2.0) == 0.0

@patch("optimization.dashboard_runner._read_lock")
@patch("optimization.dashboard_runner._process_is_running")
@patch("optimization.dashboard_runner._load_study")
def test_get_optimization_status(mock_load_study, mock_process_running, mock_read_lock, mock_config_path):
    job_cfg = OptimizationJobConfig(
        config_path=mock_config_path,
        n_trials=100
    )
    
    # Case 1: Running
    mock_read_lock.return_value = {"pid": 123, "planned_n_trials": 100}
    mock_process_running.return_value = True
    
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.state.is_finished.return_value = True
    mock_trial.datetime_start = datetime.now() - timedelta(seconds=10)
    mock_trial.datetime_complete = datetime.now()
    mock_study.trials = [mock_trial] * 20 # 20 trials
    mock_load_study.return_value = mock_study
    
    status = get_optimization_status(job_cfg)
    assert status.status == "running"
    assert status.n_trials_completed == 20
    assert abs(status.avg_trial_duration - 10.0) < 1.0
    
    # Case 2: Done
    mock_process_running.return_value = False
    mock_study.trials = [mock_trial] * 100
    status = get_optimization_status(job_cfg)
    assert status.status == "done"
    
    # Case 3: Failed (no completed trials, lock exists, process dead)
    mock_process_running.return_value = False
    mock_study.trials = []  # No trials completed
    mock_read_lock.return_value = {"pid": 123, "planned_n_trials": 100}  # Lock exists
    status = get_optimization_status(job_cfg)
    assert status.status == "failed"
    assert status.error_message is not None

@patch("optimization.dashboard_runner._read_lock")
@patch("optimization.dashboard_runner._process_is_running")
@patch("psutil.Process")
@patch("optimization.dashboard_runner.STATE_PATH")
def test_reset_job_state(mock_state_path, mock_psutil_proc, mock_process_running, mock_read_lock):
    mock_read_lock.return_value = {"pid": 123}
    mock_process_running.return_value = True
    
    mock_proc_instance = MagicMock()
    mock_proc_instance.cmdline.return_value = ["python", "run_optimization.py"]
    mock_psutil_proc.return_value = mock_proc_instance
    
    mock_state_path.exists.return_value = True
    
    reset_job_state()
    
    mock_proc_instance.terminate.assert_called_once()
    mock_state_path.unlink.assert_called_once()

@patch("optimization.dashboard_runner.load_config")
@patch("optimization.dashboard_runner.discover_strategies")
@patch("optimization.dashboard_runner.resolve_strategy")
@patch("optimization.dashboard_runner.get_strategy_defaults")
@patch("optimization.dashboard_runner.merge_params")
@patch("optimization.dashboard_runner._load_data_frames")
@patch("optimization.dashboard_runner._build_engine")
@patch("optimization.dashboard_runner.normalize_weights")
@patch("optimization.dashboard_runner.aggregate_weighted_returns")
@patch("optimization.dashboard_runner.compute_portfolio_metrics")
@patch("reports.report_generator.generate_report")
def test_generate_best_trial_report(mock_gen_report, mock_metrics, mock_agg_returns, mock_norm_weights, 
                                  mock_build_engine, mock_load_dfs, mock_merge, mock_defaults, 
                                  mock_resolve, mock_discover, mock_load_config, mock_config_path, mock_config_data):
    
    mock_load_config.return_value = mock_config_data
    mock_load_dfs.return_value = {"AAPL": pd.DataFrame()}
    
    mock_engine = MagicMock()
    mock_strat = MagicMock()
    mock_strat.analyzers.timereturns.get_analysis.return_value = {datetime.now(): 0.1}
    mock_strat.analyzers.tradelist.get_analysis.return_value = []
    mock_engine.run.return_value = [mock_strat]
    mock_build_engine.return_value = (mock_engine, 10000.0)
    
    mock_agg_returns.return_value = pd.Series([0.1], index=[datetime.now()])
    mock_metrics.return_value = ({}, pd.Series(), pd.Series(), pd.Series())
    
    best_params = {"param1": 10}
    
    report_path = generate_best_trial_report(mock_config_path, best_params)
    
    assert str(report_path).endswith(".html")
    mock_gen_report.assert_called_once()

@patch("optimization.dashboard_runner.load_config")
@patch("optimization.dashboard_runner.subprocess.Popen")
def test_start_overfitting_checks(mock_popen, mock_load_config, mock_config_path, mock_config_data):
    mock_load_config.return_value = mock_config_data
    
    # Mock existence of best_params_path
    with patch("pathlib.Path.exists", return_value=True):
        start_overfitting_checks(mock_config_path, use_best_params=True)
        
    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]
    assert "run_overfitting.py" in str(cmd)
    assert "--use-best-params" in cmd

@patch("optimization.dashboard_runner.load_config")
def test_locate_overfitting_index(mock_load_config, mock_config_path, mock_config_data):
    mock_load_config.return_value = mock_config_data
    
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.iterdir") as mock_iterdir:
        
        # Create a mock for the directory
        mock_dir1 = MagicMock()
        mock_dir1.name = "20230101_120000"
        mock_dir1.is_dir.return_value = True
        
        # Create a mock for the index.html path
        mock_index_path = MagicMock()
        mock_index_path.name = "index.html"
        mock_index_path.exists.return_value = True
        
        # Make the division operation return the index path mock
        mock_dir1.__truediv__.return_value = mock_index_path
        
        mock_iterdir.return_value = [mock_dir1]
        
        result = locate_overfitting_index(mock_config_path)
        assert result is not None
        assert result.name == "index.html"
