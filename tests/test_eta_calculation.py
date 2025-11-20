
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pytest

# We need to mock these modules BEFORE importing dashboard_runner
# because dashboard_runner imports them at top level.
# However, we cannot patch sys.modules at module level because it breaks other tests.
# So we will use a setup where we patch sys.modules and THEN import the module under test.

def test_eta_calculation_moving_average():
    # Define the mocks
    mock_modules = {
        "scripts": MagicMock(),
        "scripts.run_backtest": MagicMock(),
        "scripts.run_optimization": MagicMock(),
        "backtrader": MagicMock(),
        "backtesting": MagicMock(),
        "backtesting.engine": MagicMock(),
        "backtesting.analyzers": MagicMock(),
        "backtesting.portfolio": MagicMock(),
        "utils": MagicMock(),
        "utils.config_loader": MagicMock(),
        "utils.data_manager": MagicMock(),
        "utils.logger": MagicMock(),
        "psutil": MagicMock(),
        "optimization.optuna_optimizer": MagicMock(), # Mock this too to avoid importing real strategies
    }

    with patch.dict(sys.modules, mock_modules):
        # Now we can safely import the module under test
        # We need to reload it if it was already imported to ensure mocks are used
        if "optimization.dashboard_runner" in sys.modules:
            del sys.modules["optimization.dashboard_runner"]
        
        from optimization.dashboard_runner import get_optimization_status, OptimizationJobConfig

        # Create dummy job config
        job_config = OptimizationJobConfig(
            config_path=MagicMock(),
            n_trials=100,
            timeout=None,
            n_jobs=1,
            study_name="test_study",
            storage_url="sqlite:///test.db"
        )

        # Mock optuna study and trials
        with patch("optimization.dashboard_runner._load_study") as mock_load_study, \
             patch("optimization.dashboard_runner._read_lock") as mock_read_lock, \
             patch("optimization.dashboard_runner._process_is_running") as mock_process_running:
            
            mock_read_lock.return_value = {"pid": 12345, "planned_n_trials": 100}
            mock_process_running.return_value = True
            
            mock_study = MagicMock()
            mock_load_study.return_value = mock_study
            
            # Create 30 trials with increasing duration
            # First 10: 10 seconds each
            # Next 20: 60 seconds each
            # Global average would be (10*10 + 20*60) / 30 = 1300 / 30 = 43.33s
            # Moving average (last 20) should be 60s
            
            trials = []
            base_time = datetime.now()
            
            for i in range(30):
                start = base_time + timedelta(minutes=i*2)
                duration = 10 if i < 10 else 60
                end = start + timedelta(seconds=duration)
                
                mock_trial = MagicMock()
                mock_trial.state.is_finished.return_value = True
                mock_trial.datetime_start = start
                mock_trial.datetime_complete = end
                trials.append(mock_trial)
                
            mock_study.trials = trials
            
            status = get_optimization_status(job_config)
            
            # Remaining trials = 100 - 30 = 70
            # Expected ETA with moving average = 70 * 60 = 4200 seconds
            # Expected ETA with global average = 70 * 43.33 = 3033.33 seconds
            
            expected_eta = 70 * 60
            print(f"Calculated ETA: {status.eta_seconds}")
            print(f"Expected ETA: {expected_eta}")
            
            assert abs(status.eta_seconds - expected_eta) < 1.0

if __name__ == "__main__":
    test_eta_calculation_moving_average()
