import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

# Mock plotly before it's imported by the module under test.
# This prevents ImportError if plotly is not installed, allowing tests to run.
if "plotly" not in sys.modules:
    sys.modules["plotly"] = Mock()
    sys.modules["plotly.graph_objects"] = Mock()

from reports import overfitting_report


@pytest.fixture
def sample_summary_df():
    return pd.DataFrame({
        "metric": ["Sharpe Ratio", "Max Drawdown"],
        "value": [1.5, 0.2],
    })

@pytest.fixture
def sample_folds_df():
    return pd.DataFrame({
        "fold": [1, 2, 3],
        "train_sharpe": [1.2, 1.5, 1.3],
        "test_sharpe": [1.0, 1.3, 1.1],
    })

@pytest.fixture
def sample_windows_df():
    return pd.DataFrame({
        "window": [1, 2, 3],
        "sharpe_ratio": [0.9, 1.1, 1.0],
    })

@pytest.fixture
def sample_simulations_df():
    return pd.DataFrame({
        "simulation": [1, 2, 3],
        "sharpe_ratio": [1.4, 1.5, 1.6],
        "max_drawdown": [0.15, 0.18, 0.12],
    })

@pytest.fixture
def sample_neighbors_df():
    return pd.DataFrame({
        "param_name": ["A", "A", "B"],
        "param_value": [10, 20, 5],
        "relative_sharpe": [0.95, 0.98, 1.02],
    })

def test_render_wfa_report(tmp_path, sample_summary_df, sample_folds_df):
    """Test WFA report generation with valid data."""
    output_file = tmp_path / "wfa_report.html"
    result_path = overfitting_report.render_wfa_report(
        summary_df=sample_summary_df,
        folds_df=sample_folds_df,
        output_path=output_file,
        title="Test WFA Report"
    )
    assert result_path == output_file
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "Test WFA Report" in content
    assert "Sharpe Ratio" in content
    assert "1.5" in content

def test_render_wfa_report_empty(tmp_path):
    """Test WFA report with empty data."""
    output_file = tmp_path / "wfa_report_empty.html"
    result_path = overfitting_report.render_wfa_report(
        summary_df=pd.DataFrame(),
        folds_df=pd.DataFrame(),
        output_path=output_file
    )
    assert result_path.exists()
    content = result_path.read_text(encoding="utf-8")
    assert "Aucune donnée disponible." in content

def test_render_wfa_report_no_plotly(tmp_path, sample_summary_df, sample_folds_df, monkeypatch):
    """Test WFA report generation when plotly is not available."""
    monkeypatch.setattr(overfitting_report, "go", None)
    output_file = tmp_path / "wfa_report_no_plotly.html"
    overfitting_report.render_wfa_report(
        summary_df=sample_summary_df,
        folds_df=sample_folds_df,
        output_path=output_file
    )
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "Visualisations" not in content  # The whole section should be missing

def test_render_oos_report(tmp_path, sample_summary_df, sample_windows_df):
    """Test OOS report generation."""
    output_file = tmp_path / "oos_report.html"
    result_path = overfitting_report.render_oos_report(
        summary_df=sample_summary_df,
        windows_df=sample_windows_df,
        output_path=output_file,
        title="Test OOS Report"
    )
    assert result_path.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "Test OOS Report" in content
    assert "0.9" in content # from sharpe_ratio

def test_render_oos_report_no_plotly(tmp_path, sample_summary_df, sample_windows_df, monkeypatch):
    """Test OOS report when plotly is missing."""
    monkeypatch.setattr(overfitting_report, "go", None)
    output_file = tmp_path / "oos_report_no_plotly.html"
    result_path = overfitting_report.render_oos_report(
        summary_df=sample_summary_df,
        windows_df=sample_windows_df,
        output_path=output_file
    )
    assert result_path.exists()
    content = result_path.read_text(encoding="utf-8")
    assert "Histogramme indisponible" in content

def test_render_monte_carlo_report(tmp_path, sample_summary_df, sample_simulations_df):
    """Test Monte Carlo report generation."""
    output_file = tmp_path / "mc_report.html"
    result_path = overfitting_report.render_monte_carlo_report(
        summary_df=sample_summary_df,
        simulations_df=sample_simulations_df,
        output_path=output_file
    )
    assert result_path.exists()
    content = result_path.read_text(encoding="utf-8")
    assert "Monte Carlo" in content
    assert "0.18" in content # max_drawdown

def test_render_stability_report(tmp_path, sample_summary_df, sample_neighbors_df):
    """Test stability report generation."""
    output_file = tmp_path / "stability_report.html"
    result_path = overfitting_report.render_stability_report(
        summary_df=sample_summary_df,
        neighbors_df=sample_neighbors_df,
        output_path=output_file
    )
    assert result_path.exists()
    content = result_path.read_text(encoding="utf-8")
    assert "Stability Tests" in content
    assert "relative_sharpe" in content

def test_render_overfitting_index(tmp_path):
    """Test the main index report generation."""
    output_file = tmp_path / "index.html"
    meta = {"strategy": "TestStrategy", "version": "1.0"}
    sections = [
        {"name": "WFA", "path": "wfa.html", "description": "Walk-forward analysis.", "status": "Robust"},
        {"name": "OOS", "path": "oos.html", "description": "Out-of-sample test.", "status": "Overfitted"},
    ]
    result_path = overfitting_report.render_overfitting_index(
        meta=meta,
        sections=sections,
        output_path=output_file
    )
    assert result_path.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "Overfitting Report" in content
    assert "TestStrategy" in content
    assert "wfa.html" in content
    assert "badge-robust" in content
    assert "badge-overfitted" in content

def test_render_overfitting_index_empty(tmp_path):
    """Test the main index report with no sections."""
    output_file = tmp_path / "index_empty.html"
    result_path = overfitting_report.render_overfitting_index(
        meta={},
        sections=[],
        output_path=output_file
    )
    assert result_path.exists()
    content = result_path.read_text(encoding="utf-8")
    assert "Aucune section générée pour l'instant." in content
