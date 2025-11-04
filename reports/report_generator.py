from __future__ import annotations

# --- 1. Bibliothèques natives ---
from pathlib import Path
from typing import Dict, Any, Optional

# --- 2. Bibliothèques tierces ---
import pandas as pd
import numpy as np

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except Exception:  # pragma: no cover
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    select_autoescape = None  # type: ignore

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None  # type: ignore

from utils.logger import setup_logger

logger = setup_logger(__name__)


def _ensure_env(templates_dir: Path):
    if Environment is None:
        raise RuntimeError(
            "Jinja2 n'est pas installé. Installez-le pour générer les rapports."
        )
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env


def _plot_equity(equity: pd.Series, title: str = "Equity Curve") -> str:
    if go is None:
        return ""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity")
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        margin=dict(t=60, r=30, b=40, l=60),
        height=420,
    )
    return fig.to_html(full_html=False, include_plotlyjs=True)


def _plot_underwater(underwater: pd.Series, title: str = "Underwater Curve") -> str:
    if go is None:
        return ""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=underwater.index, y=(underwater.values * 100.0), name="Drawdown %")
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis=dict(ticksuffix="%"),
        margin=dict(t=60, r=30, b=40, l=60),
        height=420,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _plot_monthly_heatmap(returns: Optional[pd.Series]) -> str:
    if go is None or returns is None or returns.empty:
        return ""

    try:
        monthly = (1.0 + returns).resample("ME").prod() - 1.0
        if monthly.empty:
            return ""

        df = monthly.to_frame("return")
        df["year"] = df.index.year.astype(str)
        df["month"] = df.index.strftime("%b")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

        pivot = df.pivot(index="year", columns="month", values="return").fillna(0.0)
        pivot = pivot.reindex(columns=month_order, fill_value=0.0)

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values * 100.0,
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale=[[0, "#8B1E3F"], [0.5, "#1D1F2E"], [1, "#2EC4B6"]],
                colorbar=dict(title="Retour %"),
                zmid=0,
            )
        )
        fig.update_layout(
            template="plotly_dark",
            title="Rendements Mensuels",
            xaxis_title="Mois",
            yaxis_title="Année",
            margin=dict(t=60, r=30, b=40, l=60),
            height=420,
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)
    except Exception:
        return ""


def _plot_rolling_metrics(
    log_returns: Optional[pd.Series],
    *,
    periods_per_year: int,
    risk_free_rate: float,
    window: int,
) -> str:
    if go is None or log_returns is None or log_returns.empty:
        return ""

    series = log_returns.dropna()
    if series.empty:
        return ""

    window = max(int(window), 1)
    rf_period = risk_free_rate / periods_per_year

    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std().replace(0, np.nan)

    sharpe = ((rolling_mean - rf_period) / rolling_std) * np.sqrt(periods_per_year)
    volatility = rolling_std * np.sqrt(periods_per_year)

    if sharpe.dropna().empty and volatility.dropna().empty:
        return ""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sharpe.index,
            y=sharpe.values,
            name="Rolling Sharpe",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=volatility.index,
            y=volatility.values,
            name="Rolling Volatilité",
            mode="lines",
            yaxis="y2",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=f"Métriques Rolling ({window} périodes)",
        xaxis_title="Date",
        yaxis=dict(title="Sharpe"),
        yaxis2=dict(title="Volatilité", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=60, r=80, b=40, l=60),
        height=420,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_report(
    meta: Dict[str, Any],
    metrics: Dict[str, Any],
    equity: Optional[pd.Series],
    underwater: Optional[pd.Series],
    trades: Optional[pd.DataFrame],
    out_path: str,
    template: str = "default.html",
    returns: Optional[pd.Series] = None,
    log_returns: Optional[pd.Series] = None,
    analytics_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Génère un rapport HTML de backtest.

    Args:
        meta: Informations générales (stratégie, ticker, période...)
        metrics: Dictionnaire de métriques calculées
        equity: Série de la courbe d'equity
        underwater: Série de drawdown (proportion)
        trades: DataFrame des trades détaillés (peut être None)
        out_path: Chemin de sortie du fichier HTML
        template: Nom du template Jinja2 dans reports/templates/
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    templates_dir = Path("reports/templates")
    templates_dir.mkdir(parents=True, exist_ok=True)

    # Assurer un template par défaut si absent
    default_tpl = templates_dir / "default.html"
    if not default_tpl.exists():
        default_tpl.write_text(
            """
<!doctype html>
<html lang="fr">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Rapport Backtest - {{ meta.strategy_name }}</title>
    <style>
      body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
      h1, h2 { margin: 0.4em 0; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
      table { width: 100%; border-collapse: collapse; }
      th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: right; }
      th { text-align: left; }
      .muted { color: #666; }
      .kpi { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
      .kpi .card { text-align: center; }
    </style>
  </head>
  <body>
    <div class="header">
      <div>
        <h1>Rapport Backtest</h1>
        <div class="meta">
          Stratégie · {{ meta.strategy_name }} &nbsp;&nbsp;|&nbsp;&nbsp;
          Ticker · {{ meta.ticker }} &nbsp;&nbsp;|&nbsp;&nbsp;
          {{ meta.start_date }} → {{ meta.end_date }}
        </div>
      </div>
    </div>

    <div class="summary-banner">
      <div class="summary-card">
        <div class="summary-grid">
          <div class="summary-item">
            <span>Capital Final</span>
            <span>{{ '{:,.2f}'.format(metrics.final_value) if metrics.final_value else '—' }} €</span>
          </div>
          <div class="summary-item">
            <span>P&amp;L</span>
            <span>{{ '{:+.2f}'.format(metrics.pnl or 0.0) }} € ({{ '{:+.2f}%'.format(metrics.pnl_pct or 0.0) }})</span>
          </div>
          <div class="summary-item">
            <span>Total Trades</span>
            <span>{{ metrics.total_trades }}</span>
          </div>
          <div class="summary-item">
            <span>Max Drawdown</span>
            <span>{{ '{:.2f}%'.format((metrics.max_drawdown or 0.0) * 100) }}</span>
          </div>
        </div>
      </div>

      <div class="kpi">
        <div class="card">
          <div class="label">Sharpe</div>
          <div class="value">{{ '%.2f'|format(metrics.sharpe_ratio or 0.0) }}</div>
        </div>
        <div class="card">
          <div class="label">Sortino</div>
          <div class="value">{{ '%.2f'|format(metrics.sortino_ratio or 0.0) }}</div>
        </div>
        <div class="card">
          <div class="label">Calmar</div>
          <div class="value">{{ '%.2f'|format(metrics.calmar_ratio or 0.0) }}</div>
        </div>
      </div>
    </div>

    <div class="grid two">
      <div class="card">
        <h2>Equity Curve</h2>
        {{ plots.equity | safe }}
      </div>
      <div class="card">
        <h2>Underwater Curve</h2>
        {{ plots.underwater | safe }}
      </div>
    </div>

    <div class="grid two">
      <div class="card">
        <h2>Performance</h2>
        <table>
          <tr><th>CAGR</th><td>{{ '%.2f%%'|format((metrics.cagr or 0.0) * 100) }}</td></tr>
          <tr><th>Ulcer Index</th><td>{{ '%.2f'|format(metrics.ulcer_index or 0.0) }}</td></tr>
          <tr><th>Expectancy</th><td>{{ '%.2f'|format(metrics.expectancy or 0.0) }}</td></tr>
        </table>
      </div>
      <div class="card">
        <h2>Trades Overview</h2>
        <table>
          <tr>
            <th>Total</th><td>{{ metrics.total_trades }}</td>
            <th>Win Rate</th><td>{{ '%.2f%%'|format(metrics.win_rate or 0.0) }}</td>
          </tr>
          <tr>
            <th>Profit Factor</th><td>{{ '%.2f'|format(metrics.profit_factor or 0.0) }}</td>
            <th>Payoff Ratio</th><td>{{ '%.2f'|format(metrics.payoff_ratio or 0.0) }}</td>
          </tr>
        </table>
      </div>
    </div>

    <div class="grid two">
      <div class="card">
        <h2>Métriques Rolling</h2>
        {{ plots.rolling_metrics | safe }}
      </div>
      <div class="card">
        <h2>Heatmap Rendements Mensuels</h2>
        {{ plots.monthly_heatmap | safe }}
      </div>
    </div>

    {% if trades is not none and trades|length > 0 %}
    <div class="card" style="margin-top: 24px;">
      <h2>Tableau des Trades</h2>
      <table>
        <tr>
          <th>Open</th>
          <th>Close</th>
          <th>Side</th>
          <th>Size</th>
          <th>Entry</th>
          <th>Exit</th>
          <th>Net PnL</th>
          <th>Ret %</th>
          <th>Dur (j)</th>
        </tr>
        {% for _, row in trades.iterrows() %}
        <tr>
          <td>{{ row.entry_dt }}</td>
          <td>{{ row.exit_dt }}</td>
          <td>{{ 'Long' if row.is_long else 'Short' }}</td>
          <td>{{ (row.size|abs) if row.size is not none else 0 }}</td>
          <td>{{ '%.2f'|format(row.entry_price or 0.0) }}</td>
          <td>{{ '%.2f'|format(row.exit_price or 0.0) }}</td>
          <td>{{ '%.2f'|format(row.net_pnl or 0.0) }}</td>
          <td>{{ '%.2f%%'|format(row.ret_pct or 0.0) }}</td>
          <td>{{ row.duration_days|int if row.duration_days is not none else '—' }}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
    {% endif %}

  </body>
  </html>
            """,
            encoding="utf-8",
        )

    env = _ensure_env(templates_dir)
    tpl = env.get_template(template)

    # Préparer dict template-friendly
    trades_tbl = None
    if trades is not None and not trades.empty:
        # Convertir index/colonnes non sérialisables
        trades_tbl = trades.copy()

    periods_per_year = 252
    risk_free_rate = 0.0
    rolling_window = 63
    if analytics_config:
        periods_per_year = analytics_config.get("periods_per_year", periods_per_year)
        risk_free_rate = analytics_config.get("risk_free_rate", risk_free_rate)
        rolling_window = analytics_config.get("rolling_window", rolling_window)

    plots = {
        "equity": "",
        "underwater": "",
        "monthly_heatmap": "",
        "rolling_metrics": "",
    }
    if equity is not None and not equity.empty:
        plots["equity"] = _plot_equity(equity)
    if underwater is not None and not underwater.empty:
        plots["underwater"] = _plot_underwater(underwater)
    if returns is not None and not returns.empty:
        plots["monthly_heatmap"] = _plot_monthly_heatmap(returns)
    if log_returns is not None and not log_returns.empty:
        plots["rolling_metrics"] = _plot_rolling_metrics(
            log_returns,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            window=rolling_window,
        )

    html = tpl.render(meta=meta, metrics=metrics, plots=plots, trades=trades_tbl)
    out_file.write_text(html, encoding="utf-8")
    logger.info(f"Rapport généré: {out_file}")
    return str(out_file)
