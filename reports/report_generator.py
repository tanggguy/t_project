from __future__ import annotations

# --- 1. Bibliothèques natives ---
from pathlib import Path
from typing import Dict, Any, Optional

# --- 2. Bibliothèques tierces ---
import pandas as pd
import numpy as np


from jinja2 import Environment, FileSystemLoader, select_autoescape


import plotly.graph_objects as go

# --- 3. Bibliothèques internes ---

from utils.logger import setup_logger

logger = setup_logger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_number(value: Any, digits: int = 2) -> str:
    val = _safe_float(value, default=float("nan"))
    if np.isnan(val):
        return "â€”"
    return f"{val:.{digits}f}"


def _fmt_pct(value: Any, scale: float = 100.0, digits: int = 2) -> str:
    val = _safe_float(value, default=float("nan"))
    if np.isnan(val):
        return "â€”"
    return f"{val * scale:.{digits}f}%"


def _fmt_currency(value: Any, currency_symbol: str = "â‚¬") -> str:
    val = _safe_float(value, default=float("nan"))
    if np.isnan(val):
        return "â€”"
    return f"{val:,.2f} {currency_symbol}"


def _fix_mojibake(text: str) -> str:
    """
    Corrige les accents mal encodés de type 'nÃ©gatif' -> 'négatif'.
    """
    try:
        return text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text


def _build_interpretation(metrics: Dict[str, Any]) -> Dict[str, str]:
    sharpe = _safe_float(metrics.get("sharpe_ratio"), 0.0)
    max_dd_pct = _safe_float(metrics.get("max_drawdown"), 0.0) * 100.0
    win_rate = _safe_float(metrics.get("win_rate"), 0.0)
    payoff = _safe_float(metrics.get("payoff_ratio"), 0.0)

    fr_parts: list[str] = []
    en_parts: list[str] = []

    # Sharpe
    if sharpe >= 2.0:
        fr_parts.append(
            "Le ratio de Sharpe est excellent, la stratÃ©gie offre un trÃ¨s bon profil rendement/risque."
        )
        en_parts.append(
            "The Sharpe ratio is excellent, the strategy shows a very strong risk-adjusted return."
        )
    elif sharpe >= 1.0:
        fr_parts.append(
            "Le ratio de Sharpe est correct, avec un profil rendement/risque globalement acceptable."
        )
        en_parts.append(
            "The Sharpe ratio is decent, with an overall acceptable risk-adjusted profile."
        )
    elif sharpe > 0.0:
        fr_parts.append(
            "Le ratio de Sharpe est faible, les gains restent modestes par rapport au risque pris."
        )
        en_parts.append(
            "The Sharpe ratio is low, returns look modest relative to the risk taken."
        )
    else:
        fr_parts.append(
            "Le ratio de Sharpe est nÃ©gatif, la stratÃ©gie ne compense pas le risque pris."
        )
        en_parts.append(
            "The Sharpe ratio is negative, the strategy does not compensate for the risk taken."
        )

    # Drawdown
    if max_dd_pct <= 10:
        fr_parts.append(
            f"Le drawdown maximal reste contenu (â‰ˆ {max_dd_pct:.1f}%), indiquant une bonne maÃ®trise du risque."
        )
        en_parts.append(
            f"Maximum drawdown is contained (â‰ˆ {max_dd_pct:.1f}%), suggesting good downside control."
        )
    elif max_dd_pct <= 25:
        fr_parts.append(
            f"Le drawdown maximal est modÃ©rÃ© (â‰ˆ {max_dd_pct:.1f}%), acceptable selon le contexte."
        )
        en_parts.append(
            f"Maximum drawdown is moderate (â‰ˆ {max_dd_pct:.1f}%), which may be acceptable depending on the context."
        )
    else:
        fr_parts.append(
            f"Le drawdown maximal est Ã©levÃ© (â‰ˆ {max_dd_pct:.1f}%), la stratÃ©gie peut Ãªtre difficile Ã  supporter."
        )
        en_parts.append(
            f"Maximum drawdown is high (â‰ˆ {max_dd_pct:.1f}%), the strategy may be hard to tolerate in practice."
        )

    # Win rate & payoff
    if win_rate >= 60.0:
        if payoff >= 1.0:
            fr_parts.append(
                f"Le taux de rÃ©ussite est Ã©levÃ© (~{win_rate:.1f}%) avec un payoff satisfaisant (â‰ˆ {payoff:.2f})."
            )
            en_parts.append(
                f"Win rate is high (~{win_rate:.1f}%) with a satisfactory payoff (â‰ˆ {payoff:.2f})."
            )
        else:
            fr_parts.append(
                f"Le taux de rÃ©ussite est Ã©levÃ© (~{win_rate:.1f}%) mais le payoff est limitÃ© (â‰ˆ {payoff:.2f})."
            )
            en_parts.append(
                f"Win rate is high (~{win_rate:.1f}%) but the payoff is limited (â‰ˆ {payoff:.2f})."
            )
    elif win_rate >= 45.0:
        fr_parts.append(
            f"Le taux de rÃ©ussite est moyen (~{win_rate:.1f}%) ; le couple win rate / payoff doit Ãªtre suivi dans le temps."
        )
        en_parts.append(
            f"Win rate is medium (~{win_rate:.1f}%); the win-rate / payoff combination should be monitored over time."
        )
    else:
        if payoff > 1.5:
            fr_parts.append(
                f"Le taux de rÃ©ussite est faible (~{win_rate:.1f}%), mais le payoff (â‰ˆ {payoff:.2f}) compense en partie."
            )
            en_parts.append(
                f"Win rate is low (~{win_rate:.1f}%), but the payoff (â‰ˆ {payoff:.2f}) partly compensates."
            )
        else:
            fr_parts.append(
                f"Le taux de rÃ©ussite est faible (~{win_rate:.1f}%) et le payoff (â‰ˆ {payoff:.2f}) reste modeste."
            )
            en_parts.append(
                f"Win rate is low (~{win_rate:.1f}%) and the payoff (â‰ˆ {payoff:.2f}) is modest."
            )

    fr_text = _fix_mojibake(" ".join(fr_parts))
    en_text = _fix_mojibake(" ".join(en_parts)) if en_parts else ""
    return {
        "fr": fr_text,
        "en": en_text,
    }


def _build_checklist(metrics: Dict[str, Any]) -> list[Dict[str, Any]]:
    sharpe = _safe_float(metrics.get('sharpe_ratio'), default=float('nan'))
    max_dd = _safe_float(metrics.get('max_drawdown'), default=float('nan'))
    profit_factor = _safe_float(metrics.get('profit_factor'), default=float('nan'))
    win_rate = _safe_float(metrics.get('win_rate'), default=float('nan'))
    payoff = _safe_float(metrics.get('payoff_ratio'), default=float('nan'))
    cagr = _safe_float(metrics.get('cagr'), default=float('nan'))

    def _fmt(value: float, fmt: str) -> str:
        if np.isnan(value):
            return 'N/A'
        return fmt.format(value)

    return [
        {
            'label': 'Sharpe >= 1.0',
            'passed': not np.isnan(sharpe) and sharpe >= 1.0,
            'value': _fmt(sharpe, '{:.2f}'),
        },
        {
            'label': 'Max DD <= 20%',
            'passed': not np.isnan(max_dd) and max_dd <= 0.20,
            'value': _fmt(max_dd * 100 if not np.isnan(max_dd) else np.nan, '{:.1f}%'),
        },
        {
            'label': 'Profit factor >= 1.3',
            'passed': not np.isnan(profit_factor) and profit_factor >= 1.3,
            'value': _fmt(profit_factor, '{:.2f}'),
        },
        {
            'label': 'Win rate >= 45% ou Payoff > 1.5',
            'passed': ((not np.isnan(win_rate) and win_rate >= 45.0) or (not np.isnan(payoff) and payoff > 1.5)),
            'value': f"WR {_fmt(win_rate, '{:.1f}%')} / Payoff {_fmt(payoff, '{:.2f}')}",
        },
        {
            'label': 'CAGR >= 10%',
            'passed': not np.isnan(cagr) and cagr >= 0.10,
            'value': _fmt(cagr * 100 if not np.isnan(cagr) else np.nan, '{:.1f}%'),
        },
    ]

def _ensure_env(templates_dir: Path):
    if Environment is None:
        raise RuntimeError(
            "Jinja2 n'est pas installé. Installez-le pour générer les rapports."
        )
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    # Filtres de formatage simples pour garder les templates lisibles
    env.filters["fmt_number"] = _fmt_number
    env.filters["fmt_pct"] = _fmt_pct
    env.filters["fmt_currency"] = _fmt_currency
    return env


def _plot_equity(
    equity: pd.Series,
    title: str = "Equity Curve",
    benchmark_equity: Optional[pd.Series] = None,
    benchmark_name: str = "Benchmark",
) -> str:
    if go is None:
        return ""
    eq_series = equity.sort_index()
    bench_series = None
    if benchmark_equity is not None and not benchmark_equity.empty:
        bench_series = benchmark_equity.sort_index()
        combined_idx = eq_series.index.union(bench_series.index)
        eq_series = (
            eq_series.reindex(combined_idx).ffill().bfill().replace([np.inf, -np.inf], np.nan)
        )
        bench_series = (
            bench_series.reindex(combined_idx).ffill().bfill().replace([np.inf, -np.inf], np.nan)
        )
        eq_series = eq_series.dropna()
        bench_series = bench_series.dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eq_series.index,
            y=eq_series.values,
            mode="lines",
            name="Stratégie",
            line=dict(color="#38bdf8", width=2),
        )
    )
    if bench_series is not None and not bench_series.empty:
        fig.add_trace(
            go.Scatter(
                x=bench_series.index,
                y=bench_series.values,
                mode="lines",
                name=benchmark_name or "Benchmark",
                line=dict(color="#f97316", width=2, dash="dash"),
                opacity=0.9,
            )
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


def _plot_returns_hist(
    period_returns: Optional[pd.Series] = None,
    trade_returns_pct: Optional[pd.Series] = None,
    title: str = "Distribution des rendements",
) -> str:
    if go is None:
        return ""

    series: Optional[pd.Series] = None
    if trade_returns_pct is not None and not trade_returns_pct.empty:
        try:
            series = pd.to_numeric(trade_returns_pct, errors="coerce") / 100.0
        except Exception:
            series = None
    if series is None and period_returns is not None and not period_returns.empty:
        try:
            series = pd.to_numeric(period_returns, errors="coerce")
        except Exception:
            series = None

    if series is None:
        return ""

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return ""

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=series * 100.0,
            nbinsx=40,
            marker=dict(color="#38bdf8"),
            opacity=0.85,
            name="Rendements",
        )
    )

    try:
        mean_val = float(series.mean() * 100.0)
        fig.add_vline(
            x=mean_val,
            line_color="#f97316",
            line_dash="dash",
            annotation_text="Moyenne",
            annotation_position="top left",
        )
    except Exception:
        pass

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Rendement (%)",
        yaxis_title="FrÃ©quence",
        margin=dict(t=60, r=30, b=40, l=60),
        height=420,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _compute_exposure_series(
    trades: Optional[pd.DataFrame], index: Optional[pd.Index]
) -> Optional[pd.Series]:
    if trades is None or trades.empty or index is None or len(index) == 0:
        return None
    if "size" not in trades.columns:
        return None

    try:
        df = trades.copy()
        df["entry_dt"] = pd.to_datetime(df.get("entry_dt"), errors="coerce")
        df["exit_dt"] = pd.to_datetime(df.get("exit_dt"), errors="coerce")
    except Exception:
        return None

    idx = pd.to_datetime(index, errors="coerce")
    idx = idx.dropna()
    if len(idx) == 0:
        return None

    exposure = pd.Series(0.0, index=idx)

    for _, row in df.iterrows():
        size = row.get("size")
        entry_dt = row.get("entry_dt")
        exit_dt = row.get("exit_dt")
        if size in (None, 0) or pd.isna(entry_dt):
            continue
        try:
            entry_dt = pd.to_datetime(entry_dt)
            if pd.isna(exit_dt):
                exit_dt = idx[-1]
            else:
                exit_dt = pd.to_datetime(exit_dt)
        except Exception:
            continue

        mask = (idx >= entry_dt) & (idx <= exit_dt)
        if not mask.any():
            continue
        try:
            exposure.loc[mask] += float(size)
        except Exception:
            continue

    exposure = exposure.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if exposure.abs().sum() == 0.0:
        return None
    return exposure


def _plot_exposure(exposure: Optional[pd.Series]) -> str:
    if go is None or exposure is None or exposure.empty:
        return ""

    exposure = exposure.sort_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=exposure.index,
            y=exposure.values,
            mode="lines",
            name="Taille de position",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Exposition / Taille de position",
        xaxis_title="Date",
        yaxis_title="Taille (signÃ©e)",
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
        month_order = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
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
    benchmark: Optional[Dict[str, Any]] = None,
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

    benchmark_context: Optional[Dict[str, Any]] = None
    benchmark_equity: Optional[pd.Series] = None
    benchmark_name = "Benchmark"
    if isinstance(benchmark, dict):
        benchmark_name = benchmark.get("name", benchmark_name)
        metrics_dict = benchmark.get("metrics") or {}
        if not isinstance(metrics_dict, dict):
            try:
                metrics_dict = dict(metrics_dict)
            except Exception:
                metrics_dict = {}
        benchmark_context = {"name": benchmark_name, "metrics": metrics_dict}
        equity_candidate = benchmark.get("equity")
        if isinstance(equity_candidate, pd.Series) and not equity_candidate.empty:
            benchmark_equity = equity_candidate.copy()
    elif benchmark:
        benchmark_context = {"name": benchmark_name, "metrics": {}}

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
        "returns_hist": "",
        "exposure": "",
    }
    if equity is not None and not equity.empty:
        plots["equity"] = _plot_equity(
            equity, benchmark_equity=benchmark_equity, benchmark_name=benchmark_name
        )
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

    # Distribution des rendements (par pÃ©riode ou par trade)
    if returns is not None and not returns.empty:
        plots["returns_hist"] = _plot_returns_hist(period_returns=returns)
    elif trades_tbl is not None and not trades_tbl.empty and "ret_pct" in trades_tbl.columns:
        trade_rets = pd.to_numeric(trades_tbl["ret_pct"], errors="coerce")
        plots["returns_hist"] = _plot_returns_hist(trade_returns_pct=trade_rets)

    # Exposition / taille de position dans le temps (si possible)
    if trades_tbl is not None and equity is not None:
        try:
            exposure_series = _compute_exposure_series(trades_tbl, equity.index)
        except Exception:
            exposure_series = None
        if exposure_series is not None and not exposure_series.empty:
            plots["exposure"] = _plot_exposure(exposure_series)

    interpretation = _build_interpretation(metrics or {})
    checklist = _build_checklist(metrics or {})

    html = tpl.render(
        meta=meta,
        metrics=metrics,
        plots=plots,
        trades=trades_tbl,
        benchmark=benchmark_context,
        interpretation=interpretation,
        checklist=checklist,
    )
    out_file.write_text(html, encoding="utf-8")
    logger.info(f"Rapport généré: {out_file}")
    return str(out_file)
