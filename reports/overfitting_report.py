"""Génération des rapports HTML spécifiques aux analyses d'overfitting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

try:  # pragma: no cover - dépendance optionnelle
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    go = None  # type: ignore

from utils.logger import setup_logger


logger = setup_logger(__name__)

_BASE_STYLE = """
html, body {
    margin: 0;
    padding: 0;
    font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
    background-color: #0b0e11;
    color: #f5f6f7;
}
header {
    background: #111824;
    padding: 24px;
    border-bottom: 1px solid #1e2633;
}
h1, h2, h3 {
    margin: 0 0 12px 0;
    font-weight: 600;
}
.container {
    padding: 28px 32px 42px;
}
.section {
    margin-bottom: 48px;
    border: 1px solid #1f2a3b;
    border-radius: 10px;
    padding: 24px;
    background: #121826;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
table thead {
    background: #1f2a3b;
}
table th, table td {
    padding: 10px;
    border-bottom: 1px solid #1c2534;
    text-align: left;
}
table tr:nth-child(even) {
    background: #151d2b;
}
a {
    color: #33c1ff;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
.cards {
    display: grid;
    gap: 16px;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}
.card {
    border: 1px solid #1f2a3b;
    border-radius: 10px;
    padding: 16px;
    background: #111723;
}
.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    gap: 12px;
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-robust {
    background: #1f6f43;
    color: #e4ffee;
}
.badge-borderline {
    background: #8a6d1d;
    color: #fff4d0;
}
.badge-overfitted {
    background: #7b1e1e;
    color: #ffe4e4;
}
.plot {
    margin-top: 24px;
}
""".strip()


def _write_html(path: Path, title: str, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""
<!doctype html>
<html lang=\"fr\">
  <head>
    <meta charset=\"utf-8\" />
    <title>{title}</title>
    <style>{_BASE_STYLE}</style>
  </head>
  <body>
    {body}
  </body>
</html>
""".strip()
    path.write_text(html, encoding="utf-8")
    return path


def _table_html(df: Optional[pd.DataFrame], *, classes: str = "") -> str:
    if df is None or df.empty:
        return "<p>Aucune donnée disponible.</p>"
    safe_df = df.copy()
    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = pd.to_datetime(safe_df[col]).dt.strftime("%Y-%m-%d")
    return safe_df.to_html(index=False, classes=classes, border=0)


def _plot_wfa_scatter(folds_df: pd.DataFrame, *, include_js: bool = True) -> str:
    if go is None or folds_df.empty:
        return ""
    df = folds_df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["train_sharpe"],
            y=df["test_sharpe"],
            mode="markers+text",
            text=df["fold"],
            textposition="top center",
            marker=dict(size=10, color=df["test_sharpe"] - df["train_sharpe"], colorscale="Bluered"),
            name="Folds",
        )
    )
    min_val = float(min(df["train_sharpe"].min(), df["test_sharpe"].min()) - 0.5)
    max_val = float(max(df["train_sharpe"].max(), df["test_sharpe"].max()) + 0.5)
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#888", dash="dash"),
            name="Train = Test",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        title="Sharpe Train vs Test",
        xaxis_title="Sharpe (Train)",
        yaxis_title="Sharpe (Test)",
        margin=dict(t=50, r=20, b=40, l=50),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)


def _plot_wfa_delta(folds_df: pd.DataFrame) -> str:
    if go is None or folds_df.empty:
        return ""
    df = folds_df.copy()
    df["delta"] = df["test_sharpe"] - df["train_sharpe"]
    colors = ["#2ec4b6" if val >= 0 else "#e63946" for val in df["delta"]]
    fig = go.Figure(
        data=go.Bar(x=df["fold"], y=df["delta"], marker_color=colors, name="Δ Test-Train")
    )
    fig.update_layout(
        template="plotly_dark",
        height=340,
        title="Écart Sharpe (Test - Train)",
        xaxis_title="Fold",
        yaxis_title="Delta Sharpe",
        margin=dict(t=50, r=20, b=40, l=50),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _plot_histogram(
    df: pd.DataFrame,
    column: str,
    *,
    title: str,
    xaxis_title: str,
    include_js: bool,
) -> str:
    if go is None or df.empty or column not in df.columns:
        return ""
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return ""
    fig = go.Figure(
        data=go.Histogram(
            x=series,
            marker_color="#2ec4b6",
            opacity=0.85,
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Fréquence",
        margin=dict(t=50, r=20, b=40, l=50),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)


def _plot_stability_heatmap(
    neighbors_df: pd.DataFrame,
    *,
    include_js: bool,
) -> str:
    if go is None or neighbors_df.empty:
        return ""
    required_cols = {"param_name", "param_value", "relative_sharpe"}
    if not required_cols.issubset(neighbors_df.columns):
        return ""
    subset = neighbors_df[list(required_cols)].dropna()
    if subset.empty:
        return ""
    subset = subset.copy()
    subset["param_value_label"] = subset["param_value"].map(str)

    pivot = subset.pivot_table(
        index="param_name",
        columns="param_value_label",
        values="relative_sharpe",
        aggfunc="mean",
    )
    if pivot.empty:
        return ""

    def _sort_labels(labels: Iterable[str]) -> List[str]:
        try:
            sortable = [(float(label), label) for label in labels]
            sortable.sort(key=lambda item: item[0])
            return [label for _, label in sortable]
        except ValueError:
            return sorted(labels, key=str)

    pivot = pivot.reindex(sorted(pivot.index, key=str))
    sorted_columns = _sort_labels(list(pivot.columns))
    pivot = pivot.reindex(columns=sorted_columns)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="Relative Sharpe"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        title="Heatmap des variations de paramètres",
        xaxis_title="Valeur du paramètre",
        yaxis_title="Paramètre",
        margin=dict(t=60, r=60, b=60, l=80),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)


def render_wfa_report(
    summary_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    *,
    output_path: Path | str,
    title: str = "Walk-Forward Analysis",
) -> Path:
    """Génère un rapport HTML enrichi pour la WFA."""

    out_path = Path(output_path)
    summary_html = _table_html(summary_df)
    folds_html = _table_html(folds_df)
    scatter_html = _plot_wfa_scatter(folds_df, include_js=True)
    delta_html = _plot_wfa_delta(folds_df)

    sections: List[str] = []
    sections.append(
        f"""
        <section class=\"section\">
          <h2>Résumé</h2>
          {summary_html}
        </section>
        """.strip()
    )

    plot_block = ""
    if scatter_html or delta_html:
        plot_block = (
            "<section class=\"section\">"
            "<h2>Visualisations</h2>"
            f"<div class='plot'>{scatter_html}</div>"
            f"<div class='plot'>{delta_html}</div>"
            "</section>"
        )
        sections.append(plot_block)

    sections.append(
        f"""
        <section class=\"section\">
          <h2>Détails des folds</h2>
          {folds_html}
        </section>
        """.strip()
    )

    body = (
        f"<header><h1>{title}</h1><p>Analyse walk-forward des résultats Optuna.</p></header>"
        f"<div class='container'>{''.join(sections)}</div>"
    )
    return _write_html(out_path, title, body)


def render_oos_report(
    summary_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    output_path: Path | str,
    title: str = "Out-of-Sample Testing",
) -> Path:
    out_path = Path(output_path)
    summary_html = _table_html(summary_df)
    windows_html = _table_html(windows_df)
    histogram_html = _plot_histogram(
        windows_df,
        "sharpe_ratio",
        title="Distribution des Sharpe OOS",
        xaxis_title="Sharpe Ratio",
        include_js=True,
    )

    hist_section = (
        "<section class='section'>"
        "<h2>Histogramme Sharpe OOS</h2>"
        f"{histogram_html or '<p>Histogramme indisponible (Plotly absent ou données manquantes).</p>'}"
        "</section>"
    )

    sections = [
        f"<section class='section'><h2>Résumé</h2>{summary_html}</section>",
        hist_section,
        f"<section class='section'><h2>Détails des fenêtres</h2>{windows_html}</section>",
    ]
    body = (
        f"<header><h1>{title}</h1><p>Analyse out-of-sample.</p></header>"
        f"<div class='container'>{''.join(sections)}</div>"
    )
    return _write_html(out_path, title, body)


def render_monte_carlo_report(
    summary_df: pd.DataFrame,
    simulations_df: pd.DataFrame,
    *,
    output_path: Path | str,
    title: str = "Monte Carlo",
) -> Path:
    out_path = Path(output_path)
    summary_html = _table_html(summary_df)
    simulations_html = _table_html(simulations_df)

    plot_sections: List[str] = []
    include_js = True
    for column, label in (
        ("sharpe_ratio", "Histogramme Sharpe"),
        ("max_drawdown", "Histogramme Max Drawdown"),
    ):
        plot_html = _plot_histogram(
            simulations_df,
            column,
            title=label,
            xaxis_title=label.replace("Histogramme ", ""),
            include_js=include_js,
        )
        if plot_html:
            include_js = False
            plot_sections.append(
                f"<div class='plot'><h3>{label}</h3>{plot_html}</div>"
            )

    if plot_sections:
        plots_block = (
            "<section class='section'>"
            "<h2>Distributions simulées</h2>"
            f"{''.join(plot_sections)}"
            "</section>"
        )
    else:
        plots_block = (
            "<section class='section'>"
            "<h2>Distributions simulées</h2>"
            "<p>Visualisations indisponibles (Plotly absent ou données insuffisantes).</p>"
            "</section>"
        )

    sections = [
        f"<section class='section'><h2>Résumé</h2>{summary_html}</section>",
        plots_block,
        f"<section class='section'><h2>Détails des simulations</h2>{simulations_html}</section>",
    ]
    body = (
        f"<header><h1>{title}</h1><p>Analyse Monte Carlo des métriques.</p></header>"
        f"<div class='container'>{''.join(sections)}</div>"
    )
    return _write_html(out_path, title, body)


def render_stability_report(
    summary_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    *,
    output_path: Path | str,
    title: str = "Stability Tests",
) -> Path:
    out_path = Path(output_path)
    summary_html = _table_html(summary_df)
    neighbors_html = _table_html(neighbors_df)
    heatmap_html = _plot_stability_heatmap(neighbors_df, include_js=True)

    heatmap_section = (
        "<section class='section'>"
        "<h2>Heatmap relative_sharpe</h2>"
        f"{heatmap_html or '<p>Heatmap indisponible (Plotly absent ou données manquantes).</p>'}"
        "</section>"
    )

    sections = [
        f"<section class='section'><h2>Résumé</h2>{summary_html}</section>",
        heatmap_section,
        f"<section class='section'><h2>Détails des voisins</h2>{neighbors_html}</section>",
    ]
    body = (
        f"<header><h1>{title}</h1><p>Analyse de sensibilité des paramètres.</p></header>"
        f"<div class='container'>{''.join(sections)}</div>"
    )
    return _write_html(out_path, title, body)


def render_overfitting_index(
    meta: Dict[str, Any],
    sections: Sequence[Dict[str, Any]],
    *,
    output_path: Path | str,
    title: str = "Overfitting Report",
) -> Path:
    out_path = Path(output_path)
    cards_html = []
    status_summary: List[str] = []
    for section in sections:
        link = section.get("path") or "#"
        description = section.get("description", "")
        name = section.get("name", "Section")
        status_value = section.get("status") or section.get("badge")
        badge_html = ""
        if status_value:
            slug = str(status_value).lower().replace(" ", "-")
            label = str(status_value).replace("-", " ").title()
            badge_html = f"<span class='badge badge-{slug}'>{label}</span>"
            status_summary.append(f"{name}: {label}")
        cards_html.append(
            f"""
            <div class=\"card\">
              <div class="card-header">
                <h3><a href='{link}'>{name}</a></h3>
                {badge_html}
              </div>
              <p>{description}</p>
            </div>
            """.strip()
        )

    meta_items = []
    if status_summary:
        global_text = ", ".join(status_summary)
        meta_items.append(
            f"<li><strong>global_summary:</strong> {global_text}</li>"
        )
    for key, value in meta.items():
        if isinstance(value, dict):
            continue
        meta_items.append(f"<li><strong>{key}:</strong> {value}</li>")
    meta_html = "<ul>" + "".join(meta_items) + "</ul>"

    cards_block = "".join(cards_html) or "<p>Aucune section générée pour l'instant.</p>"
    body = (
        f"<header><h1>{title}</h1><p>Vue d'ensemble des analyses d'overfitting.</p></header>"
        f"<div class='container'>"
        f"  <section class='section'><h2>Méta</h2>{meta_html}</section>"
        f"  <section class='section'><h2>Rapports disponibles</h2><div class='cards'>{cards_block}</div></section>"
        f"</div>"
    )
    return _write_html(out_path, title, body)
