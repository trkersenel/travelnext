"""Exploratory data analysis over the destination catalog and interactions.

    python -m src.analysis.eda

Writes static figures to ``reports/figures/`` and a text summary to
``reports/eda_summary.md``. Figures use matplotlib (PNG) plus one interactive
Plotly map saved as self-contained HTML, so nothing here needs a browser plugin
or a paid tile service.

The analysis deliberately separates the two halves of the dataset: the
destination sections describe real measured data, while the interaction
sections describe the synthetic generator's output and are labelled as such in
every title.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless: no display needed in Docker or CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data.dataset import TravelDataset, load_dataset
from src.preprocessing.features import PROFILE_CATEGORIES
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

_FIGSIZE = (10, 6)
_DPI = 110


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", path.name)


def plot_geographic_distribution(destinations: pd.DataFrame, figures: Path) -> None:
    """Scatter every destination on a lat/lon plane, coloured by continent."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for continent, group in destinations.groupby("continent"):
        ax.scatter(
            group["longitude"],
            group["latitude"],
            s=8 + 40 * group["popularity_score"],
            alpha=0.65,
            label=f"{continent} ({len(group)})",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Destination catalog by continent (marker size = popularity proxy)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    _save(fig, figures / "geographic_distribution.png")


def plot_interactive_map(destinations: pd.DataFrame, figures: Path) -> None:
    """Self-contained Plotly map of the catalog (free OpenStreetMap basemap)."""
    try:
        import plotly.express as px
    except ImportError:  # pragma: no cover - plotly is a declared dependency
        LOGGER.warning("Plotly not installed; skipping the interactive map")
        return

    figure = px.scatter_geo(
        destinations,
        lat="latitude",
        lon="longitude",
        color="continent",
        size="popularity_score",
        hover_name="city",
        hover_data={"country": True, "cost_category": True, "popularity_score": ":.2f"},
        title="TravelNext destination catalog",
        projection="natural earth",
    )
    output = figures / "destination_map.html"
    figure.write_html(output, include_plotlyjs="cdn")
    LOGGER.info("Wrote %s", output.name)


def plot_popularity_distribution(destinations: pd.DataFrame, figures: Path) -> None:
    """Show the long tail of the popularity proxy on a log scale."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    views = destinations["pageviews_monthly_mean"].clip(lower=1)
    axes[0].hist(np.log10(views), bins=40, color="#4c72b0", edgecolor="white")
    axes[0].set_xlabel("log10(mean monthly Wikipedia pageviews)")
    axes[0].set_ylabel("Destinations")
    axes[0].set_title("Popularity proxy is heavily long-tailed")

    ordered = np.sort(views)[::-1]
    axes[1].plot(np.arange(1, len(ordered) + 1), np.cumsum(ordered) / ordered.sum())
    axes[1].set_xlabel("Destinations ranked by popularity")
    axes[1].set_ylabel("Cumulative share of pageviews")
    axes[1].set_title("Concentration of attention")
    axes[1].grid(alpha=0.3)
    _save(fig, figures / "popularity_distribution.png")


def plot_country_distribution(destinations: pd.DataFrame, figures: Path, top: int = 25) -> None:
    """Bar chart of the countries contributing the most destinations."""
    counts = destinations["country"].value_counts().head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(counts.index, counts.to_numpy(), color="#55a868")
    ax.set_xlabel("Destinations in catalog")
    ax.set_title(f"Top {top} countries by destination count")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, figures / "country_distribution.png")


def plot_attribute_correlations(destinations: pd.DataFrame, figures: Path) -> None:
    """Correlation heatmap of the measured OSM attribute scores."""
    columns = [f"score_{category}" for category in PROFILE_CATEGORIES]
    matrix = destinations[columns].dropna().corr()
    labels = [c.replace("score_", "") for c in columns]

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Correlation between measured destination attributes")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, f"{matrix.iat[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="black" if abs(matrix.iat[i, j]) < 0.6 else "white",
            )
    fig.colorbar(image, ax=ax, shrink=0.8)
    _save(fig, figures / "attribute_correlations.png")


def plot_cost_and_climate(destinations: pd.DataFrame, figures: Path) -> None:
    """Cost bands per continent, and the seasonal comfort curve by latitude."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    crosstab = pd.crosstab(destinations["continent"], destinations["cost_category"])
    crosstab.plot(kind="bar", stacked=True, ax=axes[0], colormap="viridis")
    axes[0].set_title("Cost band by continent (World Bank proxy)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Destinations")
    axes[0].tick_params(axis="x", rotation=30)

    northern = destinations[destinations["latitude"] > 20]
    southern = destinations[destinations["latitude"] < -20]
    months = range(1, 13)
    for label, subset, colour in (
        ("Northern (lat > 20°)", northern, "#c44e52"),
        ("Southern (lat < -20°)", southern, "#4c72b0"),
    ):
        if subset.empty:
            continue
        means = [subset[f"season_score_m{m:02d}"].mean() for m in months]
        axes[1].plot(list(months), means, marker="o", label=label, color=colour)
    axes[1].set_xticks(list(months))
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Mean season comfort score")
    axes[1].set_title("Seasonality is inverted across hemispheres")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    _save(fig, figures / "cost_and_climate.png")


def plot_interaction_distributions(
    interactions: pd.DataFrame, destinations: pd.DataFrame, figures: Path
) -> None:
    """Trips per user and the popularity bias present in the synthetic data."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    trips = interactions.groupby("user_id").size()
    axes[0].hist(trips, bins=range(trips.min(), trips.max() + 2), color="#8172b2", edgecolor="white")
    axes[0].set_xlabel("Trips per user")
    axes[0].set_ylabel("Users")
    axes[0].set_title("SYNTHETIC: travel history length")

    counts = interactions["destination_id"].value_counts()
    ordered = np.sort(counts.to_numpy())[::-1]
    axes[1].plot(np.arange(1, len(ordered) + 1), ordered, color="#c44e52")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Destination rank")
    axes[1].set_ylabel("Times visited (log)")
    axes[1].set_title("SYNTHETIC: destination popularity bias")
    axes[1].grid(alpha=0.3)

    merged = counts.rename("visits").to_frame().join(
        destinations.set_index("destination_id")["popularity_score"]
    )
    axes[2].scatter(merged["popularity_score"], merged["visits"], alpha=0.4, s=12, color="#4c72b0")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Real popularity proxy (percentile)")
    axes[2].set_ylabel("Synthetic visits (log)")
    axes[2].set_title("Generator reproduces real popularity skew")
    axes[2].grid(alpha=0.3)
    _save(fig, figures / "interaction_distributions.png")


def plot_mechanism_mix(interactions: pd.DataFrame, figures: Path) -> None:
    """How often each generative mechanism produced a trip."""
    if "mechanism" not in interactions.columns:
        return
    shares = interactions["mechanism"].value_counts(normalize=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(shares.index, shares.to_numpy(), color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_ylabel("Share of trips")
    ax.set_title("SYNTHETIC: which mechanism generated each trip")
    for index, value in enumerate(shares.to_numpy()):
        ax.text(index, value + 0.01, f"{value:.1%}", ha="center")
    _save(fig, figures / "generator_mechanisms.png")


def summarise(dataset: TravelDataset) -> str:
    """Build the markdown EDA summary."""
    destinations, interactions = dataset.destinations, dataset.interactions
    trips = interactions.groupby("user_id").size()
    counts = interactions["destination_id"].value_counts()
    top_decile = int(np.ceil(0.1 * dataset.n_destinations))
    head_share = counts.head(top_decile).sum() / counts.sum()

    lines: List[str] = [
        "# Exploratory data analysis",
        "",
        "## Destination catalog (real data)",
        "",
        f"- **{dataset.n_destinations}** destinations across "
        f"**{destinations['country_code'].nunique()}** countries and "
        f"**{destinations['continent'].nunique()}** continents",
        f"- OpenStreetMap attributes available for **{destinations['poi_available'].mean():.1%}**",
        f"- Climate normals available for **{destinations['climate_available'].mean():.1%}**",
        f"- Country cost proxy available for **{destinations['cost_available'].mean():.1%}**",
        "",
        "### Continent breakdown",
        "",
        "| Continent | Destinations | Share |",
        "|---|---:|---:|",
    ]
    for continent, count in destinations["continent"].value_counts().items():
        lines.append(f"| {continent} | {count} | {count / dataset.n_destinations:.1%} |")

    lines += [
        "",
        "### Most popular destinations (Wikipedia pageview proxy)",
        "",
        "| Rank | City | Country | Mean monthly views |",
        "|---:|---|---|---:|",
    ]
    top = destinations.nlargest(10, "pageviews_monthly_mean")
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.city} | {row.country} | {row.pageviews_monthly_mean:,.0f} |"
        )

    lines += [
        "",
        "## Interactions (SYNTHETIC — not real travellers)",
        "",
        f"- **{dataset.n_users:,}** generated users, **{len(interactions):,}** trips",
        f"- Trips per user: min {trips.min()}, median {trips.median():.0f}, max {trips.max()}",
        f"- Destinations ever visited: **{counts.size}/{dataset.n_destinations}** "
        f"({counts.size / dataset.n_destinations:.1%} of the catalog)",
        f"- The most-visited 10% of destinations account for **{head_share:.1%}** of all trips, "
        "which is the popularity bias the models must be measured against",
        "",
    ]
    if "mechanism" in interactions.columns:
        lines.append("### Generative mechanism mix")
        lines.append("")
        lines.append("| Mechanism | Share |")
        lines.append("|---|---:|")
        for mechanism, share in interactions["mechanism"].value_counts(normalize=True).items():
            lines.append(f"| {mechanism} | {share:.1%} |")
        lines.append("")

    return "\n".join(lines)


def run(config: Config) -> str:
    """Generate every figure and the summary document."""
    dataset = load_dataset(config)
    figures = config.path("figures")
    reports = config.path("reports")

    plot_geographic_distribution(dataset.destinations, figures)
    plot_interactive_map(dataset.destinations, figures)
    plot_popularity_distribution(dataset.destinations, figures)
    plot_country_distribution(dataset.destinations, figures)
    plot_attribute_correlations(dataset.destinations, figures)
    plot_cost_and_climate(dataset.destinations, figures)
    plot_interaction_distributions(dataset.interactions, dataset.destinations, figures)
    plot_mechanism_mix(dataset.interactions, figures)

    summary = summarise(dataset)
    (reports / "eda_summary.md").write_text(summary, encoding="utf-8")
    LOGGER.info("Wrote eda_summary.md and %d figures", len(list(figures.glob("*"))))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TravelNext EDA")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
