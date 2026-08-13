"""TravelNext Streamlit interface.

    streamlit run app/streamlit_app.py

Talks to :class:`src.service.RecommendationService` directly rather than over
HTTP, so the UI runs standalone without the API process. The map uses Folium
with OpenStreetMap tiles -- free, no key, no Google Maps dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow `streamlit run app/streamlit_app.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import AntPath
from streamlit_folium import st_folium

from src.config import load_config
from src.preprocessing.features import PROFILE_CATEGORIES
from src.service import RecommendationService

st.set_page_config(page_title="TravelNext", page_icon="🧭", layout="wide")

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MODEL_LABELS = {
    "hybrid": "Hybrid (content + collaborative + popularity)",
    "learning_to_rank": "Learning-to-Rank (LightGBM)",
    "next_destination": "Next destination (transition-aware)",
    "content": "Content-based",
    "collaborative": "Collaborative filtering",
    "popularity": "Popularity baseline",
}


@st.cache_resource(show_spinner="Loading TravelNext models…")
def load_service() -> RecommendationService:
    """Build the service once per Streamlit session."""
    return RecommendationService()


def flag_emoji(country_code: str) -> str:
    """Regional-indicator flag for an ISO alpha-2 country code."""
    code = (country_code or "").strip().upper()
    if len(code) != 2 or not code.isalpha():
        return "🏳️"
    return chr(0x1F1E6 + ord(code[0]) - 65) + chr(0x1F1E6 + ord(code[1]) - 65)


def build_map(
    service: RecommendationService,
    visited_ids: List[str],
    recommendations: List[Dict],
) -> folium.Map:
    """Map of previous trips (blue) and recommendations (red), with links."""
    frame = service.destinations_frame().set_index("destination_id")

    points: List[tuple[float, float]] = []
    for destination_id in visited_ids:
        if destination_id in frame.index:
            row = frame.loc[destination_id]
            points.append((float(row["latitude"]), float(row["longitude"])))
    for item in recommendations:
        points.append((item["latitude"], item["longitude"]))

    if points:
        centre = [
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        ]
    else:
        centre = [48.0, 10.0]

    # Pick the zoom from the spread of the markers. `fit_bounds` alone is not
    # reliable through st_folium -- the map renders at `zoom_start` and the
    # result was a whole-world view for a set of European cities.
    zoom = 4
    if len(points) > 1:
        span = max(
            max(p[0] for p in points) - min(p[0] for p in points),
            (max(p[1] for p in points) - min(p[1] for p in points)) / 2.0,
        )
        for threshold, level in ((1, 9), (3, 7), (8, 6), (20, 5), (45, 4), (90, 3)):
            if span <= threshold:
                zoom = level
                break
        else:
            zoom = 2

    fmap = folium.Map(location=centre, zoom_start=zoom, tiles="OpenStreetMap")

    for destination_id in visited_ids:
        if destination_id not in frame.index:
            continue
        row = frame.loc[destination_id]
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=7,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.9,
            popup=f"Visited: {row['city']}, {row['country']}",
            tooltip=f"✈️ {row['city']}",
        ).add_to(fmap)

    origin = None
    if visited_ids and visited_ids[-1] in frame.index:
        last = frame.loc[visited_ids[-1]]
        origin = (float(last["latitude"]), float(last["longitude"]))

    for item in recommendations:
        target = (item["latitude"], item["longitude"])
        folium.Marker(
            location=list(target),
            popup=folium.Popup(
                f"<b>#{item['rank']} {item['city']}</b><br>{item['country']}<br>"
                f"Match {item['score'] * 100:.0f}%",
                max_width=250,
            ),
            tooltip=f"#{item['rank']} {item['city']}",
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(fmap)
        if origin is not None:
            # An animated line makes the "from here, go there" story readable.
            AntPath(
                locations=[list(origin), list(target)],
                color="#d62728",
                weight=2,
                opacity=0.55,
                delay=1200,
            ).add_to(fmap)

    # No fit_bounds(): through st_folium the container is not sized when Leaflet
    # runs it, so it silently falls back to a whole-world view. The computed
    # zoom_start above is reliable.
    return fmap


def render_recommendation(item: Dict) -> None:
    """Render one recommendation card."""
    flag = flag_emoji(item["country_code"])
    header = f"{flag} **{item['rank']}. {item['city']}**, {item['country']}"

    left, right = st.columns([3, 1])
    with left:
        st.markdown(header)
        if item["reasons"]:
            # Two trailing spaces force a hard line break in markdown;
            # a plain "\n" collapses every reason onto one line.
            st.markdown("  \n".join(f"✓ {reason}" for reason in item["reasons"]))
        else:
            st.caption("No strong signal available for this destination.")
    with right:
        st.metric("Match", f"{item['score'] * 100:.0f}%")
        st.caption(f"{item['cost_category']}")

    attributes = item.get("attributes") or {}
    if attributes:
        top = sorted(attributes.items(), key=lambda kv: -kv[1])[:5]
        st.caption(" · ".join(f"{name} {value * 100:.0f}%" for name, value in top))
    st.divider()


def main() -> None:
    st.title("🧭 TravelNext")
    st.caption("Where should I travel next? — an explainable travel recommender built on free, open data.")

    try:
        service = load_service()
    except FileNotFoundError as exc:
        st.error(
            "Dataset not built yet. Run:\n\n"
            "```bash\npython -m src.data.build_destinations\npython -m src.data.build_dataset\n```"
        )
        st.caption(str(exc))
        return

    frame = service.destinations_frame()
    label_by_id = {
        row.destination_id: f"{row.city}, {row.country}"
        for row in frame.sort_values("popularity_score", ascending=False).itertuples(index=False)
    }
    ids_by_label = {label: key for key, label in label_by_id.items()}
    labels = list(ids_by_label)

    # ------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Your travel profile")

        default_history = [
            label_by_id[key]
            for key in ("amsterdam-nl", "berlin-de", "prague-cz")
            if key in label_by_id
        ]
        history_labels = st.multiselect(
            "Destinations you have visited",
            options=labels,
            default=default_history,
            help="Oldest first. The most recent visit is used as your starting point.",
        )
        current_label = st.selectbox(
            "Current / most recent destination",
            options=["(use the last one selected above)"] + labels,
            index=0,
        )

        st.divider()
        st.subheader("This trip")
        month = st.select_slider(
            "Travel month", options=list(range(1, 13)), value=9, format_func=lambda m: MONTH_NAMES[m - 1]
        )
        duration = st.slider("Trip duration (days)", min_value=2, max_value=21, value=5)
        budget = st.radio("Budget", options=["budget", "mid-range", "expensive"], index=1, horizontal=True)
        interests = st.multiselect(
            "Interests", options=list(PROFILE_CATEGORIES), default=[],
            help="Used most heavily when you have no travel history yet.",
        )

        st.divider()
        available = service.available_models()
        model = st.selectbox(
            "Recommendation model",
            options=available,
            index=available.index("hybrid") if "hybrid" in available else 0,
            format_func=lambda name: MODEL_LABELS.get(name, name),
        )
        k = st.slider("Number of recommendations (K)", min_value=3, max_value=25, value=10)

        if "learning_to_rank" not in available:
            st.info("Run `python -m src.evaluation.run_experiments` to enable the Learning-to-Rank model.")

    history_ids = [ids_by_label[label] for label in history_labels]
    current_id: Optional[str] = (
        ids_by_label[current_label] if current_label in ids_by_label else None
    )
    if current_id is None and history_ids:
        current_id, history_ids = history_ids[-1], history_ids[:-1]

    # ---------------------------------------------------------------- body
    result = service.recommend(
        history=history_ids,
        current_destination=current_id,
        month=int(month),
        trip_duration_days=int(duration),
        budget=budget,
        interests=interests,
        model=model,
        k=int(k),
    )
    recommendations = result["recommendations"]

    if result["cold_start"]:
        st.info(
            "No travel history selected, so this is the **cold-start** path: "
            "recommendations come from your stated interests, the month, your "
            "budget and overall popularity."
        )

    visited_ids = ([*history_ids, current_id] if current_id else list(history_ids))

    tab_results, tab_map, tab_compare, tab_about = st.tabs(
        ["Your next destinations", "Map", "Destination detail", "About the data"]
    )

    with tab_results:
        st.subheader("Your next destinations")
        st.caption(f"Model used: **{MODEL_LABELS.get(result['model'], result['model'])}**")
        if result["unknown_destinations"]:
            st.warning(f"Ignored unknown destinations: {', '.join(result['unknown_destinations'])}")
        if not recommendations:
            st.warning("No recommendations could be produced for this request.")
        for item in recommendations:
            render_recommendation(item)

    with tab_map:
        st.subheader("Where you have been, and where to go next")
        st.caption("Blue = visited · Red = recommended · Tiles © OpenStreetMap contributors")
        st_folium(build_map(service, visited_ids, recommendations), height=520, use_container_width=True)

    with tab_compare:
        if recommendations:
            choice = st.selectbox(
                "Destination", options=[item["destination_id"] for item in recommendations],
                format_func=lambda key: label_by_id.get(key, key),
            )
            detail = service.destination_detail(choice)

            left, right = st.columns([1, 1])
            with left:
                st.markdown(f"### {detail['city']}, {detail['country']}")
                st.caption(detail["summary"] or "No description available.")
                st.markdown(
                    f"**Population** {detail['population']:,} · "
                    f"**Cost band** {detail['cost_category']} · "
                    f"**Popularity percentile** {detail['popularity_percentile']:.2f}"
                )
                if not detail["osm_data_available"]:
                    st.warning("No OpenStreetMap attribute data for this destination.")

                attributes = {k: v for k, v in detail["attributes"].items() if v is not None}
                if attributes:
                    chart = pd.DataFrame(
                        {"attribute": list(attributes), "score": list(attributes.values())}
                    )
                    st.plotly_chart(
                        px.bar(
                            chart.sort_values("score"),
                            x="score", y="attribute", orientation="h",
                            range_x=[0, 1], title="Attribute percentiles (from OpenStreetMap)",
                        ),
                        width='stretch',
                    )
            with right:
                climate = pd.DataFrame(detail["monthly_climate"])
                if climate["mean_temp_c"].notna().any():
                    climate["month_name"] = [MONTH_NAMES[m - 1][:3] for m in climate["month"]]
                    st.plotly_chart(
                        px.line(
                            climate, x="month_name", y="mean_temp_c", markers=True,
                            title="Mean temperature by month (Open-Meteo)",
                            labels={"mean_temp_c": "°C", "month_name": ""},
                        ),
                        width='stretch',
                    )
                st.markdown("**Most similar destinations**")
                st.dataframe(
                    pd.DataFrame(detail["similar_destinations"])[["city", "similarity"]],
                    hide_index=True, width='stretch',
                )

    with tab_about:
        st.markdown(
            """
### Where this data comes from

| Layer | Source | Licence |
|---|---|---|
| Cities, coordinates, population | GeoNames `cities15000` | CC BY 4.0 |
| Destination attributes | OpenStreetMap via Overpass | ODbL 1.0 |
| Climate normals | Open-Meteo historical archive | CC BY 4.0 |
| Popularity | Wikimedia Pageviews API | CC0 |
| Cost band | World Bank GNI per capita (PPP) | CC BY 4.0 |
| Descriptions | English Wikipedia | CC BY-SA 4.0 |

Everything above is free and needs no API key.

### Honest limitations

* **Traveller interactions are synthetic.** No real person's travel history is
  used anywhere in this application. The collaborative signals demonstrate the
  mechanism; they are not evidence about real travel behaviour.
* **Popularity is a proxy** — Wikipedia pageviews measure online interest, not
  visitor arrivals.
* **Cost is a country-level proxy** — every city in a country shares a band.
* **OpenStreetMap coverage is uneven**, so absolute attribute scores favour
  well-mapped regions. Attribute *profiles* are computed within a city and are
  much less affected.
            """
        )
        st.caption(
            f"Catalog: {service.dataset.n_destinations} destinations · "
            f"{service.dataset.destinations['country_code'].nunique()} countries · "
            f"{service.dataset.n_users:,} synthetic users"
        )


if __name__ == "__main__":
    main()
