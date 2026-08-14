/* TravelNext web client.
 *
 * Talks to the FastAPI backend on the same origin. All ranking, explanation
 * and context logic lives server-side in `src/service.py`; this file only
 * presents it, so the web UI and the Streamlit UI can never disagree about
 * what is recommended or why.
 *
 * The one piece of genuinely client-side logic is re-sorting (Match / Budget /
 * Nearest), which reorders results already returned rather than re-querying.
 */

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

const COST_ORDER = { budget: 0, "mid-range": 1, expensive: 2 };

const state = {
  history: ["amsterdam-nl", "berlin-de"],
  current: "prague-cz",
  month: 9,
  duration: 5,
  budget: "mid-range",
  model: "hybrid",
  sort: "match",
  k: 9,
  catalog: new Map(),
  results: [],
};

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- utils */

function titleCase(value) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

/** Great-circle distance, used only to sort by "Nearest" on the client. */
function distanceKm(a, b) {
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLon = toRad(b.lon - a.lon);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(Math.min(1, h)));
}

function flagEmoji(code) {
  if (!code || code.length !== 2) return "";
  return String.fromCodePoint(
    ...[...code.toUpperCase()].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)
  );
}

function notice(message) {
  $("notice").innerHTML = message ? `<div class="notice">${message}</div>` : "";
}

/* ------------------------------------------------------------------ api */

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function loadCatalog() {
  // The catalog powers name lookups, the "past trips" strip and photography.
  const data = await api("/destinations?limit=500");
  data.destinations.forEach((d) => state.catalog.set(d.destination_id, d));
  return data;
}

async function loadModels() {
  const data = await api("/models");
  const select = $("model");
  select.innerHTML = "";
  const labels = {
    hybrid: "Hybrid",
    learning_to_rank: "Learning-to-Rank",
    next_destination: "Next destination",
    content: "Content-based",
    collaborative: "Collaborative",
    popularity: "Popularity baseline",
  };
  data.available_models.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = labels[name] || name;
    if (name === state.model) option.selected = true;
    select.appendChild(option);
  });
}

/* -------------------------------------------------------------- render */

function renderSkeleton() {
  $("cards").innerHTML = Array.from({ length: 3 })
    .map(
      () => `<article class="card skeleton">
        <div class="card-photo"></div>
        <div class="card-body">
          <div class="bar" style="width:60%;height:22px"></div>
          <div class="bar" style="width:35%"></div>
          <div class="bar" style="width:90%"></div>
          <div class="bar" style="width:80%"></div>
        </div>
      </article>`
    )
    .join("");
}

function sortResults(items) {
  const copy = [...items];
  if (state.sort === "budget") {
    copy.sort(
      (a, b) =>
        (COST_ORDER[a.cost_category] ?? 3) - (COST_ORDER[b.cost_category] ?? 3) ||
        b.score - a.score
    );
  } else if (state.sort === "nearest") {
    const origin = state.catalog.get(state.current);
    if (origin) {
      const from = { lat: origin.latitude, lon: origin.longitude };
      copy.forEach((item) => {
        item._distance = distanceKm(from, { lat: item.latitude, lon: item.longitude });
      });
      copy.sort((a, b) => a._distance - b._distance);
    }
  } else {
    copy.sort((a, b) => b.score - a.score);
  }
  return copy;
}

function cardHtml(item, index) {
  // Per the spec the lead card gets a taller photo, but only under the
  // default "match" ordering — re-sorting genuinely changes the layout.
  const isLead = index === 0 && state.sort === "match";
  const match = Math.round(item.score * 100);
  const photo = item.image_url
    ? `<img src="${item.image_url}" alt="${item.city}" loading="lazy"
         onerror="this.style.display='none'" />`
    : "";
  const distance =
    item._distance !== undefined ? `<span class="chip">${Math.round(item._distance)} km</span>` : "";

  return `<article class="card${isLead ? " is-lead" : ""}">
    <div class="card-photo">
      ${photo}
      <span class="match-badge">${match}% match</span>
    </div>
    <div class="card-body">
      <h3 class="card-city">${item.city}</h3>
      <p class="card-country">${flagEmoji(item.country_code)} ${item.country}</p>
      <div class="match-bar"><span style="width:${match}%"></span></div>
      <div class="card-meta">
        <span class="chip">${item.cost_category}</span>
        ${distance}
      </div>
      <ul class="reasons">
        ${item.reasons
          .slice(0, 4)
          .map((r) => `<li><span class="tick">✓</span><span>${r}</span></li>`)
          .join("")}
      </ul>
      <div class="card-foot">
        <button class="link-next" data-next="${item.destination_id}">
          Where to go after ${item.city} →
        </button>
      </div>
    </div>
  </article>`;
}

function renderCards() {
  const sorted = sortResults(state.results);
  $("cards").innerHTML = sorted.map(cardHtml).join("");
  $("result-count").textContent = `${sorted.length} destinations`;

  const lead = sorted[0];
  $("hero-lead").textContent = lead ? lead.city : "somewhere new";

  document.querySelectorAll("[data-next]").forEach((button) => {
    button.addEventListener("click", () => {
      // Treat "where after X" as: X becomes the current destination.
      if (!state.history.includes(state.current)) state.history.push(state.current);
      state.current = button.dataset.next;
      renderTrips();
      refresh();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });
}

function renderTrips() {
  const ids = [...state.history, state.current].filter(Boolean);
  const cards = ids
    .map((id) => {
      const d = state.catalog.get(id);
      if (!d) return "";
      const isCurrent = id === state.current;
      const photo = d.image_url
        ? `<img src="${d.image_url}" alt="${d.city}" loading="lazy" onerror="this.style.display='none'" />`
        : `<div style="height:118px;background:#ece4d8"></div>`;
      return `<div class="trip">
        ${photo}
        <button class="remove" data-remove="${id}" title="Remove">×</button>
        <div class="trip-name">${d.city}
          <small>${isCurrent ? "Current" : d.country}</small>
        </div>
      </div>`;
    })
    .join("");

  $("trip-row").innerHTML =
    cards +
    `<button class="trip-add" id="add-trip">+ Add a trip<br /><span class="label">pick a city</span></button>`;

  $("add-trip").addEventListener("click", openPicker);
  document.querySelectorAll("[data-remove]").forEach((button) => {
    button.addEventListener("click", () => {
      const id = button.dataset.remove;
      state.history = state.history.filter((h) => h !== id);
      if (state.current === id) state.current = state.history.pop() || null;
      renderTrips();
      refresh();
    });
  });
}

function openPicker() {
  const options = [...state.catalog.values()]
    .filter((d) => d.destination_id !== state.current && !state.history.includes(d.destination_id))
    .map((d) => `${d.city}, ${d.country}`);
  const answer = window.prompt(
    "Add a city you have visited:\n\n(type part of a name, e.g. Lisbon)"
  );
  if (!answer) return;
  const needle = answer.trim().toLowerCase();
  const match = [...state.catalog.values()].find(
    (d) => `${d.city}, ${d.country}`.toLowerCase().includes(needle)
  );
  if (!match) {
    notice(`No destination in the catalog matches “${answer}”.`);
    return;
  }
  notice("");
  if (state.current) state.history.push(state.current);
  state.current = match.destination_id;
  renderTrips();
  refresh();
}

function renderSignals(payload) {
  $("sig-trips").textContent = state.history.length + (state.current ? 1 : 0);
  const origin = state.catalog.get(state.current);
  $("sig-origin").textContent = origin ? origin.city : "—";
  $("current-place").textContent = origin ? `${origin.city}, ${origin.country}` : "nowhere yet";
  $("sig-month").textContent = MONTHS[state.month - 1];
  $("sig-budget").textContent = titleCase(state.budget);
  $("sig-model").textContent = payload
    ? payload.model.replace(/_/g, " ")
    : titleCase(state.model);
}

/* -------------------------------------------------------------- refresh */

async function refresh() {
  renderSignals(null);
  renderSkeleton();
  try {
    const payload = await api("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        history: state.history,
        current_destination: state.current,
        month: state.month,
        trip_duration_days: state.duration,
        budget: state.budget,
        model: state.model,
        k: state.k,
      }),
    });
    state.results = payload.recommendations;
    renderSignals(payload);
    renderCards();

    if (payload.cold_start) {
      notice(
        "No travel history selected, so this is the <strong>cold-start</strong> path: " +
          "results come from the month, your budget and overall popularity rather than " +
          "from personal history."
      );
    } else if (payload.model !== payload.requested_model) {
      notice(`The ${payload.requested_model} model was unavailable; showing ${payload.model}.`);
    } else {
      notice("");
    }
  } catch (error) {
    $("cards").innerHTML = "";
    notice(`Could not load recommendations: ${error.message}`);
  }
}

/* ----------------------------------------------------------------- init */

function wireControls() {
  const monthSelect = $("month");
  MONTHS.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index + 1);
    option.textContent = name;
    if (index + 1 === state.month) option.selected = true;
    monthSelect.appendChild(option);
  });

  monthSelect.addEventListener("change", (e) => {
    state.month = Number(e.target.value);
    refresh();
  });
  $("duration").addEventListener("change", (e) => {
    state.duration = Math.max(2, Math.min(21, Number(e.target.value) || 5));
    e.target.value = state.duration;
    refresh();
  });
  $("budget").addEventListener("change", (e) => {
    state.budget = e.target.value;
    refresh();
  });
  $("model").addEventListener("change", (e) => {
    state.model = e.target.value;
    refresh();
  });

  $("sort").addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    state.sort = button.dataset.sort;
    document
      .querySelectorAll("#sort button")
      .forEach((b) => b.classList.toggle("is-active", b === button));
    renderCards();
  });

  $("nav").addEventListener("click", (e) => {
    const button = e.target.closest("button");
    if (!button) return;
    document
      .querySelectorAll("#nav button")
      .forEach((b) => b.classList.toggle("is-active", b === button));
    const target = { recommend: "view-recommend", how: "view-how", trips: "view-trips" }[
      button.dataset.view
    ];
    document.getElementById(target).scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

async function main() {
  wireControls();
  renderSkeleton();
  try {
    await Promise.all([loadCatalog(), loadModels()]);
  } catch (error) {
    notice(
      `Could not reach the API: ${error.message}. Start it with ` +
        `<code>uvicorn api.main:app</code>.`
    );
    return;
  }
  renderTrips();
  await refresh();
}

main();
