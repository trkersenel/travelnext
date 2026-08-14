/* TravelNext — onboarding and recommendation flow.
 *
 * A single-document flow: every screen exists in index.html and `show()`
 * swaps which one is visible. No framework and no build step, so the whole
 * client is three static files served by the same FastAPI process.
 *
 * All ranking, explanation and profile inference happens server-side in
 * src/service.py. This file collects input and renders results; it never
 * invents a reason, a match score or a trait.
 *
 * Flow:
 *   login -> welcome -> history -> map -> style -> building -> profile
 *         -> recommend <-> trips
 */

/* ------------------------------------------------------------------ data */

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

/* Interest labels shown to the traveller, mapped server-side onto measured
   OpenStreetMap categories (see RecommendationService.INTEREST_MAP). */
const INTERESTS = [
  ["architecture", "Architecture"],
  ["culture", "Culture"],
  ["food", "Food"],
  ["nightlife", "Nightlife"],
  ["museums", "Museums"],
  ["nature", "Nature"],
  ["history", "History"],
  ["beaches", "Beaches"],
  ["shopping", "Shopping"],
  ["local_life", "Local life"],
  ["photography", "Photography"],
  ["outdoor", "Outdoor activities"],
];

const DURATIONS = [
  [4, "3–4 days"],
  [6, "5–7 days"],
  [10, "1–2 weeks"],
  [18, "2+ weeks"],
];

const BUDGETS = [
  ["budget", "€"],
  ["mid-range", "€€"],
  ["expensive", "€€€"],
];

const COST_SYMBOL = { budget: "€", "mid-range": "€€", expensive: "€€€" };

const state = {
  history: [],          // destination ids, in the order the traveller added them
  interests: [],
  duration: 6,
  budget: "mid-range",
  month: new Date().getMonth() + 1,
  model: "hybrid",
  origin: null,         // starting point for the next trip
  catalog: [],
  byId: new Map(),
  results: [],
  profile: null,
  map: null,
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------- api */

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

const postJson = (path, payload) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

/* ---------------------------------------------------------------- router */

const SCREENS = [
  "login", "welcome", "history", "map", "style",
  "building", "profile", "recommend", "trips",
];

function show(name) {
  SCREENS.forEach((screen) => {
    $(`screen-${screen}`).classList.toggle("is-active", screen === name);
  });
  window.scrollTo(0, 0);

  if (name === "map") drawMap();
  if (name === "building") runBuild();
  if (name === "profile") renderProfile();
  if (name === "recommend") renderRefine();
  if (name === "trips") renderTimeline();
}

/* ----------------------------------------------------------------- utils */

function flagEmoji(code) {
  if (!code || code.length !== 2) return "";
  return String.fromCodePoint(
    ...[...code.toUpperCase()].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)
  );
}

function photo(destination, className) {
  const url = destination && destination.image_url;
  return url
    ? `<img class="${className || ""}" src="${url}" alt="${destination.city}"
         loading="lazy" onerror="this.style.visibility='hidden'" />`
    : `<div class="${className || ""}" style="background:#ece4d8"></div>`;
}

function durationLabel(days) {
  const found = DURATIONS.find(([value]) => value === days);
  return found ? found[1] : `${days} days`;
}

/* --------------------------------------------------- city search widget */

function attachSearch(inputId, listId, onPick) {
  const input = $(inputId);
  const list = $(listId);
  let focusIndex = -1;

  const close = () => {
    list.hidden = true;
    list.innerHTML = "";
    focusIndex = -1;
  };

  const render = (matches) => {
    if (!matches.length) return close();
    list.innerHTML = matches
      .map(
        (d, i) => `<li data-id="${d.destination_id}" class="${i === 0 ? "is-focus" : ""}">
          ${photo(d)}
          <span><strong>${d.city}</strong>
            <span class="s-country">${flagEmoji(d.country_code)} ${d.country}</span>
          </span>
        </li>`
      )
      .join("");
    list.hidden = false;
    focusIndex = 0;
  };

  input.addEventListener("input", () => {
    const needle = input.value.trim().toLowerCase();
    if (needle.length < 2) return close();
    const matches = state.catalog
      .filter(
        (d) =>
          !state.history.includes(d.destination_id) &&
          (d.city.toLowerCase().includes(needle) ||
            d.country.toLowerCase().includes(needle))
      )
      .slice(0, 7);
    render(matches);
  });

  input.addEventListener("keydown", (event) => {
    const items = [...list.querySelectorAll("li")];
    if (!items.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      focusIndex =
        (focusIndex + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((li, i) => li.classList.toggle("is-focus", i === focusIndex));
    } else if (event.key === "Enter" && focusIndex >= 0) {
      event.preventDefault();
      onPick(items[focusIndex].dataset.id);
      input.value = "";
      close();
    } else if (event.key === "Escape") {
      close();
    }
  });

  list.addEventListener("click", (event) => {
    const li = event.target.closest("li");
    if (!li) return;
    onPick(li.dataset.id);
    input.value = "";
    close();
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(`#${inputId}`) && !event.target.closest(`#${listId}`)) close();
  });
}

/* ------------------------------------------------- 3. travel history */

function addCity(id) {
  if (!state.byId.has(id) || state.history.includes(id)) return;
  state.history.push(id);
  state.origin = id; // most recent trip becomes the default starting point
  renderVisited();
  renderTimeline();
}

function removeCity(id) {
  state.history = state.history.filter((h) => h !== id);
  if (state.origin === id) state.origin = state.history[state.history.length - 1] || null;
  renderVisited();
  renderTimeline();
}

function renderVisited() {
  const list = $("visited-list");
  list.innerHTML = state.history
    .map((id) => {
      const d = state.byId.get(id);
      return `<li>
        ${photo(d)}
        <span>
          <span class="v-name"><span class="v-tick">✓</span>${d.city}</span><br />
          <span class="v-country">${d.country}</span>
        </span>
        <button class="v-remove" data-remove="${id}" aria-label="Remove ${d.city}">×</button>
      </li>`;
    })
    .join("");

  const count = state.history.length;
  $("history-count").textContent = count
    ? `${count} ${count === 1 ? "destination" : "destinations"} added`
    : "";
  $("history-empty").hidden = count > 0;
  $("history-continue").disabled = count === 0;

  list.querySelectorAll("[data-remove]").forEach((button) =>
    button.addEventListener("click", () => removeCity(button.dataset.remove))
  );
}

/* ------------------------------------------------------------ 4. map */

function drawMap() {
  const points = state.history.map((id) => state.byId.get(id)).filter(Boolean);
  $("map-summary").textContent = points.length
    ? `${points.map((d) => d.city).join(" → ")}`
    : "";

  if (typeof L === "undefined") {
    // Leaflet is loaded from a CDN; without it the flow must still work.
    $("map").innerHTML =
      '<p style="padding:24px;color:#8a7d74">Map unavailable offline — ' +
      "your destinations are still recorded.</p>";
    return;
  }

  if (!state.map) {
    state.map = L.map("map", { scrollWheelZoom: false, attributionControl: false });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
    }).addTo(state.map);
  }

  // Clear previous markers and lines before redrawing.
  state.map.eachLayer((layer) => {
    if (layer instanceof L.Marker || layer instanceof L.Polyline) state.map.removeLayer(layer);
  });

  if (!points.length) {
    state.map.setView([48, 10], 4);
    return;
  }

  const coords = points.map((d) => [d.latitude, d.longitude]);
  points.forEach((d) => {
    L.marker([d.latitude, d.longitude], {
      icon: L.divIcon({ className: "", html: '<div class="pin"></div>', iconSize: [13, 13] }),
    })
      .addTo(state.map)
      .bindTooltip(`${d.city}, ${d.country}`);
  });

  if (coords.length > 1) {
    L.polyline(coords, { color: "#2a1f1b", weight: 1, opacity: 0.35, dashArray: "3 5" })
      .addTo(state.map);
  }

  // The map is built while its screen is still hidden, so Leaflet measures the
  // container as 0x0 and a fitBounds computed then resolves to maximum zoom —
  // the view lands on a random street instead of the traveller's cities.
  // Waiting for a real measured width is the only reliable fix; a fixed
  // timeout races with layout and fits to a narrow strip.
  const fitWhenSized = (attempt = 0) => {
    const width = $("map").clientWidth;
    if (width < 80 && attempt < 30) {
      requestAnimationFrame(() => fitWhenSized(attempt + 1));
      return;
    }
    state.map.invalidateSize();
    if (coords.length > 1) {
      state.map.fitBounds(coords, { padding: [48, 48], maxZoom: 8 });
    } else {
      state.map.setView(coords[0], 6);
    }
  };
  fitWhenSized();
}

/* ---------------------------------------------------------- 5. style */

function renderChoices() {
  $("interests").innerHTML = INTERESTS.map(
    ([value, label]) =>
      `<button class="chip-toggle${state.interests.includes(value) ? " is-on" : ""}"
         data-interest="${value}">${label}</button>`
  ).join("");

  $("durations").innerHTML = DURATIONS.map(
    ([value, label]) =>
      `<button class="chip-toggle${state.duration === value ? " is-on" : ""}"
         data-duration="${value}">${label}</button>`
  ).join("");

  $("budgets").innerHTML = BUDGETS.map(
    ([value, label]) =>
      `<button class="chip-toggle${state.budget === value ? " is-on" : ""}"
         data-budget="${value}">${label}</button>`
  ).join("");
}

/* ------------------------------------------------------- 6. building */

function drawRouteAnimation() {
  const svg = $("route-anim");
  const points = state.history.map((id) => state.byId.get(id)).filter(Boolean).slice(0, 6);
  if (points.length < 1) {
    svg.innerHTML = "";
    return;
  }

  // Lay the traveller's own cities out along a gentle arc: geographic in
  // feel, with no spinner and nothing that reads as "AI processing".
  const width = 420;
  const step = points.length > 1 ? (width - 80) / (points.length - 1) : 0;
  const coords = points.map((d, i) => [
    40 + i * step,
    82 + Math.sin(i * 1.05) * 26,
  ]);

  const path = coords.map(([x, y], i) => `${i ? "L" : "M"}${x},${y}`).join(" ");
  svg.innerHTML =
    `<path d="${path}" />` +
    `<path class="drawn" d="${path}" />` +
    coords
      .map(
        ([x, y], i) =>
          `<circle cx="${x}" cy="${y}" r="3.5" /><text x="${x}" y="${y - 11}">${
            points[i].city
          }</text>`
      )
      .join("");

  const drawn = svg.querySelector(".drawn");
  const length = drawn.getTotalLength();
  drawn.style.strokeDasharray = length;
  drawn.style.strokeDashoffset = length;
  drawn.style.transition = "stroke-dashoffset 2.4s ease";
  requestAnimationFrame(() => {
    drawn.style.strokeDashoffset = "0";
  });
}

async function runBuild() {
  drawRouteAnimation();
  const steps = [...document.querySelectorAll("#build-steps li")];
  steps.forEach((li) => li.classList.remove("is-done", "is-active"));

  const advance = (index) => {
    steps.forEach((li, i) => {
      li.classList.toggle("is-done", i < index);
      li.classList.toggle("is-active", i === index);
    });
  };

  advance(0);
  // Real work happens behind the steps: the profile and the first
  // recommendation set are fetched while the sequence plays.
  const work = (async () => {
    state.profile = await postJson("/profile", { history: state.history });
    await fetchRecommendations();
  })();

  for (let i = 1; i <= 3; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 620));
    advance(i);
  }
  await work.catch((error) => notice(`Could not build your profile: ${error.message}`));
  advance(4);
  await new Promise((resolve) => setTimeout(resolve, 320));
  show("profile");
}

/* -------------------------------------------------------- 7. profile */

function renderProfile() {
  const profile = state.profile;
  if (!profile) return;

  const maxDeviation = Math.max(...profile.traits.map((t) => t.deviation), 1);
  $("profile-traits").innerHTML = profile.traits.length
    ? profile.traits
        .map(
          (t) => `<li><span>${t.label}</span>
            <span class="trait-bar"><span style="width:${Math.round(
              (t.deviation / maxDeviation) * 100
            )}%"></span></span></li>`
        )
        .join("")
    : "<li><span>Add a few more cities to infer a pattern</span><span></span></li>";

  $("trait-hint").textContent = profile.traits.length
    ? "Inferred by comparing your destinations against the rest of the catalog — " +
      "these are the traits your history leans towards, not simply what scores highly."
    : "";

  $("profile-duration").textContent = durationLabel(state.duration);
  $("profile-region").textContent = profile.region || profile.continent || "—";
  $("profile-budget").textContent =
    `${COST_SYMBOL[state.budget]} · ${state.budget}`;

  $("profile-visited").innerHTML = (profile.visited || [])
    .map(
      (d) => `<div class="trip">
        ${photo(d)}
        <div class="trip-name">${d.city}<small>${d.country}</small></div>
      </div>`
    )
    .join("");
}

/* ------------------------------------------- 8 & 10. recommendations */

function notice(message) {
  $("notice").innerHTML = message ? `<div class="notice">${message}</div>` : "";
}

async function fetchRecommendations() {
  const payload = await postJson("/recommend", {
    history: state.history.filter((id) => id !== state.origin),
    current_destination: state.origin,
    month: state.month,
    trip_duration_days: state.duration,
    budget: state.budget,
    interests: state.interests,
    model: state.model,
    k: 9,
  });
  state.results = payload.recommendations;
  renderCards();

  if (payload.cold_start) {
    notice(
      "No travel history yet, so these come from your stated interests, the " +
        "month and overall popularity rather than from your own trips."
    );
  } else {
    notice("");
  }
  $("rec-status").textContent = `${payload.recommendations.length} results · ${payload.model.replace(
    /_/g,
    " "
  )}`;
  return payload;
}

function renderCards() {
  $("cards").innerHTML = state.results
    .map((item, index) => {
      const match = Math.round(item.score * 100);
      const attributes = Object.entries(item.attributes || {})
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([name]) => name);
      return `<article class="card${index === 0 ? " is-lead" : ""}">
        <div class="card-photo">
          ${photo(item)}
          <span class="match-badge">${match}% match</span>
        </div>
        <div class="card-body">
          <h3 class="card-city">${item.city}</h3>
          <p class="card-country">${flagEmoji(item.country_code)} ${item.country}</p>
          <div class="match-bar"><span style="width:${match}%"></span></div>
          <div class="card-meta">
            ${attributes.map((a) => `<span class="chip">${a}</span>`).join("")}
          </div>
          <ul class="reasons">
            ${(item.reasons || [])
              .slice(0, 2)
              .map((r) => `<li><span class="tick">✓</span><span>${r}</span></li>`)
              .join("")}
          </ul>
          <div class="card-foot">
            <button class="link-next" data-explain="${item.destination_id}">
              Why ${item.city}? →
            </button>
          </div>
        </div>
      </article>`;
    })
    .join("");

  document.querySelectorAll("[data-explain]").forEach((button) =>
    button.addEventListener("click", () => openDrawer(button.dataset.explain))
  );
}

function renderRefine() {
  const origin = $("r-origin");
  origin.innerHTML = state.history
    .map(
      (id) =>
        `<option value="${id}"${id === state.origin ? " selected" : ""}>${
          state.byId.get(id).city
        }</option>`
    )
    .join("");

  $("r-duration").innerHTML = DURATIONS.map(
    ([value, label]) =>
      `<option value="${value}"${value === state.duration ? " selected" : ""}>${label}</option>`
  ).join("");

  $("r-budget").innerHTML = BUDGETS.map(
    ([value, label]) =>
      `<option value="${value}"${value === state.budget ? " selected" : ""}>${label} ${value}</option>`
  ).join("");

  $("r-month").innerHTML = MONTHS.map(
    (name, i) =>
      `<option value="${i + 1}"${i + 1 === state.month ? " selected" : ""}>${name}</option>`
  ).join("");

  $("r-interests").innerHTML = INTERESTS.map(
    ([value, label]) =>
      `<button class="chip-toggle${state.interests.includes(value) ? " is-on" : ""}"
         data-interest="${value}">${label}</button>`
  ).join("");
}

/* ---------------------------------------------------- 9. explanation */

async function openDrawer(destinationId) {
  const item = state.results.find((r) => r.destination_id === destinationId);
  if (!item) return;

  $("drawer-content").innerHTML = `
    ${item.image_url ? `<img class="drawer-photo" src="${item.image_url}" alt="${item.city}" />` : ""}
    <span class="label">${Math.round(item.score * 100)}% match</span>
    <h2>Why ${item.city}?</h2>
    <p class="d-country">${flagEmoji(item.country_code)} ${item.country}</p>
    <ul class="evidence">
      ${(item.reason_details || [])
        .map(
          (reason) => `<li>
            <span class="tick">✓</span>
            <span class="e-text">${reason.text}
              <span class="e-detail">${formatEvidence(reason)}</span>
            </span>
          </li>`
        )
        .join("")}
    </ul>
    <h3 class="sub">Measured attributes</h3>
    <ul class="attr-list">
      ${Object.entries(item.attributes || {})
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6)
        .map(
          ([name, value]) => `<li><span>${name}</span>
            <span class="trait-bar"><span style="width:${Math.round(value * 100)}%"></span></span>
          </li>`
        )
        .join("")}
    </ul>
    <p class="hint">
      Percentiles across the ${state.catalog.length}-destination catalog, counted
      from OpenStreetMap. Every reason above is generated from a value the model
      computed, not from a template.
    </p>`;

  $("drawer").hidden = false;
  $("drawer-backdrop").hidden = false;
}

/** Render the evidence numbers behind a reason, so a claim can be checked. */
function formatEvidence(reason) {
  const e = reason.evidence || {};
  const parts = [];
  if (e.similarity !== undefined) parts.push(`cosine similarity ${e.similarity}`);
  if (e.cf_similarity !== undefined) parts.push(`co-visitation ${e.cf_similarity}`);
  if (e.distance_km !== undefined) parts.push(`${e.distance_km} km`);
  if (e.mean_temp_c !== undefined) parts.push(`${e.mean_temp_c}°C average`);
  if (e.season_score !== undefined) parts.push(`season fit ${e.season_score}`);
  if (e.budget_fit !== undefined) parts.push(`budget fit ${e.budget_fit}`);
  if (e.popularity_percentile !== undefined)
    parts.push(`popularity percentile ${e.popularity_percentile}`);
  if (e.scores) {
    parts.push(
      Object.entries(e.scores)
        .map(([k, v]) => `${k} ${v}`)
        .join(" · ")
    );
  }
  return parts.join(" · ");
}

function closeDrawer() {
  $("drawer").hidden = true;
  $("drawer-backdrop").hidden = true;
}

/* -------------------------------------------------------- 11. my trips */

function renderTimeline() {
  const timeline = $("trips-timeline");
  if (!timeline) return;
  timeline.innerHTML = state.history
    .map((id) => {
      const d = state.byId.get(id);
      return `<li>
        ${photo(d)}
        <span>
          <span class="t-name">${d.city}</span><br />
          <span class="t-country">${flagEmoji(d.country_code)} ${d.country}</span>
        </span>
        <button class="t-remove" data-remove-trip="${id}" aria-label="Remove ${d.city}">×</button>
      </li>`;
    })
    .join("");

  timeline.querySelectorAll("[data-remove-trip]").forEach((button) =>
    button.addEventListener("click", async () => {
      removeCity(button.dataset.removeTrip);
      await refreshAfterHistoryChange();
    })
  );
}

/** History changed on the My Trips page: profile and results must follow. */
async function refreshAfterHistoryChange() {
  if (!state.history.length) return;
  if (!state.history.includes(state.origin)) {
    state.origin = state.history[state.history.length - 1];
  }
  state.profile = await postJson("/profile", { history: state.history });
  renderRefine();
  await fetchRecommendations();
}

/* -------------------------------------------------------------- wiring */

function wire() {
  // Login — demo entry points. No credentials are requested or transmitted.
  document.querySelectorAll("[data-auth]").forEach((button) =>
    button.addEventListener("click", () => show("welcome"))
  );

  document.querySelectorAll("[data-go]").forEach((button) =>
    button.addEventListener("click", () => show(button.dataset.go))
  );

  attachSearch("city-search", "suggestions", addCity);
  attachSearch("trips-search", "trips-suggestions", async (id) => {
    addCity(id);
    await refreshAfterHistoryChange();
  });

  // Preference chips (onboarding + refine share the same handler).
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-interest]");
    if (button) {
      const value = button.dataset.interest;
      state.interests = state.interests.includes(value)
        ? state.interests.filter((i) => i !== value)
        : [...state.interests, value];
      renderChoices();
      renderRefine();
    }
    const duration = event.target.closest("[data-duration]");
    if (duration) {
      state.duration = Number(duration.dataset.duration);
      renderChoices();
    }
    const budget = event.target.closest("[data-budget]");
    if (budget) {
      state.budget = budget.dataset.budget;
      renderChoices();
    }
  });

  // Refine panel
  $("r-origin").addEventListener("change", (e) => (state.origin = e.target.value));
  $("r-duration").addEventListener("change", (e) => (state.duration = Number(e.target.value)));
  $("r-budget").addEventListener("change", (e) => (state.budget = e.target.value));
  $("r-month").addEventListener("change", (e) => (state.month = Number(e.target.value)));
  $("r-model").addEventListener("change", (e) => (state.model = e.target.value));

  $("update-recs").addEventListener("click", async () => {
    $("rec-status").textContent = "Updating…";
    try {
      await fetchRecommendations();
      state.profile = await postJson("/profile", { history: state.history });
    } catch (error) {
      notice(`Could not update: ${error.message}`);
    }
  });

  // Cross-page navigation
  document.querySelectorAll("#nav button, #nav-trips button").forEach((button) =>
    button.addEventListener("click", () => show(button.dataset.view))
  );

  $("drawer-close").addEventListener("click", closeDrawer);
  $("drawer-backdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => e.key === "Escape" && closeDrawer());
}

async function main() {
  wire();
  renderChoices();

  try {
    const [catalog, models] = await Promise.all([
      api("/destinations?limit=500"),
      api("/models"),
    ]);
    state.catalog = catalog.destinations;
    state.catalog.forEach((d) => state.byId.set(d.destination_id, d));

    const labels = {
      hybrid: "Hybrid",
      learning_to_rank: "Learning-to-Rank",
      next_destination: "Next destination",
      content: "Content-based",
      collaborative: "Collaborative",
      popularity: "Popularity baseline",
    };
    $("r-model").innerHTML = models.available_models
      .map(
        (name) =>
          `<option value="${name}"${name === state.model ? " selected" : ""}>${
            labels[name] || name
          }</option>`
      )
      .join("");

    // Login hero: a real photograph of a real destination, chosen as the most
    // popular European city in the catalog. Data-driven rather than hardcoded,
    // but scoped to the continent the product is pitched around -- the plain
    // catalog order is global and opened on New York.
    const hero =
      state.catalog.find((d) => d.image_url && d.continent === "Europe") ||
      state.catalog.find((d) => d.image_url);
    if (hero) {
      $("login-photo").src = hero.image_url;
      $("login-photo").alt = `${hero.city}, ${hero.country}`;
      $("login-credit").textContent = `${hero.city}, ${hero.country} · Wikimedia Commons`;
    }
  } catch (error) {
    document.body.insertAdjacentHTML(
      "afterbegin",
      `<div class="notice" style="margin:20px">Could not reach the API: ${error.message}</div>`
    );
    return;
  }

  renderVisited();
  renderRefine();
}

main();
