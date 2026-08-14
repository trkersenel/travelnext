/* Waygo — onboarding and recommendation flow.
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

/* ------------------------------------------------------------- gamification
 *
 * Everything below is computed from the traveller's OWN data — the countries
 * they've marked visited and the trips they've added — never from a fake
 * "streak" or an invented event. There is no day-by-day usage tracking in
 * this app, so there is no login streak: XP, levels and badges are the
 * honest alternative, each one a transparent function of real counts.
 */

/** Explorer levels, keyed by total XP. Ordered ascending; the last one with
 *  a threshold at or below the traveller's XP wins. */
const LEVELS = [
  { min: 0, name: "Homebody", icon: "🏠" },
  { min: 20, name: "Wanderer", icon: "🧭" },
  { min: 60, name: "Explorer", icon: "🗺️" },
  { min: 120, name: "Globetrotter", icon: "🌍" },
  { min: 200, name: "Jetsetter", icon: "✈️" },
  { min: 320, name: "World Nomad", icon: "🌐" },
  { min: 480, name: "Legend", icon: "🏆" },
];

/** Badge definitions. `test` reads only real, already-available state. */
const BADGES = [
  {
    id: "first-trip",
    label: "First trip",
    icon: "🎒",
    test: (s) => s.history.length >= 1,
  },
  {
    id: "first-country",
    label: "First country",
    icon: "📍",
    test: (s, visited) => visited.size >= 1,
  },
  {
    id: "five-countries",
    label: "5 countries",
    icon: "🗺️",
    test: (s, visited) => visited.size >= 5,
  },
  {
    id: "ten-countries",
    label: "10 countries",
    icon: "🌍",
    test: (s, visited) => visited.size >= 10,
  },
  {
    id: "three-continents",
    label: "3 continents",
    icon: "🧳",
    test: (s, visited, continents) => continents.size >= 3,
  },
  {
    id: "six-continents",
    label: "Every continent",
    icon: "🌐",
    test: (s, visited, continents) => continents.size >= 6,
  },
  {
    id: "city-hopper",
    label: "5 trips logged",
    icon: "🏙️",
    test: (s) => s.history.length >= 5,
  },
  {
    id: "twenty-countries",
    label: "20 countries",
    icon: "🏆",
    test: (s, visited) => visited.size >= 20,
  },
];

/** XP: a plain, disclosed formula over real counts — 20 per country visited,
 *  8 per trip added. Not a hidden score; the profile menu labels it as such. */
function computeXP() {
  const visited = visitedCountryCodes();
  return visited.size * 20 + state.history.length * 8;
}

function computeLevel(xp) {
  let current = LEVELS[0];
  let next = LEVELS[1] || null;
  for (let i = 0; i < LEVELS.length; i += 1) {
    if (xp >= LEVELS[i].min) {
      current = LEVELS[i];
      next = LEVELS[i + 1] || null;
    }
  }
  const span = next ? next.min - current.min : 1;
  const progress = next ? Math.min(1, (xp - current.min) / span) : 1;
  return { current, next, progress, xp };
}

function computeBadges() {
  const visited = visitedCountryCodes();
  const continents = new Set(
    [...visited].map((c) => (state.countryByCode.get(c) || {}).continent).filter(Boolean)
  );
  return BADGES.map((badge) => ({
    ...badge,
    unlocked: Boolean(badge.test(state, visited, continents)),
  }));
}

/** A short, DOM-based confetti burst — no canvas, no library. */
function burstConfetti(originX, originY, count = 26) {
  const layer = document.getElementById("confetti-layer");
  if (!layer || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  // --terracotta and --brand are the same purple by design (see styles.css),
  // so only one of them appears here -- otherwise purple would outnumber
  // gold and blue two to one in every burst.
  const colours = ["--terracotta", "--gold", "--blue"];
  for (let i = 0; i < count; i += 1) {
    const piece = document.createElement("div");
    const colour = token(colours[i % colours.length], "#7c4fe0");
    const angle = Math.random() * Math.PI * 2;
    const distance = 60 + Math.random() * 140;
    const dx = Math.cos(angle) * distance;
    const dy = Math.abs(Math.sin(angle) * distance) + 220;
    piece.className = `confetti-piece${Math.random() > 0.5 ? " is-round" : ""}`;
    piece.style.left = `${originX}px`;
    piece.style.top = `${originY}px`;
    piece.style.background = colour;
    piece.style.setProperty("--confetti-end", `translate(${dx}px, ${dy}px) rotate(${360 + Math.random() * 360}deg)`);
    piece.style.animationDuration = `${900 + Math.random() * 500}ms`;
    layer.appendChild(piece);
    piece.addEventListener("animationend", () => piece.remove());
  }
}

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
  user: null,        // signed-in traveller, or null when anonymous
  loginEnabled: false,

  // Countries visited. `countriesPicked` holds explicit choices only; the
  // countries implied by cities in the trip history are derived, never stored,
  // so the two views can never disagree.
  countriesPicked: new Set(),
  countryCatalog: [],
  countryByCode: new Map(),
  worldMap: null,
  worldLayer: null,
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

/* -------------------------------------------------------------- account */

/** Persist the history server-side, but only for a signed-in traveller. */
async function syncTrips() {
  if (!state.user) return;
  try {
    await api("/me/trips", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history: state.history }),
    });
    markSynced();
  } catch (error) {
    // A failed sync must not break the session; the in-memory history stands.
    console.warn("Could not save trips:", error.message);
  }
}

async function syncPreferences() {
  if (!state.user) return;
  try {
    await api("/me/preferences", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        interests: state.interests,
        duration_days: state.duration,
        budget: state.budget,
      }),
    });
  } catch (error) {
    console.warn("Could not save preferences:", error.message);
  }
}

function markSynced() {
  // The profile menu shows sync state; nothing else to update.
}

/** Load auth state; restore a signed-in traveller's stored history. */
async function loadSession() {
  let config;
  try {
    config = await api("/auth/config");
  } catch (error) {
    return false;
  }
  state.loginEnabled = Boolean(config.enabled);
  state.user = config.user || null;

  const button = $("google-signin");
  const note = $("auth-note");
  if (state.loginEnabled) {
    button.hidden = false;
    note.textContent =
      "Signing in saves your trips so your profile is there next time.";
  } else {
    button.hidden = true;
    note.textContent =
      "Google Sign-In is not configured on this deployment, so trips are kept " +
      "in this browser tab only. Everything else works exactly the same.";
  }

  if (state.user) {
    const [trips, preferences] = await Promise.all([
      api("/me/trips").catch(() => ({ history: [] })),
      api("/me/preferences").catch(() => ({})),
    ]);
    if (trips.history && trips.history.length) {
      state.history = trips.history;
      state.origin = state.history[state.history.length - 1];
    }
    const stored = await api("/me/countries").catch(() => ({ countries: [] }));
    state.countriesPicked = new Set(stored.countries || []);
    if (preferences.interests) state.interests = preferences.interests;
    if (preferences.duration_days) state.duration = preferences.duration_days;
    if (preferences.budget) state.budget = preferences.budget;
  }
  renderProfileMenus();
  return Boolean(state.user);
}

/* ---------------------------------------------------------------- router */

const SCREENS = [
  "login", "welcome", "history", "map", "style",
  "building", "profile", "recommend", "trips", "countries",
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
  if (name === "countries") renderCountries();
  // The menu shows a live trip count, so refresh it on every navigation.
  renderProfileMenus();
}

/* ----------------------------------------------------------------- utils */

function flagEmoji(code) {
  if (!code || code.length !== 2) return "";
  return String.fromCodePoint(
    ...[...code.toUpperCase()].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65)
  );
}

/**
 * Render a destination photograph.
 *
 * `hd` picks the 1600px file and advertises both widths through srcset, so the
 * browser downloads the large one only where it is actually large (hero, card,
 * drawer). List rows and search suggestions stay on the 400px file — shipping
 * 1600px into a twenty-row list would cost megabytes for no visible gain.
 */
// How wide the image is actually displayed, per context. Without this the
// browser assumes a huge slot and downloads the 1920px file (~530KB) for a
// 353px-wide card -- nine of those is 4.7MB for no visible benefit.
const PHOTO_SIZES = {
  hero: "(max-width: 860px) 100vw, 50vw",
  card: "(max-width: 640px) 100vw, (max-width: 980px) 50vw, 33vw",
  drawer: "(max-width: 640px) 100vw, 480px",
  thumb: "190px",
  row: "80px",
};

function photo(destination, className, hd, context) {
  const small = destination && destination.image_url;
  const large = (destination && destination.image_url_hd) || small;
  const medium = (destination && destination.image_url_md) || large;
  if (!small && !large) {
    return `<div class="${className || ""}" style="background:#ece4d8"></div>`;
  }
  const src = hd ? large : small;
  // Descriptors must be the images' true widths. The API reports the width
  // that was *requested* rather than the one delivered, so these come from
  // the URL; advertising the wrong number makes the browser choose badly.
  const candidates = [
    [small, destination.image_width],
    [destination.image_url_md, destination.image_width_md],
    [large, destination.image_width_hd],
  ].filter(([url, width]) => url && width);
  const seen = new Set();
  const srcset = candidates
    .filter(([, width]) => !seen.has(width) && seen.add(width))
    .map(([url, width]) => `${url} ${width}w`)
    .join(", ");
  return `<img class="${className || ""}" src="${src}"
    ${srcset ? `srcset="${srcset}" sizes="${PHOTO_SIZES[context] || (hd ? PHOTO_SIZES.card : PHOTO_SIZES.row)}"` : ""}
    alt="${destination.city}" loading="lazy"
    onerror="this.style.visibility='hidden'" />`;
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
  renderProfileMenus();
  syncTrips();
}

function removeCity(id) {
  state.history = state.history.filter((h) => h !== id);
  if (state.origin === id) state.origin = state.history[state.history.length - 1] || null;
  renderVisited();
  renderTimeline();
  renderProfileMenus();
  syncTrips();
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
    L.polyline(coords, { color: token("--charcoal", "#2a1f1b"), weight: 1, opacity: 0.35, dashArray: "3 5" })
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
        ${photo(d, "", true, "thumb")}
        <div class="trip-name">${d.city}<small>${d.country}</small></div>
      </div>`
    )
    .join("");

  renderLevelBanner();
}

/** The celebratory level strip at the top of the Travel Profile screen. */
function renderLevelBanner() {
  const banner = $("level-banner");
  if (!banner) return;
  const level = computeLevel(computeXP());
  banner.innerHTML = `
    <span class="level-badge">${level.current.icon}</span>
    <div class="level-text">
      <span class="level-name">You're a ${level.current.name}!</span>
      <span class="level-xp">${level.xp} XP earned from your countries and trips${
        level.next ? ` · ${level.next.min - level.xp} XP to ${level.next.name}` : ""
      }</span>
    </div>`;
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

/* Presentation sugar only: the medal restates the rank the card is already
   shown in (the same score, same order), it is never a second signal. */
const MEDALS = ["🥇", "🥈", "🥉"];

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
          ${photo(item, '', true, 'card')}
          ${MEDALS[index] ? `<span class="medal">${MEDALS[index]}</span>` : ""}
          <span class="match-badge">
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <polygon points="12 2 15 9 22 9.5 16.5 14.5 18.5 22 12 17.8 5.5 22 7.5 14.5 2 9.5 9 9" />
            </svg>
            ${match}% match
          </span>
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
    ${item.image_url_hd || item.image_url
      ? `<img class="drawer-photo" src="${item.image_url_hd || item.image_url}" alt="${item.city}" />`
      : ""}
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



/* ============================================================ countries */

/**
 * Every country the traveller counts as visited.
 *
 * Two sources are unioned: countries they ticked explicitly, and the countries
 * of the cities already in their trip history. Deriving the second means the
 * map is never out of step with the trips list — adding Amsterdam colours the
 * Netherlands without asking the user to say it twice.
 */
function visitedCountryCodes() {
  const fromCities = state.history
    .map((id) => state.byId.get(id))
    .filter(Boolean)
    .map((d) => d.country_code);
  return new Set([...state.countriesPicked, ...fromCities]);
}

/** Countries implied by a city, which therefore cannot be removed directly. */
function impliedCountryCodes() {
  return new Set(
    state.history.map((id) => state.byId.get(id)).filter(Boolean).map((d) => d.country_code)
  );
}

async function loadCountryCatalog() {
  const data = await api("/countries");
  state.countryCatalog = data.countries;
  state.countryByCode = new Map(data.countries.map((c) => [c.country_code, c]));
}

/** Read a CSS custom property, so map fills follow the active theme. */
function token(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function countryStyle(code, visited) {
  const stroke = token("--map-stroke", "#ffffff");
  return visited
    ? { fillColor: token("--map-visited", "#c4622d"), fillOpacity: 0.88, color: stroke, weight: 0.6 }
    : { fillColor: token("--map-land", "#dfd8cc"), fillOpacity: 0.9, color: stroke, weight: 0.6 };
}

async function drawWorldMap() {
  const container = $("world-map");
  if (typeof L === "undefined" || !container) return;

  const visited = visitedCountryCodes();

  if (!state.worldMap) {
    state.worldMap = L.map("world-map", {
      scrollWheelZoom: false,
      attributionControl: false,
      worldCopyJump: true,
      minZoom: 1,
    }).setView([25, 8], 2);

    // No tile layer: the basemap IS the country polygons. That keeps the look
    // flat and editorial, avoids 200+ tile requests, and means the map works
    // with no network at all once the GeoJSON is cached.
    const geojson = await fetch("/static/countries.geojson").then((r) => r.json());

    state.worldLayer = L.geoJSON(geojson, {
      style: (feature) => countryStyle(feature.properties.iso, visited.has(feature.properties.iso)),
      onEachFeature: (feature, layer) => {
        const { iso, name } = feature.properties;
        layer.bindTooltip(name, { sticky: true });
        layer.on("click", () => toggleCountry(iso));
        layer.on("mouseover", () => layer.setStyle({ weight: 1.6, color: token("--charcoal", "#2a1f1b") }));
        layer.on("mouseout", () =>
          layer.setStyle(countryStyle(iso, visitedCountryCodes().has(iso)))
        );
      },
    }).addTo(state.worldMap);
  } else {
    repaintWorldMap();
  }

  // Same zero-width trap as the history map: the screen is hidden when the
  // map is created, so Leaflet measures the container as 0x0.
  const fitWhenSized = (attempt = 0) => {
    if (container.clientWidth < 80 && attempt < 30) {
      requestAnimationFrame(() => fitWhenSized(attempt + 1));
      return;
    }
    state.worldMap.invalidateSize();
  };
  fitWhenSized();
}

function repaintWorldMap() {
  if (!state.worldLayer) return;
  const visited = visitedCountryCodes();
  state.worldLayer.eachLayer((layer) => {
    const iso = layer.feature.properties.iso;
    layer.setStyle(countryStyle(iso, visited.has(iso)));
  });
}

/** Bounding-box centre of the XP pill in the currently visible header. */
function xpPillOrigin() {
  const active = document.querySelector(".screen.is-active .xp-pill");
  if (!active) return { x: window.innerWidth - 60, y: 40 };
  const rect = active.getBoundingClientRect();
  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
}

async function toggleCountry(code) {
  const implied = impliedCountryCodes();
  if (implied.has(code)) {
    const country = state.countryByCode.get(code);
    notice(
      `${country ? country.name : code} is on your map because of a city in your trips. ` +
        "Remove the city from My trips to clear it."
    );
    return;
  }

  const badgesBefore = computeBadges().filter((b) => b.unlocked).map((b) => b.id);
  const adding = !state.countriesPicked.has(code);

  if (adding) {
    state.countriesPicked.add(code);
  } else {
    state.countriesPicked.delete(code);
  }
  await persistCountries();
  renderCountries();
  renderProfileMenus();

  if (adding) {
    const origin = xpPillOrigin();
    const badgesAfter = computeBadges().filter((b) => b.unlocked).map((b) => b.id);
    const newlyUnlocked = badgesAfter.filter((id) => !badgesBefore.includes(id));
    burstConfetti(origin.x, origin.y, newlyUnlocked.length ? 42 : 24);
    document.querySelectorAll(".screen.is-active .xp-pill").forEach((pill) => {
      pill.classList.remove("is-bumped");
      // Re-trigger the CSS animation even if it just played.
      void pill.offsetWidth;
      pill.classList.add("is-bumped");
    });
  }
}

/** Push the explicit picks to the server when the traveller is signed in. */
async function persistCountries() {
  if (!state.user) return;
  try {
    await api("/me/countries", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ countries: [...state.countriesPicked] }),
    });
  } catch (error) {
    notice(`Could not save your countries: ${error.message}`);
  }
}

function renderCountries() {
  const visited = visitedCountryCodes();
  const implied = impliedCountryCodes();
  const total = state.countryCatalog.length || 175;

  const continents = new Set(
    [...visited].map((c) => (state.countryByCode.get(c) || {}).continent).filter(Boolean)
  );

  $("country-stats").innerHTML = `
    <div class="stat-block">
      <span class="stat-value">${visited.size}<span class="stat-of"> / ${total}</span></span>
      <span class="label">Countries visited</span>
    </div>
    <div class="stat-block">
      <span class="stat-value">${Math.round((visited.size / total) * 100)}<span class="stat-of">%</span></span>
      <span class="label">Of the world</span>
    </div>
    <div class="stat-block">
      <span class="stat-value">${continents.size}<span class="stat-of"> / 6</span></span>
      <span class="label">Continents</span>
    </div>`;

  const sorted = [...visited]
    .map((code) => state.countryByCode.get(code) || { country_code: code, name: code })
    .sort((a, b) => a.name.localeCompare(b.name));

  $("country-chips").innerHTML = sorted.length
    ? sorted
        .map((c) => {
          const locked = implied.has(c.country_code);
          return `<span class="country-chip">
            ${flagEmoji(c.country_code)} ${c.name}
            ${locked ? '<span class="cc-count">from trips</span>' : ""}
            ${locked ? "" : `<button class="cc-remove" data-country-remove="${c.country_code}"
                 aria-label="Remove ${c.name}">×</button>`}
          </span>`;
        })
        .join("")
    : '<p class="empty-note">No countries yet — tap the map or use the search above.</p>';

  document.querySelectorAll("[data-country-remove]").forEach((button) =>
    button.addEventListener("click", () => toggleCountry(button.dataset.countryRemove))
  );

  renderBadges();
  repaintWorldMap();
  drawWorldMap();
}

/** The badge grid on the Countries screen — locked vs unlocked, both honest:
 *  a badge shows unlocked only when its real-data test actually passes. */
function renderBadges() {
  const grid = $("badge-grid");
  if (!grid) return;
  grid.innerHTML = computeBadges()
    .map(
      (badge) => `<div class="badge ${badge.unlocked ? "is-unlocked" : "is-locked"}">
        <span class="badge-icon">${badge.icon}</span>
        <span class="badge-label">${badge.label}</span>
        <span class="badge-note">${badge.unlocked ? "Unlocked" : "Locked"}</span>
      </div>`
    )
    .join("");
}

function attachCountrySearch() {
  const input = $("country-search");
  const list = $("country-suggestions");
  if (!input) return;

  const close = () => {
    list.hidden = true;
    list.innerHTML = "";
  };

  input.addEventListener("input", () => {
    const needle = input.value.trim().toLowerCase();
    if (needle.length < 2) return close();
    const visited = visitedCountryCodes();
    const matches = state.countryCatalog
      .filter((c) => !visited.has(c.country_code) && c.name.toLowerCase().includes(needle))
      .slice(0, 7);
    if (!matches.length) return close();
    list.innerHTML = matches
      .map(
        (c) => `<li data-code="${c.country_code}">
          <span style="font-size:20px">${flagEmoji(c.country_code)}</span>
          <span><strong>${c.name}</strong>
            <span class="s-country">${c.continent}</span></span>
        </li>`
      )
      .join("");
    list.hidden = false;
  });

  list.addEventListener("click", async (event) => {
    const li = event.target.closest("li");
    if (!li) return;
    input.value = "";
    close();
    // Routed through toggleCountry() rather than touching state directly, so
    // the XP pill refresh, badge check and confetti burst all fire here too
    // — a country added by search is otherwise indistinguishable from one
    // added by tapping the map. The list only ever offers countries that are
    // not already visited, so this always adds, never removes.
    await toggleCountry(li.dataset.code);
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest("#country-search") && !event.target.closest("#country-suggestions")) {
      close();
    }
  });
}


/* ================================================================ theme */

/**
 * Light / dark / system appearance.
 *
 * The choice is written to <html data-theme> because CSS custom properties
 * are scoped to :root, and persisted to localStorage so it survives a reload.
 * "system" removes the attribute and re-applies whatever the OS reports, and
 * keeps listening — so a machine that flips to dark at sunset follows along
 * without the user touching anything.
 */
const THEME_KEY = "waygo-theme";
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

/**
 * Dark mode: one button, two states.
 *
 * The first visit follows the operating system. After that the traveller's
 * own choice wins and is remembered, because someone who reached for the
 * toggle has said what they want more clearly than the OS setting has.
 */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem(THEME_KEY, theme);
  // Leaflet writes country fills inline, so a token change alone will not
  // repaint them.
  if (typeof repaintWorldMap === "function") repaintWorldMap();
}

function toggleTheme() {
  const next =
    document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  applyTheme(next);
}

function wireTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  applyTheme(stored || (systemDark.matches ? "dark" : "light"));

  document.querySelectorAll("[data-theme-toggle]").forEach((button) =>
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleTheme();
    })
  );

  // Follow the OS only until the traveller expresses a preference.
  systemDark.addEventListener("change", (event) => {
    if (!localStorage.getItem(THEME_KEY)) applyTheme(event.matches ? "dark" : "light");
  });
}

/* ============================================================== profile */

function renderProfileMenus() {
  const signedIn = Boolean(state.user);
  const level = computeLevel(computeXP());

  document.querySelectorAll("[data-profile]").forEach((wrapper) => {
    const name = wrapper.querySelector(".profile-name");
    const sub = wrapper.querySelector(".profile-sub");
    const signin = wrapper.querySelector("[data-profile-signin]");
    const avatar = wrapper.querySelector(".avatar");

    name.textContent = signedIn ? state.user.name || "Traveller" : "Guest";
    sub.textContent = signedIn
      ? state.user.email || "Signed in"
      : `${state.history.length} trips · not signed in`;

    if (signedIn && state.user.picture) {
      avatar.innerHTML = `<img src="${state.user.picture}" alt="" referrerpolicy="no-referrer" />`;
    }
    if (signin) {
      signin.textContent = signedIn ? "Sign out" : "Sign in to save";
      signin.hidden = !signedIn && !state.loginEnabled;
    }

    const xpPill = wrapper.querySelector(".xp-value");
    if (xpPill) xpPill.textContent = `${level.xp} XP`;

    const badge = wrapper.querySelector(".level-badge");
    if (badge) badge.textContent = level.current.icon;
    const levelName = wrapper.querySelector(".level-name");
    if (levelName) levelName.textContent = level.current.name;
    const levelXp = wrapper.querySelector(".level-xp");
    if (levelXp) {
      levelXp.textContent = level.next
        ? `${level.xp} / ${level.next.min} XP to ${level.next.name}`
        : `${level.xp} XP · max level`;
    }
    const xpBar = wrapper.querySelector(".xp-bar span");
    if (xpBar) xpBar.style.width = `${Math.round(level.progress * 100)}%`;
  });
}

function wireProfileMenus() {
  document.querySelectorAll("[data-profile]").forEach((wrapper) => {
    const avatar = wrapper.querySelector(".avatar");
    const xpPill = wrapper.querySelector(".xp-pill");
    const menu = wrapper.querySelector(".profile-menu");

    const toggleMenu = (event) => {
      event.stopPropagation();
      const open = menu.hidden;
      // Only one menu open at a time across the app's three headers.
      document.querySelectorAll(".profile-menu").forEach((m) => (m.hidden = true));
      document
        .querySelectorAll(".avatar, .xp-pill")
        .forEach((a) => a.setAttribute("aria-expanded", "false"));
      menu.hidden = !open;
      avatar.setAttribute("aria-expanded", String(open));
      if (xpPill) xpPill.setAttribute("aria-expanded", String(open));
    };

    avatar.addEventListener("click", toggleMenu);
    if (xpPill) xpPill.addEventListener("click", toggleMenu);

    menu.querySelectorAll("[data-view]").forEach((item) =>
      item.addEventListener("click", () => {
        menu.hidden = true;
        avatar.setAttribute("aria-expanded", "false");
        show(item.dataset.view);
      })
    );

    const signin = wrapper.querySelector("[data-profile-signin]");
    if (signin) {
      signin.addEventListener("click", async () => {
        if (state.user) {
          await api("/auth/logout", { method: "POST" }).catch(() => {});
          state.user = null;
          renderProfileMenus();
          menu.hidden = true;
        } else {
          window.location.href = "/auth/login?next=/";
        }
      });
    }

    const reset = wrapper.querySelector("[data-profile-reset]");
    if (reset) {
      reset.addEventListener("click", () => {
        state.history = [];
        state.countriesPicked = new Set();
        state.origin = null;
        state.results = [];
        menu.hidden = true;
        renderVisited();
        renderTimeline();
        show("login");
      });
    }
  });

  document.addEventListener("click", () => {
    document.querySelectorAll(".profile-menu").forEach((m) => (m.hidden = true));
    document
      .querySelectorAll(".avatar, .xp-pill")
      .forEach((a) => a.setAttribute("aria-expanded", "false"));
  });
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
      syncPreferences();
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
  document.querySelectorAll("#nav button, #nav-trips button, #nav-countries button").forEach((button) =>
    button.addEventListener("click", () => show(button.dataset.view))
  );

  attachCountrySearch();
  wireProfileMenus();
  wireTheme();

  // The brand in the header returns to the entry menu. History and
  // preferences are kept in state, so nothing is lost by going back.
  document.querySelectorAll("[data-home]").forEach((button) =>
    button.addEventListener("click", () => show("login"))
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
    await loadCountryCatalog().catch(() => {});
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
      $("login-photo").src = hero.image_url_hd || hero.image_url;
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

  const signedIn = await loadSession();
  renderChoices();
  renderVisited();
  renderRefine();
  renderTimeline();

  // A returning traveller with a saved history should not be asked to enter
  // it again: build their profile and go straight to recommendations.
  if (signedIn && state.history.length) {
    state.profile = await postJson("/profile", { history: state.history }).catch(() => null);
    await fetchRecommendations().catch(() => {});
    show("recommend");
  }
}

main();
