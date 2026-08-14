# Exploratory data analysis

## Destination catalog (real data)

- **400** destinations across **107** countries and **6** continents
- OpenStreetMap attributes available for **100.0%**
- Climate normals available for **100.0%**
- Country cost proxy available for **99.2%**

### Continent breakdown

| Continent | Destinations | Share |
|---|---:|---:|
| Europe | 128 | 32.0% |
| North America | 126 | 31.5% |
| Asia | 99 | 24.8% |
| Africa | 24 | 6.0% |
| Oceania | 12 | 3.0% |
| South America | 11 | 2.8% |

### Most popular destinations (Wikipedia pageview proxy)

| Rank | City | Country | Mean monthly views |
|---:|---|---|---:|
| 1 | New York City | United States | 445,596 |
| 2 | Singapore | Singapore | 395,723 |
| 3 | Utrecht | The Netherlands | 362,348 |
| 4 | Durrës | Albania | 349,398 |
| 5 | Brooklyn | United States | 293,322 |
| 6 | Hong Kong | Hong Kong | 267,869 |
| 7 | London | United Kingdom | 261,005 |
| 8 | Washington | United States | 250,962 |
| 9 | Los Angeles | United States | 248,618 |
| 10 | Chicago | United States | 186,395 |

## Interactions (SYNTHETIC — not real travellers)

- **4,000** generated users, **29,852** trips
- Trips per user: min 3, median 7, max 12
- Destinations ever visited: **400/400** (100.0% of the catalog)
- The most-visited 10% of destinations account for **45.1%** of all trips, which is the popularity bias the models must be measured against

### Generative mechanism mix

| Mechanism | Share |
|---|---:|
| preference | 39.8% |
| geographic | 30.3% |
| popularity | 29.9% |
