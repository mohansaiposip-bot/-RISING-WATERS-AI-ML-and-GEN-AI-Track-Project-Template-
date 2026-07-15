"""
Rising Waters — synthetic historical weather dataset generator.

Generates a flood-prediction dataset with realistic relationships between
meteorological features and flood occurrence. No public flood dataset was
supplied with this project, so this script builds a physically-plausible
synthetic one: heavier annual/seasonal rainfall, low cloud visibility (storm
cover), high humidity, sustained rain-days, and high river discharge all
push flood probability up, combined with region-level baseline risk and
random noise so the classes stay realistically overlapping (not perfectly
separable) — the way real climate data behaves.

Swap this out for a real historical dataset (e.g. India Meteorological
Department / Kaggle flood datasets) by replacing data/flood_dataset.csv
with the same column schema and re-running train_model.py.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SAMPLES = 6000

REGIONS = [
    "Kurnool", "Guwahati", "Patna", "Kochi", "Surat",
    "Chennai", "Bhubaneswar", "Lucknow", "Varanasi", "Kolkata",
]
# Each region gets a baseline flood-proneness (low-lying delta / riverbank
# regions run hotter) so the model also implicitly learns geography effects.
REGION_BASE_RISK = {r: v for r, v in zip(
    REGIONS, RNG.uniform(-0.6, 0.9, size=len(REGIONS))
)}


def generate(n=N_SAMPLES, seed=42):
    rng = np.random.default_rng(seed)
    regions = rng.choice(REGIONS, size=n)

    annual_rainfall_mm = rng.normal(1800, 650, n).clip(300, 4200)
    # Seasonal (monsoon) rainfall is the share of annual rainfall that falls
    # in the flood-risk season — typically 55-85% in monsoon-affected areas.
    monsoon_share = rng.uniform(0.45, 0.9, n)
    seasonal_rainfall_mm = (annual_rainfall_mm * monsoon_share).clip(100, 3600)

    # Cloud visibility drops as storm intensity rises; add independent noise.
    storm_intensity = (seasonal_rainfall_mm / seasonal_rainfall_mm.max())
    cloud_visibility_km = (9.5 - storm_intensity * 7.5 + rng.normal(0, 1.1, n)).clip(0.3, 10)

    humidity_percent = (55 + storm_intensity * 30 + rng.normal(0, 6, n)).clip(30, 100)
    temperature_c = rng.normal(29, 4, n).clip(14, 42)

    days_continuous_rainfall = (storm_intensity * 12 + rng.normal(0, 2, n)).clip(0, 20)
    river_discharge_cumecs = (
        800 + seasonal_rainfall_mm * 1.8 + days_continuous_rainfall * 120
        + rng.normal(0, 400, n)
    ).clip(200, 12000)

    base_risk = np.array([REGION_BASE_RISK[r] for r in regions])

    # Standardize the strongest predictors, combine into a logit, add noise.
    def z(x):
        return (x - x.mean()) / x.std()

    logit = (
        2.1 * z(seasonal_rainfall_mm)
        + 1.15 * z(annual_rainfall_mm)
        - 1.5 * z(cloud_visibility_km)
        + 0.85 * z(days_continuous_rainfall)
        + 0.75 * z(river_discharge_cumecs)
        + 0.45 * z(humidity_percent)
        + 1.1 * base_risk
        - 0.6
        + rng.normal(0, 0.55, n)
    )
    prob_flood = 1 / (1 + np.exp(-logit))
    flood_occurred = (rng.uniform(0, 1, n) < prob_flood).astype(int)

    df = pd.DataFrame({
        "region": regions,
        "annual_rainfall_mm": annual_rainfall_mm.round(1),
        "seasonal_rainfall_mm": seasonal_rainfall_mm.round(1),
        "cloud_visibility_km": cloud_visibility_km.round(2),
        "humidity_percent": humidity_percent.round(1),
        "temperature_c": temperature_c.round(1),
        "days_continuous_rainfall": days_continuous_rainfall.round(1),
        "river_discharge_cumecs": river_discharge_cumecs.round(1),
        "flood_occurred": flood_occurred,
    })
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/rising_waters/data/flood_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df["flood_occurred"].value_counts(normalize=True))
