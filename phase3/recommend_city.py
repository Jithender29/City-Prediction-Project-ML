from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


@dataclass
class RecommendationWeights:
    model_weight: float = 0.2
    preference_weight: float = 0.8


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    out["Affordability_Balance"] = (
        out["Purchasing Power Index"] / (out["Cost of Living Index"] + eps)
    )
    out["Health_Safety_Synergy"] = out["Health Care Index"] * out["Safety Index"]
    out["Environment_Balance"] = (
        out["Climate Index"] - out["Pollution Index"] - out["Air_Pollution_2023"]
    )

    return out


def scale_zero_one(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()

    if max_v - min_v == 0:
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - min_v) / (max_v - min_v)


def get_user_preferences(attributes: dict[str, str]) -> dict[str, float]:
    print("\nEnter importance (0–10). At least ONE must be > 0.\n")

    preferences = {}

    for attr, direction in attributes.items():
        while True:
            raw = input(f"{attr} ({direction}), importance: ").strip()

            if raw == "":
                preferences[attr] = 0.0
                break

            try:
                val = float(raw)
                if 0 <= val <= 10:
                    preferences[attr] = val
                    break
            except:
                pass

            print("Enter a number between 0 and 10.")

    if sum(preferences.values()) == 0:
        raise ValueError("At least one preference must be > 0")

    return preferences


def train_model(df):
    df = add_engineered_features(df)

    max_rank = df["Rank"].max()
    df["Livability_Score"] = max_rank + 1 - df["Rank"]

    feature_cols = [
        c for c in df.columns
        if c not in ["Rank", "City", "Country", "Livability_Score"]
    ]

    X = df[feature_cols]
    y = df["Livability_Score"]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    return model, feature_cols


def apply_hard_filters(df, preferences):
    """🔥 This is what makes results ACTUALLY change"""

    df = df.copy()

    # Example filters (adjust thresholds as needed)

    if preferences.get("Cost of Living Index", 0) >= 8:
        df = df[df["Cost of Living Index"] < df["Cost of Living Index"].median()]

    if preferences.get("Pollution Index", 0) >= 8:
        df = df[df["Pollution Index"] < df["Pollution Index"].median()]

    if preferences.get("Safety Index", 0) >= 8:
        df = df[df["Safety Index"] > df["Safety Index"].median()]

    return df


def compute_preference_score(df, preferences):
    attribute_direction = {
        "Purchasing Power Index": "higher",
        "Safety Index": "higher",
        "Health Care Index": "higher",
        "Cost of Living Index": "lower",
        "Property Price to Income Ratio": "lower",
        "Traffic Commute Time Index": "lower",
        "Pollution Index": "lower",
        "Climate Index": "higher",
        "Education": "higher",
        "Taxation": "lower",
        "Internet Access": "higher",
    }

    score = pd.Series(0.0, index=df.index)

    for attr, direction in attribute_direction.items():
        weight = preferences.get(attr, 0)

        if weight == 0:
            continue

        normalized = scale_zero_one(df[attr])

        if direction == "lower":
            normalized = 1 - normalized

        # 🚀 STRONG influence (no normalization)
        score += normalized * weight

    return scale_zero_one(score)


def recommend_cities(df, model, feature_cols, preferences, weights):
    df = add_engineered_features(df)

    # 🔥 Apply filtering FIRST
    df = apply_hard_filters(df, preferences)

    # If filtering removes too much, fallback
    if len(df) < 5:
        df = add_engineered_features(df)

    df["Predicted_Livability"] = model.predict(df[feature_cols])
    df["Model_Score"] = scale_zero_one(df["Predicted_Livability"])

    df["Preference_Score"] = compute_preference_score(df, preferences)

    df["Final_Score"] = (
        weights.model_weight * df["Model_Score"]
        + weights.preference_weight * df["Preference_Score"]
    )

    return df.sort_values("Final_Score", ascending=False)[
        ["City", "Country", "Final_Score"]
    ]


def main():
    df = pd.read_csv(PROCESSED_DIR / "processed_livable_cities.csv")

    model, feature_cols = train_model(df)

    attributes = {
        "Purchasing Power Index": "higher",
        "Safety Index": "higher",
        "Health Care Index": "higher",
        "Cost of Living Index": "lower",
        "Property Price to Income Ratio": "lower",
        "Traffic Commute Time Index": "lower",
        "Pollution Index": "lower",
        "Climate Index": "higher",
        "Education": "higher",
        "Taxation": "lower",
        "Internet Access": "higher",
    }

    preferences = get_user_preferences(attributes)

    ranked = recommend_cities(
        df,
        model,
        feature_cols,
        preferences,
        RecommendationWeights(),
    )

    print("\nTop 5 Cities:\n")
    print(ranked.head(5))


if __name__ == "__main__":
    main()