from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


@dataclass
class RecommendationConfig:
    top_n: int = 10


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
    out["Education_Internet_Synergy"] = out["Education"] * out["Internet Access"]
    out["Mobility_Stress"] = (
        out["Traffic Commute Time Index"] * out["Pollution Index"]
    )
    out["Tax_Adjusted_Power"] = out["Purchasing Power Index"] - out["Taxation"]

    return out


def scale_zero_one(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()

    if max_v - min_v == 0:
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - min_v) / (max_v - min_v)


def get_user_attribute_values(attributes: dict[str, str]) -> dict[str, float]:
    print("\nEnter desired attribute level (0-10).")
    print("10 means strongest preference in the better direction shown.\n")

    values: dict[str, float] = {}

    for attr, direction in attributes.items():
        while True:
            raw = input(f"{attr} ({direction}), level [0-10]: ").strip()
            try:
                val = float(raw)
                if 0 <= val <= 10:
                    values[attr] = val
                    break
            except ValueError:
                pass

            print("Enter a number between 0 and 10.")

    return values


def train_model(df: pd.DataFrame):
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


def convert_user_value_to_feature_scale(
    series: pd.Series, direction: str, user_value_0_to_10: float
) -> float:
    min_v = float(series.min())
    max_v = float(series.max())

    if max_v - min_v == 0:
        return min_v

    desirability = user_value_0_to_10 / 10.0
    if direction == "higher":
        return min_v + desirability * (max_v - min_v)

    return max_v - desirability * (max_v - min_v)


def build_user_profile_row(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    attribute_directions: dict[str, str],
    user_values: dict[str, float],
) -> pd.DataFrame:
    profile = {col: float(model_df[col].median()) for col in feature_cols}

    for attr, direction in attribute_directions.items():
        profile[attr] = convert_user_value_to_feature_scale(
            model_df[attr], direction, user_values[attr]
        )

    return pd.DataFrame([profile])


def score_all_cities(
    source_df: pd.DataFrame, model: GradientBoostingRegressor, feature_cols: list[str]
) -> pd.DataFrame:
    scored = add_engineered_features(source_df)
    scored["Predicted_Livability"] = model.predict(scored[feature_cols])
    return scored


def find_closest_cities_by_score(
    scored_df: pd.DataFrame, target_score: float, top_n: int
) -> pd.DataFrame:
    ranked = scored_df.copy()
    ranked["Score_Distance"] = (ranked["Predicted_Livability"] - target_score).abs()
    ranked = ranked.sort_values(
        ["Score_Distance", "Predicted_Livability"], ascending=[True, False]
    )
    return ranked[["City", "Country", "Predicted_Livability", "Score_Distance"]].head(top_n)


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

    user_values = get_user_attribute_values(attributes)

    model_df = add_engineered_features(df)
    user_profile = build_user_profile_row(
        model_df=model_df,
        feature_cols=feature_cols,
        attribute_directions=attributes,
        user_values=user_values,
    )
    target_livability_score = float(model.predict(user_profile[feature_cols])[0])

    scored_cities = score_all_cities(df, model, feature_cols)
    nearest_cities = find_closest_cities_by_score(
        scored_df=scored_cities,
        target_score=target_livability_score,
        top_n=RecommendationConfig().top_n,
    )

    print(f"\nPredicted livability score for your input profile: {target_livability_score:.4f}")
    print("\nNearest cities by livability score distance:\n")
    print(nearest_cities.to_string(index=False))


if __name__ == "__main__":
    main()