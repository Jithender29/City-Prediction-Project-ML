from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs" / "phase3"


@dataclass
class RecommendationWeights:
    model_weight: float = 0.6
    preference_weight: float = 0.4


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
    out["Mobility_Stress"] = out["Traffic Commute Time Index"] * out["Pollution Index"]
    out["Tax_Adjusted_Power"] = out["Purchasing Power Index"] - out["Taxation"]
    return out


def scale_zero_one(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()
    if max_v - min_v == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


def get_user_preferences(attributes: dict[str, str]) -> dict[str, float]:
    print("\nAll attributes are included in optimization.")
    print("Set preference importance from 0 to 10 for each attribute.")
    print("Press Enter at importance prompt to use 5.\n")

    preferences: dict[str, float] = {}
    for attr, direction in attributes.items():
        while True:
            raw = input(f"{attr} ({direction} preferred), importance [0-10]: ").strip()
            if raw == "":
                preferences[attr] = 5.0
                break
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number between 0 and 10.")
                continue

            if 0 <= value <= 10:
                preferences[attr] = value
                break
            print("Value must be in the range [0, 10].")

    return preferences


def train_model(df: pd.DataFrame) -> tuple[GradientBoostingRegressor, list[str], pd.Series]:
    model_df = add_engineered_features(df)
    max_rank = model_df["Rank"].max()
    model_df["Livability_Score"] = max_rank + 1 - model_df["Rank"]

    feature_cols = [
        c for c in model_df.columns if c not in ["Rank", "City", "Country", "Livability_Score"]
    ]
    X = model_df[feature_cols]
    y = model_df["Livability_Score"]

    model = GradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=2,
        min_samples_leaf=1,
        n_estimators=400,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model, feature_cols, y


def recommend_cities(
    df: pd.DataFrame,
    model: GradientBoostingRegressor,
    feature_cols: list[str],
    preferences: dict[str, float],
    combine_weights: RecommendationWeights,
) -> pd.DataFrame:
    scored_df = add_engineered_features(df.copy())
    scored_df["Predicted_Livability"] = model.predict(scored_df[feature_cols])
    scored_df["Model_Score_0_1"] = scale_zero_one(scored_df["Predicted_Livability"])

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

    pref_score = pd.Series(0.0, index=scored_df.index)
    total_weight = sum(preferences.values())

    if total_weight == 0:
        scored_df["Preference_Score_0_1"] = pd.Series(0.5, index=scored_df.index)
        scored_df["Final_Suitability_Score"] = (
            combine_weights.model_weight * scored_df["Model_Score_0_1"]
            + combine_weights.preference_weight * scored_df["Preference_Score_0_1"]
        )
        result_cols = [
            "City",
            "Country",
            "Predicted_Livability",
            "Model_Score_0_1",
            "Preference_Score_0_1",
            "Final_Suitability_Score",
        ]
        return scored_df[result_cols].sort_values("Final_Suitability_Score", ascending=False)

    for attr, direction in attribute_direction.items():
        if preferences.get(attr, 0.0) <= 0:
            continue
        normalized = scale_zero_one(scored_df[attr])
        desirability = normalized if direction == "higher" else 1 - normalized
        weighted = desirability * (preferences.get(attr, 0.0) / total_weight)
        pref_score = pref_score + weighted

    scored_df["Preference_Score_0_1"] = pref_score
    scored_df["Final_Suitability_Score"] = (
        combine_weights.model_weight * scored_df["Model_Score_0_1"]
        + combine_weights.preference_weight * scored_df["Preference_Score_0_1"]
    )

    result_cols = [
        "City",
        "Country",
        "Predicted_Livability",
        "Model_Score_0_1",
        "Preference_Score_0_1",
        "Final_Suitability_Score",
    ]
    return scored_df[result_cols].sort_values("Final_Suitability_Score", ascending=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "processed_livable_cities.csv")
    model, feature_cols, _ = train_model(df)

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

    selected_total_weight = sum(preferences.values())
    if selected_total_weight > 0:
        combine_weights = RecommendationWeights(model_weight=0.0, preference_weight=1.0)
        print("\nUsing your selected attributes as the optimization target (preference-first ranking).")
    else:
        combine_weights = RecommendationWeights(model_weight=1.0, preference_weight=0.0)
        print("\nNo attributes selected; falling back to baseline livability model ranking.")

    ranked = recommend_cities(
        df=df,
        model=model,
        feature_cols=feature_cols,
        preferences=preferences,
        combine_weights=combine_weights,
    )

    best = ranked.iloc[0]
    top5 = ranked.head(5).copy()

    print("\nMost optimal city based on your selected attributes:")
    print(
        f"1) {best['City']}, {best['Country']} | "
        f"Final={best['Final_Suitability_Score']:.4f} | "
        f"Model={best['Model_Score_0_1']:.4f} | "
        f"Preference={best['Preference_Score_0_1']:.4f}"
    )

    print("\nTop 5 suitable cities:")
    for idx, row in top5.reset_index(drop=True).iterrows():
        print(
            f"{idx+1}) {row['City']}, {row['Country']} | "
            f"Final={row['Final_Suitability_Score']:.4f}"
        )

    print("\nCSV export is disabled; existing outputs/phase3/city_recommendation_results.csv was not modified.")


if __name__ == "__main__":
    main()
