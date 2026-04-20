import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction and ratio features for livability ranking."""
    out = df.copy()
    eps = 1e-6

    out["Affordability_Balance"] = (
        out["Purchasing Power Index"] / (out["Cost of Living Index"] + eps)
    )
    out["Health_Safety_Synergy"] = (
        out["Health Care Index"] * out["Safety Index"]
    )
    out["Environment_Balance"] = (
        out["Climate Index"] - out["Pollution Index"] - out["Air_Pollution_2023"]
    )
    out["Education_Internet_Synergy"] = (
        out["Education"] * out["Internet Access"]
    )
    out["Mobility_Stress"] = (
        out["Traffic Commute Time Index"] * out["Pollution Index"]
    )
    out["Tax_Adjusted_Power"] = (
        out["Purchasing Power Index"] - out["Taxation"]
    )
    return out


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "processed_livable_cities.csv")
    df = add_engineered_features(df)

    # Use rank-derived score as target for recommendation.
    # Higher score = better city rank.
    max_rank = df["Rank"].max()
    df["Livability_Score"] = max_rank + 1 - df["Rank"]

    feature_cols = [
        col
        for col in df.columns
        if col not in ["Rank", "City", "Country", "Livability_Score"]
    ]
    X = df[feature_cols]
    y = df["Livability_Score"]

    model = GradientBoostingRegressor(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print("Holdout performance")
    print(f"MAE: {mae:.3f}")
    print(f"R2 : {r2:.4f}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=-1,
    )
    print("\n5-Fold CV")
    print(f"MAE: {-cv_scores['test_mae'].mean():.3f}")
    print(f"R2 : {cv_scores['test_r2'].mean():.4f}")

    importances = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(12)
    )
    print("\nTop engineered + original features")
    print(importances.to_string())


if __name__ == "__main__":
    main()
