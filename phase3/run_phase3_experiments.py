import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs" / "phase3"


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
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
    out["Tax_Adjusted_Power"] = (
        out["Purchasing Power Index"] - out["Taxation"]
    )
    return out


def main() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (11, 6)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "processed_livable_cities.csv")
    model_df = add_engineered_features(df)

    max_rank = model_df["Rank"].max()
    model_df["Livability_Score"] = max_rank + 1 - model_df["Rank"]

    feature_cols = [
        c for c in model_df.columns if c not in ["Rank", "City", "Country", "Livability_Score"]
    ]
    X = model_df[feature_cols]
    y = model_df["Livability_Score"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model_grids = {
        "RandomForest": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [200, 400, 800],
                "max_depth": [None, 8, 16],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(random_state=42),
            {
                "n_estimators": [200, 400, 800],
                "max_depth": [None, 8, 16],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200, 400],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0],
                "min_samples_leaf": [1, 2],
            },
        ),
    }

    all_outcomes = []
    best_estimators = {}
    for model_name, (base_model, grid) in model_grids.items():
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=grid,
            scoring="r2",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        grid_search.fit(X, y)
        best_estimators[model_name] = grid_search.best_estimator_
        result_df = pd.DataFrame(grid_search.cv_results_)
        result_df["model"] = model_name
        result_df = result_df[
            ["model", "mean_test_score", "std_test_score", "rank_test_score", "params"]
        ]
        all_outcomes.append(result_df)

    all_results_df = pd.concat(all_outcomes, ignore_index=True)
    all_results_df = all_results_df.sort_values(
        ["mean_test_score", "std_test_score"], ascending=[False, True]
    )

    best_model_name = all_results_df.iloc[0]["model"]
    best_params = all_results_df.iloc[0]["params"]
    best_model = best_estimators[best_model_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    holdout_r2 = r2_score(y_test, y_pred)
    holdout_mae = mean_absolute_error(y_test, y_pred)

    feature_importance_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": best_model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    best_by_model = (
        all_results_df.groupby("model", as_index=False)["mean_test_score"]
        .max()
        .sort_values("mean_test_score", ascending=False)
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=best_by_model, x="model", y="mean_test_score", palette="viridis")
    plt.title("Best Cross-Validated R2 by Model")
    plt.ylabel("Best CV R2")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3_model_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.8, color="royalblue")
    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "--", color="red")
    plt.title(f"Actual vs Predicted Livability Score ({best_model_name})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3_actual_vs_predicted.png", dpi=300)
    plt.close()

    top_fi = feature_importance_df.head(12)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_fi, y="Feature", x="Importance", palette="magma")
    plt.title(f"Top Feature Importances ({best_model_name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase3_feature_importance.png", dpi=300)
    plt.close()

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PHASE 3 REPORT: FEATURE ENGINEERING + HYPERPARAMETER TUNING")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    report_lines.append(f"Total modeling features used: {len(feature_cols)}")
    report_lines.append(
        "Engineered features: "
        + ", ".join([c for c in feature_cols if c not in df.columns])
    )
    report_lines.append("")
    report_lines.append("Best model summary:")
    report_lines.append(f"- Best model: {best_model_name}")
    report_lines.append(f"- Best parameters: {best_params}")
    report_lines.append(
        f"- Best CV R2: {all_results_df.iloc[0]['mean_test_score']:.6f}"
    )
    report_lines.append(f"- Holdout R2: {holdout_r2:.6f}")
    report_lines.append(f"- Holdout MAE: {holdout_mae:.6f}")
    report_lines.append("")
    report_lines.append("ALL HYPERPARAMETER OUTCOMES (sorted by CV R2):")

    for i, row in all_results_df.reset_index(drop=True).iterrows():
        report_lines.append(
            f"{i+1:03d}. Model={row['model']} | CV_R2={row['mean_test_score']:.6f} | "
            f"STD={row['std_test_score']:.6f} | Params={row['params']}"
        )

    report_lines.append("")
    report_lines.append("Top feature importances:")
    for _, row in feature_importance_df.head(15).iterrows():
        report_lines.append(f"- {row['Feature']}: {row['Importance']:.8f}")

    report_lines.append("")
    report_lines.append("Generated visualizations:")
    report_lines.append("- phase3_model_comparison.png")
    report_lines.append("- phase3_actual_vs_predicted.png")
    report_lines.append("- phase3_feature_importance.png")

    with open(OUTPUT_DIR / "phase3_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    all_results_df.to_csv(OUTPUT_DIR / "phase3_all_hyperparameter_outcomes.csv", index=False)

    print("Best model:", best_model_name)
    print("Best CV R2:", round(all_results_df.iloc[0]["mean_test_score"], 6))
    print("Holdout R2:", round(holdout_r2, 6))
    print("Holdout MAE:", round(holdout_mae, 6))


if __name__ == "__main__":
    main()
