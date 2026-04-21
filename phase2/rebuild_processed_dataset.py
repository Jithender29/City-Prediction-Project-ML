"""Rebuild processed_livable_cities.csv from merged_livable_cities.csv."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs" / "phase2"

CORE_FEATURES = [
    "Purchasing Power Index",
    "Safety Index",
    "Health Care Index",
    "Cost of Living Index",
    "Property Price to Income Ratio",
    "Traffic Commute Time Index",
    "Pollution Index",
    "Climate Index",
]
SPARSE_FEATURES = [
    "Meal, Inexpensive Restaurant (USD)",
    "Average Monthly Net Salary (After Tax)",
    "Air_Pollution_2023",
    "Education",
    "Taxation",
    "Internet Access",
]
ALL_NUMERIC = CORE_FEATURES + SPARSE_FEATURES


def _save_missing_values_plot(df_raw: pd.DataFrame) -> None:
    miss = df_raw.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=True)
    if miss.empty:
        return
    plt.figure(figsize=(10, max(4, len(miss) * 0.35)))
    miss.plot(kind="barh", color="steelblue")
    plt.xlabel("Missing count")
    plt.title("Missing values by column (before imputation)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values_analysis.png", dpi=300)
    plt.close()


def _save_outlier_boxplots(df_imp: pd.DataFrame) -> None:
    n = len(ALL_NUMERIC)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, col in enumerate(ALL_NUMERIC):
        ax = axes[i]
        ax.boxplot(df_imp[col].dropna(), vert=True)
        short = col.replace(" ", "\n")[:22]
        ax.set_title(short, fontsize=9)
        ax.set_xticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature distributions (after imputation, before scaling)", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "outliers_boxplots.png", dpi=300)
    plt.close()


def _save_preprocessing_summary(df_merged: pd.DataFrame, final_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    miss = df_merged.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=True)
    if not miss.empty:
        miss.plot(kind="barh", ax=axes[0, 0], color="coral")
    axes[0, 0].set_title("Missing values (merged, before imputation)")
    axes[0, 0].set_xlabel("Count")

    sample_cols = [
        "Purchasing Power Index",
        "Safety Index",
        "Pollution Index",
        "Cost of Living Index",
    ]
    scaled_sample = final_df[[c for c in sample_cols if c in final_df.columns]]
    axes[0, 1].boxplot(
        [scaled_sample[c].values for c in scaled_sample.columns],
        tick_labels=[c.replace(" ", "\n")[:12] for c in scaled_sample.columns],
    )
    axes[0, 1].set_title("Sample scaled features (after pipeline)")
    axes[0, 1].axhline(0, color="red", linestyle="--", alpha=0.6)

    top = df_merged.nsmallest(10, "Rank")[["City", "Rank"]]
    axes[1, 0].barh(top["City"][::-1], top["Rank"][::-1], color="steelblue")
    axes[1, 0].set_title("Top 10 cities by rank (lower rank = better)")
    axes[1, 0].set_xlabel("Rank")

    summary = (
        f"Rows: {len(final_df)}\n"
        f"Columns: {final_df.shape[1]}\n"
        f"Numeric features (scaled): {len(ALL_NUMERIC)}\n"
        f"Missing after pipeline: {final_df.isnull().sum().sum()}"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.1,
        0.5,
        summary,
        fontsize=12,
        family="monospace",
        verticalalignment="center",
    )
    axes[1, 1].set_title("Summary")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "preprocessing_summary.png", dpi=300)
    plt.close()


def main() -> None:
    sns.set_style("whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INTERIM_DIR / "merged_livable_cities.csv")
    _save_missing_values_plot(df)
    df_processed = df.copy()

    for col in SPARSE_FEATURES:
        if df_processed[col].isnull().sum() > 0:
            country_mean = df_processed.groupby("Country")[col].transform("mean")
            df_processed[col] = df_processed[col].fillna(country_mean)
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(0)

    _save_outlier_boxplots(df_processed)

    le_country = LabelEncoder()
    df_processed["Country_Encoded"] = le_country.fit_transform(df_processed["Country"])

    scaler = StandardScaler()
    df_scaled = df_processed.copy()
    df_scaled[ALL_NUMERIC] = scaler.fit_transform(df_processed[ALL_NUMERIC])

    final_df = df_scaled[
        ["Rank", "City", "Country", "Country_Encoded"] + ALL_NUMERIC
    ].copy()
    final_df.to_csv(PROCESSED_DIR / "processed_livable_cities.csv", index=False)
    print("Wrote data/processed/processed_livable_cities.csv", final_df.shape)

    _save_preprocessing_summary(df, final_df)


if __name__ == "__main__":
    main()
