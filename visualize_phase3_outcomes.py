import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parse_params_column(df: pd.DataFrame) -> pd.DataFrame:
    parsed_rows = []
    for _, row in df.iterrows():
        params = row.get("params", "{}")
        parsed = ast.literal_eval(params) if isinstance(params, str) else {}
        parsed_rows.append(parsed)

    params_df = pd.DataFrame(parsed_rows)
    return pd.concat([df.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)


def _save_cv_distribution_by_model(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    ax = sns.boxplot(
        data=df,
        x="model",
        y="mean_test_score",
        hue="model",
        palette="Set2",
        dodge=False,
    )
    sns.stripplot(
        data=df,
        x="model",
        y="mean_test_score",
        color="black",
        alpha=0.35,
        size=3,
        jitter=0.25,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.title("CV R2 Distribution by Model")
    plt.xlabel("Model")
    plt.ylabel("Cross-Validated R2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _save_top_configs(df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    top_df = df.sort_values("mean_test_score", ascending=False).head(top_n).copy()
    top_df = top_df.iloc[::-1]
    top_df["label"] = (
        "#"
        + top_df["rank_test_score"].astype(int).astype(str)
        + " "
        + top_df["model"].astype(str)
    )

    plt.figure(figsize=(11, 7.5))
    sns.barplot(
        data=top_df,
        x="mean_test_score",
        y="label",
        hue="model",
        dodge=False,
        palette="viridis",
    )
    plt.title(f"Top {top_n} Hyperparameter Configurations by CV R2")
    plt.xlabel("Cross-Validated R2")
    plt.ylabel("Configuration")
    plt.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _save_gb_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    gb_df = df[df["model"] == "GradientBoosting"].copy()
    if gb_df.empty:
        return

    grouped = (
        gb_df.groupby(["learning_rate", "n_estimators"], as_index=False)["mean_test_score"]
        .max()
        .pivot(index="learning_rate", columns="n_estimators", values="mean_test_score")
        .sort_index(ascending=True)
        .sort_index(axis=1)
    )

    plt.figure(figsize=(8, 5.5))
    sns.heatmap(grouped, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={"label": "CV R2"})
    plt.title("GradientBoosting: Best CV R2 by Learning Rate and Trees")
    plt.xlabel("n_estimators")
    plt.ylabel("learning_rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")

    csv_path = Path("phase3_all_hyperparameter_outcomes.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    outcomes = pd.read_csv(csv_path)
    enriched = _parse_params_column(outcomes)

    _save_cv_distribution_by_model(enriched, Path("phase3_cv_r2_distribution_by_model.png"))
    _save_top_configs(enriched, Path("phase3_top20_hyperparameter_configs.png"), top_n=20)
    _save_gb_heatmap(enriched, Path("phase3_gb_heatmap_lr_vs_estimators.png"))

    print("Generated visualizations:")
    print("- phase3_cv_r2_distribution_by_model.png")
    print("- phase3_top20_hyperparameter_configs.png")
    print("- phase3_gb_heatmap_lr_vs_estimators.png")


if __name__ == "__main__":
    main()
