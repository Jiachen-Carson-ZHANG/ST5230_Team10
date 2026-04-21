# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols


sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"
BASELINE_PERSONA = "baseline"


def find_results_csv() -> Path:
    candidates = [
        ROOT / "results.csv",
        ROOT / "financial-experiment" / "data" / "structured" / "results.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find results.csv. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def normalize_bool(series: pd.Series) -> pd.Series:
    mapping = {
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        "TRUE": 1,
        "FALSE": 0,
        1: 1,
        0: 0,
        "1": 1,
        "0": 0,
    }
    return series.map(mapping)


def save_current_figure(filename: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def prepare_dataframe() -> pd.DataFrame:
    results_path = find_results_csv()
    print(f"Loading analysis data from: {results_path}")
    df = pd.read_csv(results_path)

    df["risk_rating_score"] = pd.to_numeric(df["risk_rating_score"], errors="coerce")
    df["unsupported_financial_claim_flag"] = normalize_bool(
        df["unsupported_financial_claim_flag"]
    )
    if "compliance_refusal_flag" in df.columns:
        df["compliance_refusal_flag"] = normalize_bool(df["compliance_refusal_flag"])
    if "output_word_count" in df.columns:
        df["output_word_count"] = pd.to_numeric(df["output_word_count"], errors="coerce")

    return df


def run_risk_score_analysis(df: pd.DataFrame) -> None:
    risk_df = df.dropna(subset=["persona_id", "model", "risk_rating_score"]).copy()

    print("=" * 60)
    print("STATISTICAL TEST: Main Effect of Persona on Risk Score")
    print("=" * 60)

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x="persona_id",
        y="risk_rating_score",
        hue="model",
        data=risk_df,
        inner="quartile",
        palette="muted",
    )
    plt.title("Main Effect: Risk Rating Distribution by Persona and Model")
    plt.ylabel("Risk Rating Score (1-5)")
    plt.xlabel("Persona ID")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure("main_effect_risk_rating_distribution.png")

    personas = risk_df["persona_id"].dropna().unique()
    groups = [risk_df[risk_df["persona_id"] == p]["risk_rating_score"] for p in personas]

    if len(groups) > 1:
        stat, p_value = stats.kruskal(*groups)
        print(f"Kruskal-Wallis H-statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4e}")
        if p_value < 0.05:
            print("CONCLUSION: Persona has a significant main effect on risk score.")
        else:
            print("CONCLUSION: No significant main effect found.")


def run_action_analysis(df: pd.DataFrame) -> None:
    action_df = df.dropna(subset=["persona_id", "strategic_action"]).copy()

    print("=" * 60)
    print("STATISTICAL TEST: Main Effect of Persona on Strategic Action")
    print("=" * 60)

    action_crosstab_pct = (
        pd.crosstab(action_df["persona_id"], action_df["strategic_action"], normalize="index")
        * 100
    )

    action_crosstab_pct.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
        colormap="RdYlGn_r",
        edgecolor="white",
    )
    plt.title("Main Effect: Strategic Action Preference by Persona")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Persona ID")
    plt.xticks(rotation=0)
    plt.legend(title="Strategic Action", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure("main_effect_strategic_action_preference.png")

    action_crosstab_raw = pd.crosstab(action_df["persona_id"], action_df["strategic_action"])
    chi2, p_val, _, _ = chi2_contingency(action_crosstab_raw)

    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p_val:.4e}")
    if p_val < 0.05:
        print("CONCLUSION: Persona significantly alters the distribution of strategic actions.")


def run_interaction_analysis(df: pd.DataFrame) -> None:
    interaction_df = df.dropna(subset=["persona_id", "model", "risk_rating_score"]).copy()

    print("=" * 60)
    print("STATISTICAL TEST: Two-Way ANOVA (Persona x Model)")
    print("=" * 60)

    plt.figure(figsize=(10, 6))
    sns.pointplot(
        x="persona_id",
        y="risk_rating_score",
        hue="model",
        data=interaction_df,
        dodge=True,
        capsize=0.1,
        palette="dark",
    )
    plt.title("Interaction Effect: How Different Models React to Personas")
    plt.ylabel("Mean Risk Rating Score")
    plt.xlabel("Persona ID")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure("interaction_effect_persona_by_model.png")

    try:
        anova_model = ols(
            "risk_rating_score ~ C(persona_id) + C(model) + C(persona_id):C(model)",
            data=interaction_df,
        ).fit()
        anova_table = sm.stats.anova_lm(anova_model, typ=2)
        print(anova_table.to_string())

        interaction_p = anova_table.loc["C(persona_id):C(model)", "PR(>F)"]
        if interaction_p < 0.05:
            print("\nCONCLUSION: Significant interaction effect found.")
            print("The impact of persona depends on which model is being used.")
        else:
            print("\nCONCLUSION: No significant interaction effect found.")
    except Exception as exc:
        print(f"Error running ANOVA: {exc}")


def run_delta_analysis(df: pd.DataFrame) -> None:
    delta_base_df = df.dropna(subset=["persona_id", "model", "risk_rating_score"]).copy()

    print(f"Calculating persona-induced delta relative to '{BASELINE_PERSONA}'...")

    baseline_means = (
        delta_base_df[delta_base_df["persona_id"] == BASELINE_PERSONA]
        .groupby("model")["risk_rating_score"]
        .mean()
        .reset_index()
        .rename(columns={"risk_rating_score": "baseline_score"})
    )

    if baseline_means.empty:
        print(f"Error: baseline persona '{BASELINE_PERSONA}' not found.")
        return

    persona_means = (
        delta_base_df[delta_base_df["persona_id"] != BASELINE_PERSONA]
        .groupby(["model", "persona_id"])["risk_rating_score"]
        .mean()
        .reset_index()
    )

    delta_df = pd.merge(persona_means, baseline_means, on="model")
    delta_df["delta_score"] = abs(delta_df["risk_rating_score"] - delta_df["baseline_score"])

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="model",
        y="delta_score",
        hue="persona_id",
        data=delta_df,
        palette="magma",
    )
    plt.title(f"Persona-Induced Delta Relative to {BASELINE_PERSONA}")
    plt.ylabel("Absolute Shift in Mean Risk Score")
    plt.xlabel("Model")

    for patch in plt.gca().patches:
        if patch.get_height() > 0:
            plt.gca().annotate(
                f"{patch.get_height():.2f}",
                (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.legend(title="Persona Applied", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure("persona_induced_delta.png")


def run_flip_rate_analysis(df: pd.DataFrame) -> None:
    required = ["article_id", "model", "sample_idx", "persona_id", "strategic_action"]
    flip_df = df.dropna(subset=required).copy()

    print(f"Calculating action flip rate relative to '{BASELINE_PERSONA}'...")

    baseline_actions = flip_df[flip_df["persona_id"] == BASELINE_PERSONA][
        ["article_id", "model", "sample_idx", "strategic_action"]
    ].rename(columns={"strategic_action": "baseline_action"})

    if baseline_actions.empty:
        print("Error: baseline persona data missing for flip-rate calculation.")
        return

    merged = pd.merge(
        flip_df,
        baseline_actions,
        on=["article_id", "model", "sample_idx"],
        how="inner",
    )
    analysis_df = merged[merged["persona_id"] != BASELINE_PERSONA].copy()
    analysis_df["is_flipped"] = (
        analysis_df["strategic_action"] != analysis_df["baseline_action"]
    )

    flip_rates = (
        analysis_df.groupby(["persona_id", "model"])["is_flipped"].mean().reset_index()
    )
    flip_rates["is_flipped"] *= 100

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="model",
        y="is_flipped",
        hue="persona_id",
        data=flip_rates,
        palette="Set2",
    )
    plt.title(f"Action Flip Rate Relative to {BASELINE_PERSONA}")
    plt.ylabel("Flip Rate (%)")
    plt.xlabel("Model")

    for patch in plt.gca().patches:
        if patch.get_height() > 0:
            plt.gca().annotate(
                f"{patch.get_height():.1f}%",
                (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.legend(title="Persona Applied", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure("action_flip_rate.png")


def run_unsupported_claim_analysis(df: pd.DataFrame) -> None:
    claim_df = df.dropna(subset=["model", "persona_id", "unsupported_financial_claim_flag"]).copy()

    plt.figure(figsize=(10, 5))
    halluc_map = (
        claim_df.groupby(["model", "persona_id"])["unsupported_financial_claim_flag"]
        .mean()
        .unstack()
        * 100
    )
    sns.heatmap(halluc_map, annot=True, fmt=".1f", cmap="Reds")
    plt.title("Unsupported Claim Rate by Model and Persona (%)")
    save_current_figure("unsupported_claim_rate_heatmap.png")

    contingency = pd.crosstab(
        claim_df["persona_id"], claim_df["unsupported_financial_claim_flag"]
    )
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

    print("=" * 60)
    print("DETAILED STATISTICAL SUMMARY: PERSONA VS UNSUPPORTED CLAIM FLAG")
    print("-" * 60)
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"P-value: {p_value:.4e}")
    print(f"Cramer's V: {cramers_v:.4f}")

    residuals = (contingency - expected) / np.sqrt(expected)
    print("\nStandardized Residuals:")
    print(residuals)

    if p_value < 0.05:
        print("\nRESULT: Significant association found between persona and unsupported-claim rate.")
    else:
        print("\nRESULT: No significant association found.")


def run_attention_shift_analysis(df: pd.DataFrame) -> None:
    focus_df = df.dropna(subset=["persona_id", "analysis_primary_focus"]).copy()

    attention_data = (
        pd.crosstab(
            focus_df["persona_id"],
            focus_df["analysis_primary_focus"],
            normalize="index",
        )
        * 100
    )
    attention_data.plot(kind="bar", stacked=True, colormap="Set3", edgecolor="white")
    plt.title("Attention Shift: Analytical Focus by Persona")
    plt.ylabel("Percentage of Focus (%)")
    plt.xlabel("Persona ID")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=0)
    save_current_figure("attention_shift_by_persona.png")


def calculate_entropy(series: pd.Series) -> float:
    probs = series.value_counts(normalize=True)
    return float(-(probs * np.log2(probs + 1e-9)).sum())


def run_entropy_analysis(df: pd.DataFrame) -> None:
    entropy_df = df.dropna(subset=["model", "persona_id", "strategic_action"]).copy()
    entropy_res = (
        entropy_df.groupby(["model", "persona_id"])["strategic_action"]
        .apply(calculate_entropy)
        .reset_index(name="decision_entropy")
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="persona_id",
        y="decision_entropy",
        hue="model",
        data=entropy_res,
        palette="coolwarm",
    )
    plt.ylabel("Decision Entropy")
    plt.xlabel("Persona ID")
    plt.title("Decision Entropy by Persona")
    save_current_figure("decision_entropy_by_persona.png")


def run_word_count_analysis(df: pd.DataFrame) -> None:
    if "output_word_count" not in df.columns:
        print("Skipping word-count analysis: output_word_count column not found.")
        return

    word_df = df.dropna(subset=["output_word_count", "unsupported_financial_claim_flag"]).copy()
    if word_df.empty:
        print("Skipping word-count analysis: no complete rows available.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="unsupported_financial_claim_flag",
        y="output_word_count",
        data=word_df,
        color="#8a6dd3",
    )
    plt.title("Verbosity vs. Unsupported-Claim Flag")
    plt.xlabel("Unsupported Claim Flag (0 = No, 1 = Yes)")
    plt.ylabel("Output Word Count")
    save_current_figure("verbosity_vs_unsupported_claim.png")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = prepare_dataframe()

    run_risk_score_analysis(df)
    run_action_analysis(df)
    run_interaction_analysis(df)
    run_delta_analysis(df)
    run_flip_rate_analysis(df)
    run_unsupported_claim_analysis(df)
    run_attention_shift_analysis(df)
    run_entropy_analysis(df)
    run_word_count_analysis(df)

    print("=" * 60)
    print(f"Analysis complete. Figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
