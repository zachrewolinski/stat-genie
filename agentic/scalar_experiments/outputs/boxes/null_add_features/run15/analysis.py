import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(path: str = "boxes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected columns are present
    expected = {
        "y",
        "gender",
        "age",
        "majority_first",
        "culture",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Social vs asocial: 1 = used social info (majority or minority), 0 = undemonstrated option
    df["social"] = np.where(df["y"].isin([2, 3]), 1, 0)
    # Majority preference among social choices
    df["majority_choice"] = np.where(df["y"] == 2, 1, 0)
    # Age groups (tertiles) as a rough proxy for developmental stages in this sample
    df["age_group"] = pd.qcut(df["age"], q=3, labels=["young", "middle", "old"])
    # Treat culture as categorical index
    df["culture_cat"] = df["culture"].astype("category")
    return df


def summarize_basic(df: pd.DataFrame) -> None:
    print("N observations:", len(df))
    print("\nOutcome distribution (y):")
    print(df["y"].value_counts(normalize=True).sort_index())

    print("\nSocial information use (any social vs undemonstrated):")
    social_rate = df["social"].mean()
    print(f"Mean social use: {social_rate:.3f}")

    social_df = df[df["social"] == 1]
    if len(social_df) > 0:
        maj_rate = social_df["majority_choice"].mean()
        print(f"Majority among social choices: {maj_rate:.3f}")

    print("\nSocial use by culture:")
    print(df.groupby("culture")["social"].mean())

    print("\nMajority choice among social choices by culture:")
    print(
        social_df.groupby("culture")["majority_choice"].mean()
        if len(social_df) > 0
        else "No social choices in data"
    )

    print("\nSocial use by age group:")
    print(df.groupby("age_group")["social"].mean())

    print("\nMajority choice among social choices by age group:")
    print(
        social_df.groupby("age_group")["majority_choice"].mean()
        if len(social_df) > 0
        else "No social choices in data"
    )


def fit_models(df: pd.DataFrame) -> dict:
    """Fit simple logistic models for social use and majority preference."""
    results = {}

    # Model 1: social ~ age + C(culture_cat)
    try:
        m1 = smf.glm(
            "social ~ age + C(culture_cat)",
            data=df,
            family=sm.families.Binomial(),
        ).fit()
        results["social_model"] = m1
        print("\n=== Logistic model: social ~ age + culture ===")
        print(m1.summary().as_text())
    except Exception as e:
        print("Failed to fit social model:", e)

    # Model 2: majority_choice ~ age + C(culture_cat) among social choices
    social_df = df[df["social"] == 1]
    if len(social_df) > 0 and social_df["majority_choice"].nunique() > 1:
        try:
            m2 = smf.glm(
                "majority_choice ~ age + C(culture_cat)",
                data=social_df,
                family=sm.families.Binomial(),
            ).fit()
            results["majority_model"] = m2
            print("\n=== Logistic model: majority_choice ~ age + culture (social only) ===")
            print(m2.summary().as_text())
        except Exception as e:
            print("Failed to fit majority model:", e)
    else:
        print(
            "Not enough variation in majority_choice among social trials to fit majority model."
        )

    return results


def infer_scalar(df: pd.DataFrame, models: dict) -> int:
    """
    Map evidence about variation across cultures and age to a Likert scalar in [-100, 100].
    Positive values indicate evidence that reliance on social information
    and preference for majority cues DO vary across cultures and developmental stages.
    """
    # Baseline: moderate evidence
    score = 40

    # Magnitude of between-culture differences in social use
    culture_social = df.groupby("culture")["social"].mean()
    if len(culture_social) > 1:
        spread_social = culture_social.max() - culture_social.min()
    else:
        spread_social = 0.0

    # Magnitude of between-culture differences in majority preference
    social_df = df[df["social"] == 1]
    if len(social_df) > 0 and social_df["majority_choice"].nunique() > 0:
        culture_majority = social_df.groupby("culture")["majority_choice"].mean()
        if len(culture_majority) > 1:
            spread_majority = culture_majority.max() - culture_majority.min()
        else:
            spread_majority = 0.0
    else:
        spread_majority = 0.0

    # Age group differences
    age_social = df.groupby("age_group")["social"].mean()
    spread_age_social = age_social.max() - age_social.min()

    if len(social_df) > 0:
        age_majority = social_df.groupby("age_group")["majority_choice"].mean()
        spread_age_majority = age_majority.max() - age_majority.min()
    else:
        spread_age_majority = 0.0

    # Use spreads to gauge heterogeneity (0-1 scale)
    heterogeneity = np.mean(
        [
            spread_social,
            spread_majority,
            spread_age_social,
            spread_age_majority,
        ]
    )

    # Pull in p-values from models if available
    pvals = []
    social_model = models.get("social_model")
    if social_model is not None:
        pvals.extend(
            [
                v
                for k, v in social_model.pvalues.items()
                if k.startswith("C(culture_cat)[") or k == "age"
            ]
        )
    majority_model = models.get("majority_model")
    if majority_model is not None:
        pvals.extend(
            [
                v
                for k, v in majority_model.pvalues.items()
                if k.startswith("C(culture_cat)[") or k == "age"
            ]
        )

    if pvals:
        sig_frac = np.mean([p < 0.05 for p in pvals])
    else:
        sig_frac = 0.0

    # Combine heterogeneity and significance fraction into a 0-1 evidence score
    # Heterogeneity is capped at 0.5 to avoid extreme influence
    heterogeneity_score = min(heterogeneity / 0.5, 1.0)
    evidence_score = 0.6 * heterogeneity_score + 0.4 * sig_frac

    # Map evidence_score in [0,1] to [0, 90]
    score = int(round(evidence_score * 90))

    # Ensure strictly non-negative since the question is about presence of variation
    score = max(0, score)

    # Avoid extremely strong claims unless evidence_score is very high
    if evidence_score > 0.85:
        score = min(score, 95)

    # Bound within [-100, 100]
    score = max(-100, min(100, score))
    return score


def main() -> None:
    df = load_data()
    df = prepare_variables(df)
    summarize_basic(df)
    models = fit_models(df)
    scalar = infer_scalar(df, models)
    print("\nInferred Likert scalar (variation present):", scalar)

    # Write scalar to conclusion.txt as required
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()

