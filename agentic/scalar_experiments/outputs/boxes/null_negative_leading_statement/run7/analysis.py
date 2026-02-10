import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path


def main() -> None:
    data_path = Path("boxes.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df["social"] = (df["y"] != 1).astype(int)  # 1 = used social information
    df["majority_choice"] = (df["y"] == 2).astype(int)  # 1 = chose majority option

    print("N rows:", len(df))
    print(df.describe(include="all"))

    # Overall rates
    social_rate = df["social"].mean()
    majority_rate = df["majority_choice"].mean()
    print(f"\nOverall social-information use rate: {social_rate:.3f}")
    print(f"Overall majority-choice rate:        {majority_rate:.3f}")

    # Social-information use by culture and age
    social_by_culture = df.groupby("culture")["social"].mean()
    majority_by_culture = df.groupby("culture")["majority_choice"].mean()
    print("\nSocial-information use by culture:")
    print(social_by_culture)
    print("\nMajority-choice rate by culture:")
    print(majority_by_culture)

    # Age-binned summaries
    df["age_bin"] = pd.cut(df["age"], bins=[4, 6, 8, 10, 12, 14], right=True)
    social_by_age = df.groupby("age_bin")["social"].mean()
    majority_by_age = df.groupby("age_bin")["majority_choice"].mean()
    print("\nSocial-information use by age bin:")
    print(social_by_age)
    print("\nMajority-choice rate by age bin:")
    print(majority_by_age)

    # Logistic regressions: does age or culture predict social / majority use?
    # Treat culture as categorical.
    try:
        social_model = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False)
        print("\nLogit model: social-information use ~ age + culture")
        print(social_model.summary())
    except Exception as e:
        print("\nFailed to fit social-information model:", e)
        social_model = None

    try:
        majority_model = smf.logit("majority_choice ~ age + C(culture)", data=df).fit(disp=False)
        print("\nLogit model: majority-choice ~ age + culture")
        print(majority_model.summary())
    except Exception as e:
        print("\nFailed to fit majority-choice model:", e)
        majority_model = None

    # Heuristic: quantify variation across cultures and age bins
    culture_sd_social = social_by_culture.std()
    culture_sd_majority = majority_by_culture.std()
    age_sd_social = social_by_age.std()
    age_sd_majority = majority_by_age.std()
    print("\nStd dev of rates across cultures / age bins:")
    print(" culture SD (social):   ", culture_sd_social)
    print(" culture SD (majority): ", culture_sd_majority)
    print(" age SD (social):       ", age_sd_social)
    print(" age SD (majority):     ", age_sd_majority)

    # Map these variation magnitudes plus model significance into a scalar.
    scalar = compute_scalar(
        culture_sd_social,
        culture_sd_majority,
        age_sd_social,
        age_sd_majority,
        social_model,
        majority_model,
    )

    print(f"\nChosen scalar answer (Likert -100..100): {scalar}")

    # Write conclusion.txt with ONLY the scalar value
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(int(round(scalar))), encoding="utf-8")


def compute_scalar(
    culture_sd_social: float,
    culture_sd_majority: float,
    age_sd_social: float,
    age_sd_majority: float,
    social_model,
    majority_model,
) -> int:
    """
    Convert observed variation and model significance into a -100..100 scalar.

    Positive values mean: clear evidence that reliance on social information
    and/or majority preference DO vary across cultures and developmental stages.
    Negative values mean: data are broadly consistent with little or no variation.
    """

    # Start from neutral (0 = agnostic).
    score = 0.0

    # Variation magnitudes: small SDs suggest limited variation.
    # Thresholds are heuristic but give us a smooth mapping.
    sd_values = [
        culture_sd_social,
        culture_sd_majority,
        age_sd_social,
        age_sd_majority,
    ]

    avg_sd = sum(sd_values) / len(sd_values)
    print(f"\nAverage SD across rates: {avg_sd:.3f}")

    # If variation is tiny, push towards "No" (negative score).
    if avg_sd < 0.03:
        score -= 60
    elif avg_sd < 0.06:
        score -= 30
    elif avg_sd < 0.10:
        score -= 10
    elif avg_sd < 0.15:
        score += 10
    else:
        score += 30

    # Incorporate model-based evidence (p-values).
    for model in (social_model, majority_model):
        if model is None:
            continue
        # Drop the intercept
        pvals = model.pvalues.drop("Intercept", errors="ignore")
        # Focus on predictors (age and culture dummies).
        if not pvals.empty:
            significant = (pvals < 0.05).sum()
            mildly_sig = ((pvals >= 0.05) & (pvals < 0.10)).sum()

            # Strong significance of multiple predictors -> push positive.
            score += significant * 10
            score += mildly_sig * 5

    # Clip to [-100, 100]
    if score > 100:
        score = 100
    if score < -100:
        score = -100

    return int(round(score))


if __name__ == "__main__":
    main()

