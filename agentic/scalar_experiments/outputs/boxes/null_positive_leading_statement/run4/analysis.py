import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # Define majority choice (y == 2)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Basic descriptive statistics
    overall_rate = df["majority_choice"].mean()

    # Age groups to capture developmental stages
    df["age_group"] = pd.cut(
        df["age"],
        bins=[3, 6, 9, 11, 14],
        labels=["4-6", "7-9", "10-11", "12-14"],
        include_lowest=True,
        right=True,
    )

    # Majority-choice rates by culture and age group
    culture_rates = df.groupby("culture")["majority_choice"].mean()
    age_rates = df.groupby("age_group")["majority_choice"].mean()

    culture_range = float(culture_rates.max() - culture_rates.min())
    age_range = float(age_rates.max() - age_rates.min())

    # Logistic regression: majority_choice ~ age + culture
    # This tests whether age and culture predict majority-choice probability.
    model = smf.logit("majority_choice ~ age + C(culture)", data=df).fit(disp=False)
    pvals = model.pvalues

    age_p = float(pvals.get("age", np.nan))
    culture_term_pvals = [
        float(pvals[name])
        for name in pvals.index
        if name.startswith("C(culture)")
    ]

    # Evidence scoring for variation across cultures and development
    score = 0.0

    # Descriptive variation by culture
    if culture_range > 0.30:
        score += 35
    elif culture_range > 0.20:
        score += 30
    elif culture_range > 0.10:
        score += 20
    elif culture_range > 0.05:
        score += 10

    # Descriptive variation by age group
    if age_range > 0.30:
        score += 35
    elif age_range > 0.20:
        score += 30
    elif age_range > 0.10:
        score += 20
    elif age_range > 0.05:
        score += 10

    # Inferential evidence: significance of predictors
    if not np.isnan(age_p) and age_p < 0.05:
        score += 20
    elif not np.isnan(age_p) and age_p < 0.10:
        score += 10

    if culture_term_pvals:
        num_sig_culture = sum(p < 0.05 for p in culture_term_pvals)
        num_trend_culture = sum((p >= 0.05) and (p < 0.10) for p in culture_term_pvals)

        if num_sig_culture >= 2:
            score += 25
        elif num_sig_culture == 1:
            score += 20

        if num_sig_culture == 0 and num_trend_culture > 0:
            score += 10

    # Anchor on the overall tendency to follow the majority
    # Strong overall majority-following combined with variation implies a strong "Yes".
    if overall_rate > 0.60:
        score += 10
    elif overall_rate > 0.50:
        score += 5

    # Bound score to Likert scale [-100, 100] and bias towards 0 if very weak
    score = max(0.0, min(score, 100.0))

    # If evidence is extremely weak, treat as neutral
    if score < 5:
        score = 0.0

    scalar = int(round(score))

    # Write scalar to conclusion.txt as required
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

