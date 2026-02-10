import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def evidence_from_p(p: float) -> float:
    """
    Convert a p-value into an evidence score in [0, 1],
    where 0 means no evidence for variation and 1 means
    extremely strong evidence.
    """
    if np.isnan(p):
        return 0.0
    # Cap p at 1.0 for safety
    p = float(max(min(p, 1.0), 0.0))
    # Linear transform: p <= 0 gives 1, p >= 0.1 gives 0
    if p >= 0.1:
        return 0.0
    return 1.0 - p / 0.1


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Define key outcome variables
    df["is_social"] = (df["y"] != 1).astype(int)
    social_df = df[df["y"].isin([2, 3])].copy()
    social_df["is_majority"] = (social_df["y"] == 2).astype(int)

    # Fit logistic models to assess variation across age and culture
    # Model 1: reliance on social information (any demonstrated option vs undemonstrated)
    social_model = smf.logit("is_social ~ age + C(culture)", data=df).fit(disp=False)

    # Model 2: majority preference among social learners (majority vs minority choice)
    majority_model = smf.logit("is_majority ~ age + C(culture)", data=social_df).fit(disp=False)

    # Extract p-values for age and culture terms
    social_pvalues = social_model.pvalues
    majority_pvalues = majority_model.pvalues

    age_social_p = float(social_pvalues.get("age", np.nan))
    age_majority_p = float(majority_pvalues.get("age", np.nan))

    culture_terms_social = [name for name in social_pvalues.index if name.startswith("C(culture)")]
    culture_terms_majority = [name for name in majority_pvalues.index if name.startswith("C(culture)")]

    culture_social_p = float(
        social_pvalues[culture_terms_social].min()
    ) if culture_terms_social else float("nan")
    culture_majority_p = float(
        majority_pvalues[culture_terms_majority].min()
    ) if culture_terms_majority else float("nan")

    # Convert p-values to evidence scores
    evidence_scores = [
        evidence_from_p(age_social_p),
        evidence_from_p(culture_social_p),
        evidence_from_p(age_majority_p),
        evidence_from_p(culture_majority_p),
    ]

    # Overall evidence that reliance on social information
    # and majority preference vary across age and cultures
    overall_evidence = float(np.mean(evidence_scores))

    # Map evidence in [0, 1] to Likert scalar in [-100, 100]
    # 0.0 -> -100 (strong "No"), 0.5 -> 0 (neutral), 1.0 -> 100 (strong "Yes")
    scalar = int(round((overall_evidence * 2.0 - 1.0) * 100.0))

    # Print a short summary for interactive inspection
    print("Logistic model p-values:")
    print(f"  age (social)     p = {age_social_p:.3g}")
    print(f"  culture (social) p = {culture_social_p:.3g}")
    print(f"  age (majority)   p = {age_majority_p:.3g}")
    print(f"  culture (majority) p = {culture_majority_p:.3g}")
    print("Evidence scores (0-1):", evidence_scores)
    print("Overall evidence:", overall_evidence)
    print("Derived Likert scalar (-100 to 100):", scalar)

    # Write scalar conclusion to file as required
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

