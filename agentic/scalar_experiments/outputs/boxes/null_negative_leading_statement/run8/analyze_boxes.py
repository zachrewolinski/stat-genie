import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def lr_pvalue(full_model, reduced_model) -> float:
    """Likelihood-ratio test p-value comparing two nested models."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    return float(stats.chi2.sf(lr_stat, df_diff))


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")
    df = df.dropna(subset=["y", "age", "culture"])

    # Define key behavioral outcomes
    df["social"] = (df["y"] != 1).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)

    # Model 1: reliance on social information (any demonstrator vs undemonstrated)
    social_full = smf.logit("social ~ age + C(culture)", data=df).fit(disp=False, maxiter=200)
    social_no_age = smf.logit("social ~ C(culture)", data=df).fit(disp=False, maxiter=200)
    social_no_culture = smf.logit("social ~ age", data=df).fit(disp=False, maxiter=200)

    p_social_age = lr_pvalue(social_full, social_no_age)
    p_social_culture = lr_pvalue(social_full, social_no_culture)

    # Model 2: majority preference among children who use social information
    df_social = df[df["social"] == 1].copy()

    majority_full = smf.logit("majority_choice ~ age + C(culture)", data=df_social).fit(
        disp=False,
        maxiter=200,
    )
    majority_no_age = smf.logit("majority_choice ~ C(culture)", data=df_social).fit(
        disp=False,
        maxiter=200,
    )
    majority_no_culture = smf.logit("majority_choice ~ age", data=df_social).fit(
        disp=False,
        maxiter=200,
    )

    p_majority_age = lr_pvalue(majority_full, majority_no_age)
    p_majority_culture = lr_pvalue(majority_full, majority_no_culture)

    # Combine evidence: smaller p-values indicate stronger variation across
    # cultures/developmental stages in reliance on social info or majority bias.
    min_p = min(p_social_age, p_social_culture, p_majority_age, p_majority_culture)

    # Map evidence strength to Likert-scale scalar where:
    #   negative = evidence for "No, little variation"
    #   positive = evidence for "Yes, clear variation"
    if min_p < 1e-6:
        scalar = 95
    elif min_p < 1e-4:
        scalar = 85
    elif min_p < 1e-3:
        scalar = 75
    elif min_p < 1e-2:
        scalar = 60
    elif min_p < 5e-2:
        scalar = 40
    elif min_p < 0.1:
        scalar = 10
    else:
        # No meaningful evidence that age or culture matter
        scalar = -60

    # Write final scalar conclusion
    with open("conclusion.txt", "w") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()

