import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms


def main() -> None:
    base_path = Path(__file__).parent
    df = pd.read_csv(base_path / "affairs.csv")

    # According to metadata, the column names are slightly misaligned with their meanings.
    # age        -> coded frequency of extramarital intercourse in past year (0,1,2,3,7,12)
    # religiousness -> yes/no, actually indicates whether there are children in the marriage
    # occupation -> coded age in years (17.5,...,57)
    # children   -> coded years married
    # rating     -> 1-5 religiousness score
    # yearsmarried -> 9-20 education code
    # rownames   -> 1-7 occupation score
    # affairs    -> 1-5 self-rated marital happiness

    # Dependent variable: any extramarital affair (binary)
    df["affair_freq"] = df["age"]
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    # Key predictor: having children (1 = yes, 0 = no)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Additional covariates based on metadata
    df["age_years"] = df["occupation"]
    df["years_married"] = df["children"]
    df["religiousness_score"] = df["rating"]
    df["education_code"] = df["yearsmarried"]
    df["occupation_score"] = df["rownames"]
    df["marriage_rating"] = df["affairs"]

    model_cols = [
        "any_affair",
        "has_children",
        "age_years",
        "years_married",
        "religiousness_score",
        "education_code",
        "occupation_score",
        "marriage_rating",
        "gender",
    ]
    df_model = df[model_cols].dropna().copy()

    # Basic descriptive comparison of affair prevalence with vs without children
    group_stats = df_model.groupby("has_children")["any_affair"].agg(["mean", "count", "sum"])

    # Unadjusted test of difference in proportions (any affair vs none)
    successes = group_stats["sum"].values
    nobs = group_stats["count"].values
    try:
        z_stat, pval_prop = sms.proportions_ztest(successes, nobs)
    except Exception:
        z_stat, pval_prop = np.nan, np.nan

    # Logistic regression adjusting for other observed covariates
    try:
        formula = (
            "any_affair ~ has_children + age_years + years_married + "
            "religiousness_score + education_code + occupation_score + "
            "marriage_rating + C(gender)"
        )
        logit_model = smf.logit(formula, data=df_model).fit(disp=False)
        coef = float(logit_model.params["has_children"])
        pval = float(logit_model.pvalues["has_children"])
        or_val = float(np.exp(coef))
        conf_int = logit_model.conf_int().loc["has_children"]
        or_ci_low, or_ci_high = float(np.exp(conf_int[0])), float(np.exp(conf_int[1]))
    except Exception:
        # Fall back to unadjusted difference only if model fails
        logit_model = None
        pval = pval_prop
        coef = np.nan
        or_val = np.nan
        or_ci_low = np.nan
        or_ci_high = np.nan

    # Determine direction and strength of evidence that having children decreases affairs
    # Negative association (OR<1) with statistically significant p-value supports "Yes".
    response: int
    direction = ""

    if np.isfinite(pval) and pval < 0.05 and np.isfinite(or_val):
        if or_val < 1.0:
            direction = "decrease"
            # Map strength based on effect size and p-value
            if pval < 0.001 and or_val < 0.7:
                response = 85
            elif or_val < 0.85:
                response = 75
            else:
                response = 65
        else:
            direction = "increase"
            # Association in opposite direction of the research hypothesis
            if pval < 0.001 and or_val > 1.4:
                response = 15
            elif or_val > 1.2:
                response = 25
            else:
                response = 35
    else:
        # No clear statistical evidence either way; lean slightly based on estimated direction.
        direction = "none"
        if np.isfinite(or_val):
            if or_val < 1.0:
                # Point estimate suggests a decrease but is not statistically convincing.
                response = 45
            elif or_val > 1.0:
                # Point estimate suggests an increase but is not statistically convincing.
                response = 35
            else:
                response = 50
        else:
            response = 50

    # Build human-readable explanation
    pct_no_children = float(group_stats.loc[0, "mean"] * 100)
    pct_children = float(group_stats.loc[1, "mean"] * 100)
    n_no_children = int(group_stats.loc[0, "count"])
    n_children = int(group_stats.loc[1, "count"])

    explanation_parts = []
    explanation_parts.append(
        "Using 601 married respondents, I coded engagement in extramarital affairs as a "
        "binary outcome indicating whether the reported frequency of extramarital intercourse "
        "in the past year was greater than zero."
    )
    explanation_parts.append(
        "I treated the yes/no children indicator (labeled 'religiousness' in the file but "
        "described in the metadata as 'Are there children in the marriage?') as the main predictor "
        "of interest."
    )
    explanation_parts.append(
        f"Descriptively, among those without children (n={n_no_children}), about "
        f"{pct_no_children:.1f}% reported at least one extramarital affair, compared with "
        f"{pct_children:.1f}% among those with children (n={n_children})."
    )

    if np.isfinite(pval_prop):
        explanation_parts.append(
            "A two-sample test for equality of proportions comparing these two groups "
            f"yields a p-value of approximately {pval_prop:.3f}."
        )

    if logit_model is not None and np.isfinite(or_val):
        explanation_parts.append(
            "I then fit a logistic regression for having any extramarital affair with the children "
            "indicator as the key predictor, adjusting for age, years married, religiousness score, "
            "education, occupation score, marital happiness rating, and gender."
        )
        explanation_parts.append(
            f"In this adjusted model, the odds ratio for having children is about {or_val:.2f} "
            f"with a 95% confidence interval from {or_ci_low:.2f} to {or_ci_high:.2f} and "
            f"a p-value of approximately {pval:.3f}."
        )

    if direction == "decrease":
        explanation_parts.append(
            "Because the estimated odds ratio is below 1 and statistically significant, "
            "there is evidence that having children is associated with a lower likelihood "
            "of engaging in extramarital affairs in this sample."
        )
    elif direction == "increase":
        explanation_parts.append(
            "Because the estimated odds ratio is above 1 and statistically significant, "
            "the data instead suggest that having children is associated with a higher "
            "likelihood of engaging in extramarital affairs in this sample, contrary to "
            "the original hypothesis."
        )
    else:
        explanation_parts.append(
            "However, the statistical tests do not show a clear, statistically significant "
            "association between having children and engaging in extramarital affairs, so the "
            "data do not provide strong evidence for either an increase or decrease."
        )

    explanation_parts.append(
        "These conclusions are based on observational survey data from a single time point, "
        "so they describe associations rather than definitive causal effects."
    )

    explanation = " ".join(explanation_parts)

    result = {
        "response": int(response),
        "explanation": explanation,
    }

    with open(base_path / "conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
