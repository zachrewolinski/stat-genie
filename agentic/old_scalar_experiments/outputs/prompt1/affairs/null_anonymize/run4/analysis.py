import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DATA_FILE = Path("affairs.csv")
CONCLUSION_FILE = Path("conclusion.txt")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    # Ensure expected columns exist
    expected_cols = {
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        "feature5",
        "feature6",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Outcome variables:
    #  - affair_freq: numeric frequency of extramarital intercourse in past year.
    #  - has_affair: binary indicator of any extramarital intercourse.
    df["affair_freq"] = df["feature2"].astype(float)
    df["has_affair"] = (df["affair_freq"] > 0).astype(int)

    # Key predictor: children in marriage (yes/no).
    df["children"] = df["feature6"].astype("category")

    # Additional controls based on metadata descriptions.
    df["gender"] = df["feature3"].astype("category")
    df["age"] = df["feature4"].astype(float)
    df["years_married"] = df["feature5"].astype(float)
    df["religiousness"] = df["feature7"].astype(float)
    df["education"] = df["feature8"].astype(float)
    df["occupation"] = df["feature9"].astype(float)
    df["marriage_rating"] = df["feature10"].astype(float)

    # Drop rows with any missing values in variables used in the models.
    model_vars = [
        "has_affair",
        "affair_freq",
        "children",
        "gender",
        "age",
        "years_married",
        "religiousness",
        "education",
        "occupation",
        "marriage_rating",
    ]
    df = df.dropna(subset=model_vars)
    return df


def summarize_by_children(df: pd.DataFrame) -> dict:
    summary = {}
    grouped = df.groupby("children", observed=True)

    # Mean frequency and proportion with any affair.
    freq_means = grouped["affair_freq"].mean()
    affair_rates = grouped["has_affair"].mean()
    counts = grouped.size()

    for child_status in freq_means.index:
        summary[str(child_status)] = {
            "n": int(counts[child_status]),
            "mean_affair_frequency": float(freq_means[child_status]),
            "proportion_any_affair": float(affair_rates[child_status]),
        }

    # Simple difference measures: no children minus children.
    if set(freq_means.index) >= {"yes", "no"}:
        diff_freq = freq_means["no"] - freq_means["yes"]
        diff_prop = affair_rates["no"] - affair_rates["yes"]
    else:
        diff_freq = np.nan
        diff_prop = np.nan

    summary["differences_no_minus_yes"] = {
        "mean_affair_frequency_diff": float(diff_freq),
        "proportion_any_affair_diff": float(diff_prop),
    }
    return summary


def fit_logistic_model(df: pd.DataFrame):
    # Logistic regression for any affair vs children + covariates.
    # children is categorical (yes/no); other predictors numeric.
    formula = (
        "has_affair ~ C(children) + C(gender) + age + years_married + "
        "religiousness + education + occupation + marriage_rating"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    return model


def interpret_results(summary: dict, logit_model) -> tuple[str, str]:
    """
    Decide whether having children decreases engagement in extramarital affairs
    and generate an explanation describing evidence.
    """
    # Extract descriptive differences.
    diffs = summary.get("differences_no_minus_yes", {})
    freq_diff = diffs.get("mean_affair_frequency_diff", np.nan)
    prop_diff = diffs.get("proportion_any_affair_diff", np.nan)

    # From the logistic model, we care about the coefficient for children=yes
    # relative to children=no (baseline) and its p-value.
    params = logit_model.params
    pvalues = logit_model.pvalues

    # Depending on how statsmodels coded the factor, look for the term
    # corresponding to children[T.yes]; fall back to any term containing
    # 'children' if needed.
    child_term_candidates = [name for name in params.index if "children" in name]
    child_term = None
    if "C(children)[T.yes]" in params.index:
        child_term = "C(children)[T.yes]"
    elif child_term_candidates:
        child_term = child_term_candidates[0]

    coef = float(params[child_term]) if child_term is not None else np.nan
    pval = float(pvalues[child_term]) if child_term is not None else np.nan

    # Decision rule:
    #  - If descriptive statistics and regression both clearly suggest
    #    lower affair involvement among couples with children (negative
    #    differences and negative coefficient), we answer "Yes".
    #  - If the differences are small or mixed, or the regression effect
    #    is not statistically convincing (e.g., p > 0.05), we answer "No".
    #
    # This focuses on whether we see clear evidence of a *decrease*, not
    # merely any numerical difference.
    alpha = 0.05
    descriptive_supports_decrease = (freq_diff > 0) and (prop_diff > 0)
    regression_supports_decrease = (coef < 0) and (pval < alpha)

    if descriptive_supports_decrease and regression_supports_decrease:
        response = "Yes"
    else:
        response = "No"

    explanation_parts = []
    explanation_parts.append(
        "I examined couples with and without children using the 1969 Psychology Today "
        "survey data on 601 first-marriage respondents."
    )
    # Describe descriptive statistics.
    for child_status in ("yes", "no"):
        if child_status in summary:
            s = summary[child_status]
            explanation_parts.append(
                f"For marriages with children = '{child_status}', there were {s['n']} observations; "
                f"the mean coded frequency of extramarital intercourse was "
                f"{s['mean_affair_frequency']:.2f}, and the proportion engaging in any affair "
                f"in the past year was {s['proportion_any_affair']:.2f}."
            )
    if not np.isnan(freq_diff) and not np.isnan(prop_diff):
        explanation_parts.append(
            "Comparing groups, the mean affair-frequency code for couples without children "
            f"minus that for couples with children was {freq_diff:.2f}, and the difference "
            f"in the proportion having any affair (no children minus children) was "
            f"{prop_diff:.2f}."
        )

    # Describe regression result.
    if child_term is not None and not np.isnan(coef) and not np.isnan(pval):
        explanation_parts.append(
            "I also fit a logistic regression model predicting whether a respondent had any "
            "extramarital intercourse from the presence of children in the marriage and "
            "controls for gender, age, years married, religiousness, education, occupation, "
            "and self-rated marital happiness."
        )
        explanation_parts.append(
            f"In this model, the coefficient for having children (term '{child_term}') "
            f"was {coef:.3f} on the log-odds scale with p-value {pval:.3f}."
        )

    if response == "Yes":
        explanation_parts.append(
            "Both the descriptive comparisons and the regression indicate that having "
            "children is associated with a statistically significant decrease in the "
            "likelihood and frequency of extramarital affairs at the 5% significance level."
        )
    else:
        explanation_parts.append(
            "Although there may be small numerical differences between couples with and "
            "without children, the regression coefficient for children is not both strongly "
            "negative and statistically significant at the 5% level. I therefore conclude "
            "that this dataset does not provide clear evidence that having children "
            "decreases engagement in extramarital affairs."
        )

    explanation = " ".join(explanation_parts)
    return response, explanation


def main() -> None:
    df = load_data()
    df = prepare_variables(df)

    summary = summarize_by_children(df)
    logit_model = fit_logistic_model(df)

    response, explanation = interpret_results(summary, logit_model)

    result = {"response": response, "explanation": explanation}
    CONCLUSION_FILE.write_text(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

