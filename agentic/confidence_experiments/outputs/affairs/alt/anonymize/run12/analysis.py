import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    # Load metadata and data
    info_path = Path("info.json")
    data_path = Path("affairs.csv")

    with info_path.open("r") as f:
        info = json.load(f)

    df = pd.read_csv(data_path)

    # Rename columns for clarity
    df = df.rename(
        columns={
            "feature1": "id_like",
            "feature2": "affair_freq",
            "feature3": "gender",
            "feature4": "age",
            "feature5": "years_married",
            "feature6": "children",
            "feature7": "religiousness",
            "feature8": "education",
            "feature9": "occupation",
            "feature10": "marriage_rating",
        }
    )

    # Derived variables
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)
    df["has_children"] = (df["children"].str.lower() == "yes").astype(int)

    # Basic descriptives
    group_desc = (
        df.groupby("has_children")["affair_freq"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )

    # Proportion with any affair by children status
    prop_any = (
        df.groupby("has_children")["any_affair"]
        .mean()
        .reset_index()
        .rename(columns={"any_affair": "prop_any_affair"})
    )

    # Two-sample t-test for mean affair frequency
    affairs_children = df.loc[df["has_children"] == 1, "affair_freq"]
    affairs_no_children = df.loc[df["has_children"] == 0, "affair_freq"]
    t_stat, t_pval = stats.ttest_ind(
        affairs_children,
        affairs_no_children,
        equal_var=False,
        nan_policy="omit",
    )

    # Difference in proportions (chi-square test) for any_affair
    contingency = pd.crosstab(df["has_children"], df["any_affair"])
    chi2, chi_pval, _, _ = stats.chi2_contingency(contingency)

    # Logistic regression: any_affair ~ has_children + controls
    # Use a simple model to avoid overfitting on small sample.
    formula = (
        "any_affair ~ has_children + C(gender) + age + years_married "
        "+ religiousness + education + occupation + marriage_rating"
    )

    try:
        logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
        has_children_coef = logit_model.params.get("has_children", np.nan)
        has_children_pval = logit_model.pvalues.get("has_children", np.nan)
        logit_or = float(np.exp(has_children_coef)) if np.isfinite(has_children_coef) else np.nan
    except Exception:
        # Fallback if model fails for any reason
        logit_model = None
        has_children_coef = np.nan
        has_children_pval = np.nan
        logit_or = np.nan

    # Summarize key statistics for explanation
    # Map has_children 0/1 to human-readable labels
    mean_no_children = float(
        group_desc.loc[group_desc["has_children"] == 0, "mean"].iloc[0]
    )
    mean_children = float(
        group_desc.loc[group_desc["has_children"] == 1, "mean"].iloc[0]
    )
    prop_no_children = float(
        prop_any.loc[prop_any["has_children"] == 0, "prop_any_affair"].iloc[0]
    )
    prop_children = float(
        prop_any.loc[prop_any["has_children"] == 1, "prop_any_affair"].iloc[0]
    )

    # Decide on Yes/No and Likert score
    # We interpret "Yes" as: having children meaningfully decreases engagement in extramarital affairs.
    direction = None
    if np.isfinite(has_children_coef):
        direction = "decrease" if has_children_coef < 0 else "increase"

    # Default conservative assessment based primarily on adjusted logistic regression
    if np.isfinite(has_children_pval) and has_children_pval < 0.05 and direction == "decrease":
        # Statistically significant decrease after adjustment
        # Strength scaled by effect size (odds ratio)
        if logit_or <= 0.6:
            response_value = 90
        elif logit_or <= 0.8:
            response_value = 80
        else:
            response_value = 70
        qualitative = "Yes"
    elif np.isfinite(has_children_pval) and has_children_pval < 0.05 and direction == "increase":
        # Significant effect but opposite direction of the research hypothesis
        response_value = 5
        qualitative = "No"
    else:
        # No clear adjusted effect: treat as lack of strong evidence
        # Look at unadjusted comparisons to fine-tune how strongly we say "No".
        mean_diff = mean_no_children - mean_children
        prop_diff = prop_no_children - prop_children

        # If all signals are very close to zero, we are confident in "No".
        if abs(mean_diff) < 0.1 and abs(prop_diff) < 0.02:
            response_value = 15
        else:
            # Some suggestive but not statistically robust differences
            response_value = 30
        qualitative = "No"

    # Build explanation text
    rq = info["research_questions"][0] if info.get("research_questions") else ""

    explanation_lines = []
    explanation_lines.append(
        f"Research question: {rq.strip()} (interpreting higher values of affair_freq as more engagement)."
    )
    explanation_lines.append(
        "Children are coded as has_children=1 when respondents report that there are children in the marriage."
    )
    explanation_lines.append(
        f"Descriptively, the mean affair frequency is {mean_children:.2f} "
        f"for couples with children and {mean_no_children:.2f} for couples without children."
    )
    explanation_lines.append(
        f"The proportion of respondents reporting any extramarital affair in the past year "
        f"is {prop_children:.2%} with children versus {prop_no_children:.2%} without children."
    )
    explanation_lines.append(
        f"A Welch two-sample t-test comparing mean affair frequency between those with and without children "
        f"yields p-value {t_pval:.3f}, and a chi-square test of independence on the any-affair indicator "
        f"yields p-value {chi_pval:.3f}."
    )

    if logit_model is not None and np.isfinite(has_children_coef) and np.isfinite(has_children_pval):
        explanation_lines.append(
            "In a logistic regression of any_affair on has_children, controlling for gender, age, years married, "
            "religiousness, education, occupation, and self-rated marriage quality, "
            f"the coefficient on has_children is {has_children_coef:.3f}, corresponding to an odds ratio of "
            f"{logit_or:.2f} with p-value {has_children_pval:.3f}."
        )
    else:
        explanation_lines.append(
            "A multivariable logistic regression adjusting for demographic and marital covariates could not be "
            "reliably fit, so the assessment relies primarily on unadjusted comparisons."
        )

    if qualitative == "Yes":
        explanation_lines.append(
            "Taken together, these results provide statistically significant evidence that the presence of children "
            "is associated with reduced engagement in extramarital affairs in this sample, "
            "so I answer 'Yes' to the research question."
        )
    else:
        explanation_lines.append(
            "Overall, the differences between respondents with and without children are not consistently "
            "statistically significant after adjustment for other factors, and the observed descriptive "
            "differences are modest. Thus, there is insufficient evidence that having children materially "
            "decreases engagement in extramarital affairs in this dataset, so I answer 'No' to the research question."
        )

    explanation = " ".join(explanation_lines)

    result = {
        "response": int(response_value),
        "explanation": explanation,
    }

    with Path("conclusion.txt").open("w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()

