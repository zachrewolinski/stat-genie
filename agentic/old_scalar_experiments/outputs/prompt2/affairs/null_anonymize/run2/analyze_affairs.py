import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Define key variables based on metadata in info.json
    df["has_affair"] = (df["feature2"] > 0).astype(int)
    df["children"] = df["feature6"].str.lower().map({"yes": 1, "no": 0})

    # Basic sanity checks
    if df["children"].isna().any():
        raise ValueError("Unexpected values in feature6 when mapping children yes/no.")

    # Descriptive statistics
    grouped = df.groupby("children")
    mean_freq = grouped["feature2"].mean()
    prop_any = grouped["has_affair"].mean()
    counts = grouped.size()

    # Children flag: 1 = has children, 0 = no children
    mean_freq_children = mean_freq.get(1, np.nan)
    mean_freq_no_children = mean_freq.get(0, np.nan)
    prop_any_children = prop_any.get(1, np.nan)
    prop_any_no_children = prop_any.get(0, np.nan)
    n_children = int(counts.get(1, 0))
    n_no_children = int(counts.get(0, 0))

    # T-tests (Welch) for mean frequency and affair incidence
    freq_children = df.loc[df["children"] == 1, "feature2"]
    freq_no_children = df.loc[df["children"] == 0, "feature2"]
    t_freq, p_freq = stats.ttest_ind(freq_children, freq_no_children, equal_var=False)

    any_children = df.loc[df["children"] == 1, "has_affair"]
    any_no_children = df.loc[df["children"] == 0, "has_affair"]
    t_any, p_any = stats.ttest_ind(any_children, any_no_children, equal_var=False)

    # Logistic regression for having any affair, controlling for other covariates
    df_model = df.copy()
    df_model["children"] = df_model["children"].astype(int)
    # Encode gender as numeric (1 = male, 0 = female) for convenience
    df_model["male"] = (df_model["feature3"].str.lower() == "male").astype(int)

    # Use a formula that includes key controls based on the metadata description
    formula = (
        "has_affair ~ children + male + feature4 + feature5 + "
        "feature7 + feature8 + feature9 + feature10"
    )
    logit_model = logit(formula, data=df_model).fit(disp=False)
    coef_children = float(logit_model.params["children"])
    pval_children = float(logit_model.pvalues["children"])

    # Aggregate evidence about direction of effect
    evidence_decrease = 0
    evidence_increase = 0

    # Mean frequency comparison
    if np.isfinite(mean_freq_children) and np.isfinite(mean_freq_no_children):
        if mean_freq_children < mean_freq_no_children:
            evidence_decrease += 1
        elif mean_freq_children > mean_freq_no_children:
            evidence_increase += 1

    # Proportion with any affair
    if np.isfinite(prop_any_children) and np.isfinite(prop_any_no_children):
        if prop_any_children < prop_any_no_children:
            evidence_decrease += 1
        elif prop_any_children > prop_any_no_children:
            evidence_increase += 1

    # Logistic regression coefficient
    if coef_children < 0:
        evidence_decrease += 2  # weight the multivariate result more
    elif coef_children > 0:
        evidence_increase += 2

    # Decide on answer based on direction and statistical strength
    if evidence_decrease > evidence_increase:
        response = "Yes"
        # Confidence based on how strong and significant the effect is
        if pval_children < 0.01 and p_freq < 0.01:
            confidence = 90
        elif pval_children < 0.05 or p_freq < 0.05 or p_any < 0.05:
            confidence = 75
        else:
            confidence = 60
    elif evidence_increase > evidence_decrease:
        response = "No"
        if pval_children < 0.01 and p_freq < 0.01:
            confidence = 90
        elif pval_children < 0.05 or p_freq < 0.05 or p_any < 0.05:
            confidence = 75
        else:
            confidence = 60
    else:
        # Ambiguous or no clear direction
        response = "No"
        confidence = 55

    # Build explanation text
    explanation_lines = []
    explanation_lines.append(
        "I used the Psychology Today extramarital affairs dataset (601 first-marriage respondents) "
        "to assess whether having children is associated with lower engagement in extramarital affairs."
    )
    explanation_lines.append(
        f"For the raw affair frequency scale (feature2, higher values = more extramarital intercourse), "
        f"the mean for respondents with children (n={n_children}) was {mean_freq_children:.3f}, "
        f"compared with {mean_freq_no_children:.3f} for those without children (n={n_no_children}). "
        f"A Welch t-test for this difference yielded t={t_freq:.3f}, p={p_freq:.4f}."
    )
    explanation_lines.append(
        f"Considering a binary outcome of having any extramarital affair in the past year (feature2 > 0), "
        f"the proportion with at least one affair was {prop_any_children:.3f} among respondents with children "
        f"and {prop_any_no_children:.3f} among those without children (Welch t-test on this 0/1 outcome: "
        f"t={t_any:.3f}, p={p_any:.4f})."
    )
    explanation_lines.append(
        "To adjust for potential confounders, I fit a logistic regression model predicting the binary affair indicator "
        "from a children indicator plus controls for gender, age, years married, religiousness, education, occupation, "
        "and self-rated quality of marriage."
    )
    explanation_lines.append(
        f"In this multivariate model, the coefficient for having children (1 = children present) was "
        f"{coef_children:.3f} with p-value {pval_children:.4f}, indicating the direction and strength of the "
        "association after adjustment for these covariates."
    )
    if response == "Yes":
        explanation_lines.append(
            "Across these analyses, respondents with children show lower engagement in extramarital affairs "
            "than those without children, and the combined evidence points toward a decreasing effect of having children."
        )
    else:
        explanation_lines.append(
            "Across these analyses, having children does not robustly correspond to lower engagement in extramarital "
            "affairs; the observed differences and regression estimates do not support a clear decreasing effect."
        )
    explanation_lines.append(
        f"Given the observed effect sizes, sample sizes, and statistical significance levels, I set my confidence in "
        f"this '{response}' answer to {confidence} out of 100."
    )

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    # Write required JSON object to conclusion.txt
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

