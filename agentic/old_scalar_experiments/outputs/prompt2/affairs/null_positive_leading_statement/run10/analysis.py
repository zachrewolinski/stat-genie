import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")
    df = df.copy()

    # Key derived variables
    df["has_children"] = (df["children"] == "yes").astype(int)
    df["male"] = (df["gender"] == "male").astype(int)
    df["affair_binary"] = (df["affairs"] > 0).astype(int)

    # Descriptive comparisons by children status
    group = df.groupby("has_children")
    mean_affairs = group["affairs"].mean()
    prop_any = group["affair_binary"].mean()
    n_by_group = group.size()

    mean_affairs_children = float(mean_affairs.get(1, np.nan))
    mean_affairs_no_children = float(mean_affairs.get(0, np.nan))
    prop_any_children = float(prop_any.get(1, np.nan))
    prop_any_no_children = float(prop_any.get(0, np.nan))

    diff_prop_any = prop_any_children - prop_any_no_children
    diff_mean_affairs = mean_affairs_children - mean_affairs_no_children

    # Logistic regression for any affair, adjusting for covariates
    X_cols = [
        "has_children",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "male",
    ]
    X = df[X_cols]
    X = sm.add_constant(X, has_constant="add")
    y = df["affair_binary"]

    logit_result = None
    child_coef = np.nan
    child_p = np.nan
    child_or = np.nan
    child_ci_low = np.nan
    child_ci_high = np.nan

    try:
        logit_model = sm.Logit(y, X)
        logit_result = logit_model.fit(disp=False)
        child_coef = float(logit_result.params["has_children"])
        child_p = float(logit_result.pvalues["has_children"])
        child_or = float(np.exp(child_coef))
        ci = logit_result.conf_int().loc["has_children"]
        child_ci_low = float(np.exp(ci[0]))
        child_ci_high = float(np.exp(ci[1]))
    except Exception:
        logit_result = None

    # Evidence summary
    evidence_direction_negative = (diff_prop_any < 0) and (diff_mean_affairs < 0)
    logistic_negative = (not np.isnan(child_coef)) and (child_coef < 0)
    logistic_significant = (not np.isnan(child_p)) and (child_p < 0.05)

    if evidence_direction_negative and logistic_negative and logistic_significant:
        response = "Yes"
    else:
        response = "No"

    # Confidence scoring (0–100)
    if response == "Yes":
        if logistic_significant and logistic_negative:
            if child_p < 0.001:
                confidence = 95
            elif child_p < 0.01:
                confidence = 90
            else:
                confidence = 85
        elif evidence_direction_negative and logistic_negative:
            confidence = 70
        else:
            confidence = 55
    else:
        if logistic_negative and not logistic_significant:
            confidence = 70
        elif (not logistic_negative) and logistic_significant:
            confidence = 85
        else:
            confidence = 65

    confidence = max(0, min(100, int(round(confidence))))

    # Build explanation text
    n = len(df)
    n_children = int(n_by_group.get(1, 0))
    n_no_children = int(n_by_group.get(0, 0))

    def fmt_pct(x: float) -> str:
        return f"{x * 100:.1f}%" if np.isfinite(x) else "NA"

    explanation_parts = []
    explanation_parts.append(
        f"I analyzed {n} married individuals from the Fair (1978) affairs dataset "
        "to assess whether having children is associated with lower engagement in extramarital affairs."
    )
    explanation_parts.append(
        "Descriptively, parents (has_children=1) had an average of "
        f"{mean_affairs_children:.2f} affair-equivalent episodes in the last year, "
        f"compared with {mean_affairs_no_children:.2f} among non-parents (has_children=0)."
    )
    explanation_parts.append(
        "The proportion with any extramarital activity was "
        f"{fmt_pct(prop_any_children)} for parents (n={n_children}) "
        f"versus {fmt_pct(prop_any_no_children)} for non-parents (n={n_no_children})."
    )

    if logit_result is not None and np.isfinite(child_or):
        explanation_parts.append(
            "Using a multivariable logistic regression for any extramarital affair, "
            "adjusting for age, years married, religiousness, education, occupation, "
            "self-rated marital satisfaction, and gender, the odds ratio for having children "
            f"was {child_or:.2f} (95% CI {child_ci_low:.2f}–{child_ci_high:.2f}, p = {child_p:.3g})."
        )
        if logistic_significant and logistic_negative:
            explanation_parts.append(
                "This odds ratio is significantly below 1, indicating that, after adjustment, "
                "parents have lower odds of engaging in extramarital affairs than non-parents."
            )
        elif logistic_negative and not logistic_significant:
            explanation_parts.append(
                "The estimated odds ratio is below 1 (suggesting fewer affairs among parents), "
                "but the confidence interval includes 1 and the association is not statistically "
                "significant at the 5% level."
            )
        elif (not logistic_negative) and logistic_significant:
            explanation_parts.append(
                "The odds ratio is at or above 1 and statistically significant, indicating no evidence "
                "that having children reduces engagement in extramarital affairs and possibly a modest increase."
            )
        else:
            explanation_parts.append(
                "The regression results do not provide clear evidence that having children "
                "reduces engagement in extramarital affairs."
            )
    else:
        explanation_parts.append(
            "The logistic regression model could not be reliably fit; conclusions are based "
            "solely on descriptive differences between parents and non-parents."
        )

    if response == "Yes":
        explanation_parts.append(
            "Overall, the balance of evidence from both descriptive statistics and the "
            "regression model supports the conclusion that having children is associated "
            "with lower engagement in extramarital affairs in this sample."
        )
    else:
        explanation_parts.append(
            "Overall, the data do not provide strong enough evidence to conclude that having "
            "children reduces engagement in extramarital affairs; any apparent differences are "
            "small and/or statistically uncertain in this sample."
        )

    explanation = " ".join(explanation_parts)

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()

