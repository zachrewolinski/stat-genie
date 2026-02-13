import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError("affairs.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Basic cleaning / derived variables
    if "children" not in df.columns or "affairs" not in df.columns:
        raise ValueError("Expected 'children' and 'affairs' columns in the dataset.")

    # Drop rows with missing key variables, if any
    df = df.dropna(subset=["children", "affairs"])

    # Binary indicator: any extramarital affair in past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_with_affair=("has_affair", "mean"),
            n=("has_affair", "size"),
        )
        .reset_index()
    )

    # Prepare a logistic regression controlling for key covariates, where available
    covariates = []
    for col in [
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "gender",
    ]:
        if col in df.columns:
            covariates.append(col)

    logit_result = None
    coef_children = None
    pval_children = None
    or_children = None
    ci_lower = None
    ci_upper = None

    if len(covariates) > 0:
        # Build formula string
        numeric_terms = [c for c in covariates if c not in {"gender", "children"}]
        cat_terms = [c for c in covariates if c in {"gender"}]

        rhs_parts = []
        if numeric_terms:
            rhs_parts.append(" + ".join(numeric_terms))
        if cat_terms:
            rhs_parts.append(" + ".join([f"C({c})" for c in cat_terms]))
        # Children as a categorical predictor of interest
        rhs = " + ".join(filter(None, rhs_parts + ["C(children)"]))

        formula = f"has_affair ~ {rhs}"

        try:
            logit_result = smf.logit(formula, data=df).fit(disp=False)
            # Children coefficient is for having children vs no children
            # statsmodels encodes this as C(children)[T.yes] if levels are "no"/"yes"
            for name in logit_result.params.index:
                if name.startswith("C(children)"):
                    coef_children = float(logit_result.params[name])
                    pval_children = float(logit_result.pvalues[name])
                    or_children = float(np.exp(coef_children))
                    ci_low, ci_high = logit_result.conf_int().loc[name]
                    ci_lower = float(np.exp(ci_low))
                    ci_upper = float(np.exp(ci_high))
                    break
        except Exception:
            # If the model fails to converge for any reason, fall back to
            # descriptive comparison only.
            logit_result = None

    # Decide on response based on evidence
    # Primary evidence: sign and significance of children coefficient in logistic model.
    response = "No"
    reasoning_parts = []

    # Add descriptive comparison
    try:
        desc_children_yes = desc[desc["children"] == "yes"]
        desc_children_no = desc[desc["children"] == "no"]

        if not desc_children_yes.empty and not desc_children_no.empty:
            mean_yes = float(desc_children_yes["mean_affairs"].iloc[0])
            mean_no = float(desc_children_no["mean_affairs"].iloc[0])
            prop_yes = float(desc_children_yes["prop_with_affair"].iloc[0])
            prop_no = float(desc_children_no["prop_with_affair"].iloc[0])
            n_yes = int(desc_children_yes["n"].iloc[0])
            n_no = int(desc_children_no["n"].iloc[0])

            reasoning_parts.append(
                f"In the raw data, respondents with children (n={n_yes}) "
                f"had an average of {mean_yes:.2f} affairs and "
                f"{prop_yes:.1%} reported at least one affair, "
                f"whereas respondents without children (n={n_no}) had an "
                f"average of {mean_no:.2f} affairs and "
                f"{prop_no:.1%} reported at least one affair."
            )

            # If, even descriptively, children clearly reduce affairs, note it.
            if mean_yes < mean_no and prop_yes < prop_no:
                desc_direction = "lower"
            elif mean_yes > mean_no and prop_yes > prop_no:
                desc_direction = "higher"
            else:
                desc_direction = "mixed"
        else:
            desc_direction = "unknown"
    except Exception:
        desc_direction = "unknown"

    if coef_children is not None and pval_children is not None:
        direction = "decrease" if coef_children < 0 else "increase"
        reasoning_parts.append(
            "Using a logistic regression for having any extramarital affair, "
            "adjusting for available covariates (age, years married, religiousness, "
            "education, occupation, marital rating, and gender where present), "
            f"the coefficient for having children is {coef_children:.3f} on the "
            f"log-odds scale (odds ratio {or_children:.2f}, 95% CI "
            f"[{ci_lower:.2f}, {ci_upper:.2f}], p-value {pval_children:.3f})."
        )

        if coef_children < 0 and pval_children < 0.05:
            response = "Yes"
            reasoning_parts.append(
                "Because the children coefficient is negative and statistically "
                "significant at the 5% level, this provides evidence that, "
                "after adjusting for other factors, having children is associated "
                "with a decreased likelihood of engaging in extramarital affairs."
            )
        elif coef_children < 0 and pval_children >= 0.05:
            response = "No"
            reasoning_parts.append(
                "Although the estimated effect of having children points toward a "
                "decrease in the likelihood of affairs, the effect is not "
                "statistically distinguishable from zero at conventional levels, "
                "so the data do not provide strong evidence that children reduce "
                "engagement in extramarital affairs."
            )
        elif coef_children >= 0 and pval_children < 0.05:
            response = "No"
            reasoning_parts.append(
                "In fact, the estimated effect of having children is positive and "
                "statistically significant, indicating higher—not lower—odds of "
                "engaging in extramarital affairs among respondents with children."
            )
        else:
            response = "No"
            reasoning_parts.append(
                "The estimated effect of having children is not statistically "
                "significant, so the data do not support a clear conclusion that "
                "having children decreases engagement in extramarital affairs."
            )
    else:
        # Fall back entirely on descriptive comparisons
        if desc_direction == "lower":
            response = "Yes"
            reasoning_parts.append(
                "A regression model could not be reliably estimated, but the "
                "descriptive statistics suggest that respondents with children "
                "have fewer and less frequent affairs than those without children."
            )
        else:
            response = "No"
            reasoning_parts.append(
                "Because a regression model could not be reliably estimated and the "
                "descriptive comparisons do not consistently show fewer affairs "
                "among respondents with children, the data do not clearly support "
                "the claim that having children decreases engagement in extramarital affairs."
            )

    explanation = " ".join(reasoning_parts)

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

