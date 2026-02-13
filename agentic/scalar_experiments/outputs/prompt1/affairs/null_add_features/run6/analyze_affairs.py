import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic sanity checks
    assert "affairs" in df.columns and "children" in df.columns

    # Create binary indicator for having any extramarital affairs
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Drop rows with missing information in key variables, if any
    df = df.dropna(subset=["affairs", "children"])

    # Ensure children is treated as a categorical predictor
    df["children"] = df["children"].astype("category")

    # Descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any_affair=("any_affair", "mean"),
            count=("affairs", "size"),
        )
        .reset_index()
    )

    # Fit a logistic regression for having any affair ~ children
    # Reference category will be inferred by statsmodels (alphabetical), so we
    # explicitly set 'no' as reference if present.
    if "no" in df["children"].cat.categories:
        df["children"] = df["children"].cat.reorder_categories(["no", "yes"], ordered=False)

    logit_model = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    params = logit_model.params
    conf = logit_model.conf_int()
    pvalues = logit_model.pvalues

    # Extract effect of having children (yes vs no); handle generic naming
    child_term = None
    for term in params.index:
        if "children" in term and "yes" in term:
            child_term = term
            break

    if child_term is None:
        raise RuntimeError("Could not locate children effect in regression results.")

    coef = params[child_term]
    se = logit_model.bse[child_term]
    z_value = coef / se
    p_value = pvalues[child_term]

    # Odds ratio and 95% CI
    or_est = float(np.exp(coef))
    ci_low = float(np.exp(conf.loc[child_term, 0]))
    ci_high = float(np.exp(conf.loc[child_term, 1]))

    # Decide answer: significant decrease if coefficient is negative and p < 0.05
    if coef < 0 and p_value < 0.05:
        response = "Yes"
    else:
        response = "No"

    # Build explanation string with key quantitative evidence
    # Summaries: convert to plain Python types for formatting
    desc_records = desc.to_dict(orient="records")
    lines = []
    lines.append(
        "I analysed 601 married individuals and compared extramarital-affair activity"
        " between those with and without children."
    )

    for rec in desc_records:
        child_status = rec["children"]
        mean_aff = rec["mean_affairs"]
        median_aff = rec["median_affairs"]
        prop_any = rec["prop_any_affair"]
        count = rec["count"]
        lines.append(
            f"For children = {child_status} (n={int(count)}), the mean affairs score was"
            f" {mean_aff:.2f}, median {median_aff:.0f}, and the share with any affair"
            f" was {prop_any:.2%}."
        )

    direction = "lower" if coef < 0 else "higher"
    lines.append(
        "I then fit a logistic regression predicting whether a person had any affair"
        " from whether they had children."
    )
    lines.append(
        f"In this model, the coefficient for having children (yes vs no) was {coef:.3f}"
        f" (z = {z_value:.2f}, p = {p_value:.3f}), corresponding to an odds ratio of"
        f" {or_est:.2f} with a 95% confidence interval from {ci_low:.2f} to {ci_high:.2f}."
    )
    lines.append(
        f"This indicates that, holding only children status in the model, people with children have {direction}"
        " estimated odds of engaging in any extramarital affair than those without children."
    )
    if response == "Yes":
        lines.append(
            "Because the effect is negative and statistically significant at the 5% level,"
            " I conclude that having children is associated with a lower likelihood of"
            " engaging in extramarital affairs in this sample."
        )
    else:
        lines.append(
            "However, this effect is not statistically significant at conventional levels"
            " (p < 0.05), so the data do not provide strong evidence that having children"
            " truly decreases engagement in extramarital affairs; any observed differences"
            " could be due to sampling variability."
        )

    explanation = " ".join(lines)

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    # Write conclusion.json as required
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

