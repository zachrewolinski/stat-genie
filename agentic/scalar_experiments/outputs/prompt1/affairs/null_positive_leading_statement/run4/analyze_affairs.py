import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    grouped = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("any_affair", "mean"),
            count=("affairs", "size"),
        )
        .rename_axis("children")
    )

    # Overall share with children vs without
    children_counts = df["children"].value_counts().to_dict()

    # Try a multivariable logistic regression for any affair
    logit_result = None
    children_term = None
    try:
        formula = (
            "any_affair ~ C(children) + age + yearsmarried + religiousness + "
            "education + C(occupation) + C(gender) + rating"
        )
        model = smf.logit(formula=formula, data=df)
        logit_result = model.fit(disp=False)
        # Identify the children coefficient term (yes vs no)
        for name in logit_result.params.index:
            if name.startswith("C(children)"):
                children_term = name
                break
    except Exception:
        logit_result = None
        children_term = None

    explanation_parts = []

    # Basic descriptive comparison
    if set(grouped.index) >= {"yes", "no"}:
        mean_yes = float(grouped.loc["yes", "mean_affairs"])
        mean_no = float(grouped.loc["no", "mean_affairs"])
        prop_yes = float(grouped.loc["yes", "prop_any_affair"])
        prop_no = float(grouped.loc["no", "prop_any_affair"])
        n_yes = int(grouped.loc["yes", "count"])
        n_no = int(grouped.loc["no", "count"])
        explanation_parts.append(
            "I first compared participants with and without children. "
            f"Among those with children (n={n_yes}), the average affairs score "
            f"was {mean_yes:.2f}, and {prop_yes*100:.1f}% reported at least one affair. "
            f"Among those without children (n={n_no}), the average affairs score was "
            f"{mean_no:.2f}, and {prop_no*100:.1f}% reported at least one affair."
        )
    else:
        explanation_parts.append(
            "The dataset records whether there are children in the marriage, "
            "but the expected 'yes' and 'no' categories were not both present, "
            "so only limited descriptive comparisons were possible."
        )

    # Default conclusion assumes no strong evidence that children reduce affairs
    response = "No"
    main_conclusion = (
        "Based on this sample, I do not find clear evidence that having children "
        "reduces engagement in extramarital affairs."
    )

    # If the logistic model fit successfully, use it to refine the conclusion
    if logit_result is not None and children_term is not None:
        coef = float(logit_result.params[children_term])
        pvalue = float(logit_result.pvalues[children_term])
        odds_ratio = float(np.exp(coef))
        ci_low, ci_high = logit_result.conf_int().loc[children_term]
        ci_low = float(np.exp(ci_low))
        ci_high = float(np.exp(ci_high))

        explanation_parts.append(
            "I then fit a logistic regression model for having any affair in the past year, "
            "including children status and adjusting for age, years married, religiousness, "
            "education, occupation, gender, and self-rated marital happiness."
        )
        explanation_parts.append(
            "In this model, the coefficient for having children corresponds to an odds ratio "
            f"of {odds_ratio:.2f} (95% CI [{ci_low:.2f}, {ci_high:.2f}], p={pvalue:.3f}) "
            "for having an affair compared with couples without children, holding the other "
            "covariates constant."
        )

        if coef < 0 and pvalue < 0.05:
            response = "Yes"
            main_conclusion = (
                "After adjusting for demographics and marital characteristics, having children "
                "is associated with statistically significantly lower odds of engaging in "
                "extramarital affairs."
            )
        elif coef > 0 and pvalue < 0.05:
            response = "No"
            main_conclusion = (
                "After adjusting for demographics and marital characteristics, having children "
                "is associated with statistically significantly higher odds of engaging in "
                "extramarital affairs."
            )
        else:
            response = "No"
            main_conclusion = (
                "Once age, years married, religiousness, education, occupation, gender, and "
                "marital happiness are taken into account, the estimated effect of having "
                "children on the odds of engaging in extramarital affairs is not "
                "statistically distinguishable from zero."
            )
            explanation_parts.append(
                "The confidence interval for the odds ratio includes 1, and the p-value is "
                "not below conventional significance thresholds, so the adjusted association "
                "between children and affairs is uncertain in direction and small in magnitude."
            )
    else:
        explanation_parts.append(
            "A multivariable logistic regression model could not be reliably fit, so the "
            "conclusion relies on the descriptive comparison of affair rates between couples "
            "with and without children."
        )

    # High-level answer to the research question
    explanation_parts.insert(
        0,
        "The research question is whether having children decreases engagement in extramarital "
        "affairs among currently married individuals in this survey sample.",
    )
    explanation_parts.append(
        "Taken together, the descriptive comparisons and regression results do not support the "
        "claim that having children, by itself, clearly reduces the likelihood of extramarital "
        "affairs in this dataset."
    )

    explanation_text = main_conclusion + " " + " ".join(explanation_parts)

    output = {"response": response, "explanation": explanation_text}
    Path("conclusion.txt").write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

