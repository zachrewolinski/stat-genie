from pathlib import Path
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path(__file__).with_name("affairs.csv")
    df = pd.read_csv(data_path)

    # Outcome: any extramarital intercourse in past year
    df["affair_any"] = (df["feature2"] > 0).astype(int)

    # Exposure: having children in the marriage
    df["has_children"] = (df["feature6"] == "yes").astype(int)

    # Basic group-level summaries
    group = (
        df.groupby("has_children")["affair_any"]
        .agg(["mean", "count", "sum"])
        .rename(index={0: "no_children", 1: "children"})
    )

    rate_no_children = float(group.loc["no_children", "mean"])
    rate_children = float(group.loc["children", "mean"])
    n_no_children = int(group.loc["no_children", "count"])
    n_children = int(group.loc["children", "count"])

    # Logistic regression with controls for key covariates
    formula = (
        "affair_any ~ has_children + C(feature3) + "
        "feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    coef_children = float(model.params["has_children"])
    pval_children = float(model.pvalues["has_children"])
    odds_ratio = float(np.exp(coef_children))

    # Average predicted probabilities with vs without children,
    # holding the rest of the covariate distribution fixed.
    df_with_children = df.copy()
    df_with_children["has_children"] = 1
    df_without_children = df.copy()
    df_without_children["has_children"] = 0

    mean_prob_with = float(model.predict(df_with_children).mean())
    mean_prob_without = float(model.predict(df_without_children).mean())
    delta_prob = mean_prob_with - mean_prob_without

    # Map evidence strength and direction to a 0-100 Likert score.
    # Negative effect (children associated with fewer affairs) -> higher score.
    if coef_children < 0:
        if pval_children < 0.01:
            response = 90
        elif pval_children < 0.05:
            response = 80
        elif pval_children < 0.1:
            response = 70
        else:
            response = 60
    elif coef_children > 0:
        if pval_children < 0.01:
            response = 10
        elif pval_children < 0.05:
            response = 20
        elif pval_children < 0.1:
            response = 30
        else:
            response = 40
    else:
        response = 50

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n\n"
        "Operationalization:\n"
        "- Outcome: a binary indicator coded as 1 if the respondent reported any "
        "extramarital intercourse in the past year (feature2 > 0) and 0 otherwise.\n"
        "- Exposure: a binary indicator coded as 1 if there are children in the marriage "
        "(feature6 == 'yes') and 0 if there are no children.\n\n"
        "Descriptive evidence:\n"
        f"- Among respondents without children (n = {n_no_children}), the proportion with any "
        f"extramarital intercourse in the past year is {rate_no_children:.3f}.\n"
        f"- Among respondents with children (n = {n_children}), the proportion is "
        f"{rate_children:.3f}.\n\n"
        "Model-based evidence:\n"
        "I fit a logistic regression for having any affair (binary outcome) on the children "
        "indicator, controlling for gender (feature3), age (feature4), years married "
        "(feature5), religiousness (feature7), education (feature8), occupation (feature9), "
        "and self-rated marital happiness (feature10).\n"
        f"- The coefficient for having children is {coef_children:.3f}, corresponding to an "
        f"odds ratio of {odds_ratio:.3f}.\n"
        f"- The p-value for this coefficient is {pval_children:.3g}.\n"
        f"- Holding the covariate distribution fixed, the model's average predicted "
        f"probability of any affair is {mean_prob_without:.3f} with no children versus "
        f"{mean_prob_with:.3f} with children (difference = {delta_prob:.3f}, "
        "children minus no children).\n\n"
        "Interpretation:\n"
        "A negative coefficient and odds ratio below 1 indicate that having children is "
        "associated with a lower likelihood of engaging in extramarital affairs, after "
        "adjusting for the observed covariates. The magnitude of the effect and its "
        "statistical significance (as summarized by the p-value) reflect the strength of "
        "this evidence.\n\n"
        "Based on the direction and strength of the estimated effect from the logistic "
        "regression, along with the descriptive difference in affair rates between those "
        "with and without children, I summarize my answer on a 0-100 Likert scale where "
        "higher values indicate stronger evidence that having children decreases "
        "engagement in extramarital affairs."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    conclusion_path = Path(__file__).with_name("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

