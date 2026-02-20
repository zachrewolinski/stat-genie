import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Create binary indicator for any affair.
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Map children to numeric for clarity (yes=1, no=0).
    df["children_num"] = df["children"].map({"yes": 1, "no": 0})

    # Descriptive statistics: means by children status.
    group_means = (
        df.groupby("children")[["affairs", "had_affair"]]
        .mean()
        .rename(columns={"affairs": "mean_affairs", "had_affair": "prop_any_affair"})
    )

    # Simple two-sample comparison for mean number of affairs.
    affairs_children_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_children_no = df.loc[df["children"] == "no", "affairs"]

    # Use Welch's t-test via statsmodels.
    ttest_res = sm.stats.ttest_ind(
        affairs_children_yes,
        affairs_children_no,
        usevar="unequal",
    )
    t_stat, p_value, _ = ttest_res

    # Logistic regression for any affair, controlling for key covariates.
    formula = (
        "had_affair ~ children_num + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)
    children_coef = logit_model.params["children_num"]
    children_pvalue = logit_model.pvalues["children_num"]
    children_odds_ratio = float(np.exp(children_coef))

    # Decide on qualitative conclusion: does having children decrease affairs?
    # We interpret "decrease" as children associated with LOWER odds of any affair.
    # Negative coefficient / odds ratio < 1 would support "Yes"; otherwise "No".
    if children_odds_ratio < 1 and children_pvalue < 0.05:
        # Evidence that children substantially decrease affairs.
        response_score = 80
        qualitative = (
            "The data provide statistically significant evidence that having children "
            "is associated with LOWER odds of engaging in an extramarital affair, "
            "even after controlling for age, years married, religiousness, education, "
            "occupation, marital satisfaction rating, and gender."
        )
    elif children_odds_ratio < 1 and children_pvalue >= 0.05:
        # Direction suggests a decrease but not statistically significant.
        response_score = 60
        qualitative = (
            "The estimated effect of having children points toward slightly LOWER odds "
            "of engaging in an extramarital affair, but this effect is not "
            "statistically significant after controlling for age, years married, "
            "religiousness, education, occupation, marital satisfaction rating, and "
            "gender."
        )
    elif children_odds_ratio > 1 and children_pvalue < 0.05:
        # Evidence that children are associated with more affairs (opposite of claim).
        response_score = 10
        qualitative = (
            "The data provide statistically significant evidence that having children "
            "is associated with HIGHER odds of engaging in an extramarital affair, "
            "even after controlling for age, years married, religiousness, education, "
            "occupation, marital satisfaction rating, and gender."
        )
    else:
        # No clear evidence in either direction.
        response_score = 40
        qualitative = (
            "After controlling for age, years married, religiousness, education, "
            "occupation, marital satisfaction rating, and gender, the data do not "
            "provide clear statistically significant evidence that having children "
            "either increases or decreases the odds of engaging in an extramarital "
            "affair."
        )

    # Build detailed explanation incorporating descriptive and inferential results.
    # Round key numbers for readability.
    mean_stats = group_means.to_dict(orient="index")
    yes_stats = mean_stats.get("yes", {})
    no_stats = mean_stats.get("no", {})

    explanation = (
        "Research question: Does having children decrease engagement in extramarital affairs?\n\n"
        "1) Descriptive patterns by children status:\n"
        f"- Mean number of affairs (children = yes): {yes_stats.get('mean_affairs', float('nan')):.3f}\n"
        f"- Mean number of affairs (children = no): {no_stats.get('mean_affairs', float('nan')):.3f}\n"
        f"- Proportion with any affair (children = yes): {yes_stats.get('prop_any_affair', float('nan')):.3f}\n"
        f"- Proportion with any affair (children = no): {no_stats.get('prop_any_affair', float('nan')):.3f}\n"
        f"- Welch t-test comparing mean number of affairs (yes vs. no children): "
        f"t = {t_stat:.3f}, p = {p_value:.3f}\n\n"
        "These descriptive statistics show how both the average number of affairs and the "
        "probability of having any affair differ between couples with and without children.\n\n"
        "2) Logistic regression for any affair (had_affair):\n"
        "Model: had_affair ~ children + age + yearsmarried + religiousness + education + "
        "occupation + rating + gender.\n"
        f"- Coefficient for having children (log-odds scale): {children_coef:.3f}\n"
        f"- Odds ratio for having children: {children_odds_ratio:.3f}\n"
        f"- p-value for children effect: {children_pvalue:.3f}\n\n"
        "Interpreting the logistic model, an odds ratio greater than 1 means that having "
        "children is associated with higher odds of having an affair, whereas an odds ratio "
        "less than 1 means lower odds, after adjusting for other variables.\n\n"
        f"3) Overall conclusion:\n{qualitative}\n\n"
        "Taken together, the descriptive comparisons and the multivariable logistic regression "
        "lead to the overall answer encoded in the 0–100 response scale, where 0 corresponds "
        "to a strong 'No' and 100 to a strong 'Yes' to the question of whether having children "
        "decreases engagement in extramarital affairs."
    )

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    # Write required JSON object to conclusion.txt with no extra text.
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

