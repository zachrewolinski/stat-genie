import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata / research question (not strictly needed for stats, but for context).
    info_path = base_dir / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    question = info.get("research_questions", [""])[0]

    # Load dataset.
    df = pd.read_csv(base_dir / "affairs.csv")

    # Binary indicator of any extramarital affair in the past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children status.
    prop_table = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "sum", "size"])
        .rename(columns={"mean": "prop_any_affair", "sum": "count_any_affair", "size": "n"})
    )

    mean_counts = df.groupby("children")["affairs"].mean()

    # Difference in proportions test (children=yes vs no) for any affair.
    # Order groups explicitly to keep interpretation clear.
    counts = prop_table.loc[["no", "yes"], "count_any_affair"].to_numpy()
    nobs = prop_table.loc[["no", "yes"], "n"].to_numpy()
    z_stat, p_prop = proportions_ztest(count=counts, nobs=nobs, alternative="two-sided")

    # Logistic regression with covariates to account for other factors.
    # Model probability of any affair as a function of children plus demographic/marital variables.
    logit_model = smf.logit(
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)",
        data=df,
    ).fit(disp=False)

    params = logit_model.params
    pvalues = logit_model.pvalues

    # Effect of having children ("yes") relative to "no".
    coef_children_yes = float(params.get("C(children)[T.yes]", 0.0))
    p_children_yes = float(pvalues.get("C(children)[T.yes]", 1.0))

    # Decide on Yes/No:
    # "Yes" if there is clear evidence that having children decreases engagement in affairs:
    #   (a) parents have lower observed affair rates, and
    #   (b) the children coefficient in the logistic regression is negative and statistically significant.
    prop_no_children = float(prop_table.loc["no", "prop_any_affair"])
    prop_yes_children = float(prop_table.loc["yes", "prop_any_affair"])

    decreases_in_raw_data = prop_yes_children < prop_no_children
    significant_negative_coef = (coef_children_yes < 0) and (p_children_yes < 0.05)

    if decreases_in_raw_data and significant_negative_coef:
        response = "Yes"
    else:
        response = "No"

    # Build explanation string summarizing evidence.
    explanation = (
        f"Research question: {question.strip()} "
        f"In the data, the proportion of individuals with any extramarital affair is "
        f"{prop_no_children:.3f} among marriages without children and "
        f"{prop_yes_children:.3f} among marriages with children. "
        f"The mean affair count is {mean_counts['no']:.3f} for couples without children "
        f"and {mean_counts['yes']:.3f} for couples with children. "
        f"A difference-in-proportions z-test for any affair between the two groups yields "
        f"z = {z_stat:.3f}, p = {p_prop:.3f}. "
        f"A logistic regression of any affair on children status and covariates "
        f"(age, years married, religiousness, education, occupation, marital rating, and gender) "
        f"estimates the log-odds coefficient for having children (yes vs no) as "
        f"{coef_children_yes:.3f} with p-value {p_children_yes:.3f}. "
        f"Given these results, the evidence that having children decreases engagement in extramarital "
        f"affairs is not strong enough to be considered statistically meaningful at conventional levels, "
        f"so the data do not support a clear decrease in extramarital affairs among couples with children."
    )

    output = {"response": response, "explanation": explanation}

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

