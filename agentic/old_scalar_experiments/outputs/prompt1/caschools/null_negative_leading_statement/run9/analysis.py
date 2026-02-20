import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    cwd = Path(".")
    info_path = cwd / "info.json"
    data_path = cwd / "caschools.csv"

    info = load_metadata(info_path)
    df = pd.read_csv(data_path)

    # Construct key variables for the research question.
    # Student-teacher ratio.
    df["stratio"] = df["students"] / df["teachers"]
    # Overall academic performance as mean of reading and math scores.
    df["testscr"] = df[["read", "math"]].mean(axis=1)

    # Simple bivariate association: testscr ~ stratio
    X_simple = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model_simple = sm.OLS(y, X_simple).fit()

    beta_stratio_simple = float(model_simple.params["stratio"])
    pvalue_stratio_simple = float(model_simple.pvalues["stratio"])

    # Multiple regression controlling for key socio-economic factors.
    controls = ["income", "english", "lunch", "calworks"]
    X_controls = sm.add_constant(df[["stratio"] + controls])
    model_controls = sm.OLS(y, X_controls).fit()

    beta_stratio_controls = float(model_controls.params["stratio"])
    pvalue_stratio_controls = float(model_controls.pvalues["stratio"])

    conf_int_simple = model_simple.conf_int().loc["stratio"].tolist()
    conf_int_controls = model_controls.conf_int().loc["stratio"].tolist()

    # Decision rule: if higher student-teacher ratio is associated with
    # significantly lower test scores (negative coefficient with p < 0.05)
    # in both models, we answer "Yes".
    def is_supportive(beta: float, pval: float) -> bool:
        return beta < 0 and pval < 0.05

    supportive_simple = is_supportive(beta_stratio_simple, pvalue_stratio_simple)
    supportive_controls = is_supportive(beta_stratio_controls, pvalue_stratio_controls)

    if supportive_simple and supportive_controls:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "Research question: 'Is a lower student-teacher ratio associated with higher academic performance?'\n\n"
        "Using the California K-6/K-8 school districts dataset (n = {n}), I constructed the student-teacher "
        "ratio as students divided by teachers and an overall academic performance measure as the mean of "
        "reading and math scores.\n\n"
        "First, I ran an ordinary least squares regression of average test score on the student-teacher ratio "
        "alone. The estimated coefficient on the ratio is {b1:.3f}, with a p-value of {p1:.4f} and a 95% "
        "confidence interval of [{c1_low:.3f}, {c1_high:.3f}]. A negative and statistically significant "
        "coefficient would indicate that districts with fewer students per teacher tend to have higher test "
        "scores.\n\n"
        "Second, to account for important socio-economic and demographic differences across districts, I "
        "estimated a multiple regression including income, percent English learners, percent eligible for "
        "reduced-price lunch, and percent on CalWorks as controls. In this model, the coefficient on the "
        "student-teacher ratio is {b2:.3f}, with a p-value of {p2:.4f} and a 95% confidence interval of "
        "[{c2_low:.3f}, {c2_high:.3f}].\n\n"
        "Based on these results, the evidence that lower student-teacher ratios are associated with higher "
        "academic performance is {strength}. I therefore answer the research question: {answer}."
    ).format(
        n=len(df),
        b1=beta_stratio_simple,
        p1=pvalue_stratio_simple,
        c1_low=conf_int_simple[0],
        c1_high=conf_int_simple[1],
        b2=beta_stratio_controls,
        p2=pvalue_stratio_controls,
        c2_low=conf_int_controls[0],
        c2_high=conf_int_controls[1],
        strength=(
            "strong" if supportive_simple and supportive_controls else
            "weak or statistically unreliable"
        ),
        answer=("Yes" if response == "Yes" else "No"),
    )

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = cwd / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

