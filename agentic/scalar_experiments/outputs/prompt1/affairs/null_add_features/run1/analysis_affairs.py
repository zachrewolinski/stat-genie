import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Ensure children has a consistent ordering: baseline = "no children"
    if "children" in df.columns:
        df["children"] = pd.Categorical(df["children"], categories=["no", "yes"])

    # Keep a core set of covariates that are plausibly related to affairs
    covariates = ["children", "gender", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    available_covariates = [c for c in covariates if c in df.columns]
    formula_terms = ["C(children)"]

    for col in available_covariates:
        if col == "children":
            continue
        if df[col].dtype == "O":
            formula_terms.append(f"C({col})")
        else:
            formula_terms.append(col)

    formula = "any_affair ~ " + " + ".join(sorted(set(formula_terms)))

    # Fit logistic regression
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Extract effect of having children (yes vs no)
    # With the categorical ordering above, this term measures the log-odds difference
    coef_name = "C(children)[T.yes]"
    if coef_name not in model.params.index:
        raise RuntimeError(f"Expected coefficient {coef_name!r} not found in model.")

    coef = float(model.params[coef_name])
    se = float(model.bse[coef_name])
    z_value = float(model.tvalues[coef_name])
    p_value = float(model.pvalues[coef_name])
    odds_ratio = float(np.exp(coef))

    # Decide answer: Does having children decrease engagement in extramarital affairs?
    # We interpret this as: is having children associated with a *lower* likelihood
    # of having an affair, and is the effect at least moderately statistically supported?
    alpha = 0.05
    decreases_affairs = (coef < 0) and (p_value < alpha)

    if decreases_affairs:
        response = "Yes"
    else:
        response = "No"

    # Build plain-language explanation
    explanation = (
        "I modelled the probability of having any extramarital affair in the past year "
        "using a logistic regression with a binary outcome (any affair vs. none). "
        "The key predictor was whether the respondent had children, and I controlled "
        "for gender, age, years married, religiousness, education, occupation, and "
        "self-rated marriage quality (rating). "
        f"In this model, the coefficient for having children (yes vs. no) was {coef:.3f}, "
        f"with a standard error of {se:.3f}, z = {z_value:.2f}, and p-value = {p_value:.3f}. "
        f"This corresponds to an odds ratio of {odds_ratio:.3f} for having at least one affair "
        "among people with children compared with those without. "
        f"Because the estimated effect is {'negative' if coef < 0 else 'positive'} but "
        f"{'statistically significant' if p_value < alpha else 'not statistically significant'} "
        f"at the 5% level, I conclude that having children "
        f"{'is associated with a lower likelihood of extramarital affairs' if decreases_affairs else 'does not show a clear, statistically robust decrease in extramarital affairs'}. "
        "This conclusion is based on the sign, size, and statistical significance of the "
        "children coefficient in the multivariable logistic model, as well as simple "
        "comparisons of average affair rates between couples with and without children."
    )

    conclusion = {"response": response, "explanation": explanation}

    # Write required JSON output to conclusion.txt with no extra text
    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

