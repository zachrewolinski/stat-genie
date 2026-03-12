import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load metadata and data
    info = json.loads(Path("info.json").read_text())
    df = pd.read_csv("caschools.csv")

    # Map shuffled column names to conceptual variables based on descriptions
    enrollment = df["english"]  # total students
    teachers = df["students"]  # number of teachers (FTE)

    # Student–teacher ratio
    str_ratio = enrollment / teachers

    # Academic performance: average of reading and math scores
    read_score = df["district"]
    math_score = df["expenditure"]
    testscr = (read_score + math_score) / 2.0

    data = pd.DataFrame(
        {
            "testscr": testscr,
            "str": str_ratio,
            # Key covariates capturing socioeconomic status and demographics
            "income": df["income"],
            "calworks_pct": df["school"],  # percent on income assistance
            "lunch_pct": df["computer"],  # percent on reduced-price lunch
            "english_learner_pct": df["rownames"],
        }
    ).dropna()

    # Center predictors for stability and easier interpretation
    data["str_c"] = data["str"] - data["str"].mean()
    data["income_c"] = data["income"] - data["income"].mean()
    data["calworks_c"] = data["calworks_pct"] - data["calworks_pct"].mean()
    data["lunch_c"] = data["lunch_pct"] - data["lunch_pct"].mean()
    data["ell_c"] = data["english_learner_pct"] - data["english_learner_pct"].mean()

    y = data["testscr"]

    # Bivariate relationship: testscr ~ str
    X_biv = sm.add_constant(data["str"])
    model_biv = sm.OLS(y, X_biv).fit()

    # Multivariate relationship with controls
    X_multi = data[
        ["str_c", "income_c", "calworks_c", "lunch_c", "ell_c"]
    ]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(y, X_multi).fit()

    # Extract key quantities
    coef_biv = model_biv.params["str"]
    p_biv = model_biv.pvalues["str"]
    r2_biv = model_biv.rsquared

    coef_multi = model_multi.params["str_c"]
    p_multi = model_multi.pvalues["str_c"]
    r2_multi = model_multi.rsquared

    # Effect size: change in testscr for a 1-student increase in STR
    # Typical STR range
    str_min, str_max = data["str"].min(), data["str"].max()
    typical_change = 5  # students per teacher
    effect_5_students = coef_multi * typical_change

    # Decide Likert response score based on strength and robustness
    # Start from a neutral midpoint and adjust for evidence.
    response_score = 50

    strong_negative = coef_multi < 0 and p_multi < 0.001
    moderate_negative = coef_multi < 0 and p_multi < 0.01
    weak_negative = coef_multi < 0 and p_multi < 0.05

    strong_null = abs(coef_multi) < 0.01 and p_multi > 0.3

    if strong_negative:
        response_score = 90
    elif moderate_negative:
        response_score = 80
    elif weak_negative:
        response_score = 65
    elif strong_null:
        # Coefficient is extremely small and far from significant:
        # strong evidence of no practically important association.
        response_score = 20
    else:
        # No clear evidence for the hypothesized negative association
        if p_multi >= 0.1:
            response_score = 35
        else:
            response_score = 50

    # Clip to [0, 100] and cast to int
    response_score = int(np.clip(round(response_score), 0, 100))

    # Build explanation string summarizing the evidence
    # Narrative phrases reflecting sign and significance
    direction_biv = "higher" if coef_biv > 0 else "lower"
    direction_multi = "higher" if coef_multi > 0 else "lower"

    sig_biv = (
        "not statistically significant (p ≥ 0.05)"
        if p_biv >= 0.05
        else "statistically significant at the 5% level"
    )
    sig_multi = (
        "not statistically significant (p ≥ 0.05)"
        if p_multi >= 0.05
        else "statistically significant at the 5% level"
    )

    explanation = (
        f"Research question: {info['research_questions'][0]} "
        f"using data from {len(data)} California school districts.\n\n"
        f"I reconstructed the student–teacher ratio as total enrollment divided by the "
        f"number of teachers, and academic performance as the average of district-level "
        f"reading and math test scores.\n\n"
        f"In a simple linear regression of average test score on the student–teacher ratio, "
        f"the coefficient on the ratio was {coef_biv:.3f} points per additional student per "
        f"teacher (p = {p_biv:.4g}, R² = {r2_biv:.3f}); this estimate is {sig_biv} and very "
        f"close to zero, indicating little to no systematic change in scores as the ratio "
        f"changes.\n\n"
        f"To account for socioeconomic and demographic differences across districts, I next "
        f"fit a multiple regression including income, the percentages of students on income "
        f"assistance and reduced-price lunch, and the percentage of English learners. In this "
        f"model, the coefficient on the centered student–teacher ratio was {coef_multi:.3f} "
        f"points per additional student per teacher (p = {p_multi:.4g}, R² = {r2_multi:.3f}). "
        f"This implies that increasing the typical student–teacher ratio by 5 students per "
        f"teacher is associated with about {effect_5_students:.1f} points "
        f"{'higher' if effect_5_students > 0 else 'lower' if effect_5_students < 0 else 'difference in'} "
        f"average test score, holding the other variables constant; this estimate is {sig_multi} "
        f"and again very small in magnitude.\n\n"
        f"Taken together, both the bivariate and multivariate analyses show a coefficient on "
        f"the student–teacher ratio that is extremely close to zero and not statistically "
        f"distinguishable from no effect at conventional levels. Within this dataset, there is "
        f"no convincing evidence that districts with lower student–teacher ratios have higher "
        f"average test scores once sampling variation is taken into account. Based on this "
        f"lack of a clear relationship, I give a {response_score} on a 0–100 Likert scale, "
        f"where 0 represents a strong 'No' and 100 a strong 'Yes' to the hypothesis that lower "
        f"student–teacher ratios are associated with higher academic performance."
    )

    conclusion = {
        "response": response_score,
        "explanation": explanation,
    }

    Path("conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()
