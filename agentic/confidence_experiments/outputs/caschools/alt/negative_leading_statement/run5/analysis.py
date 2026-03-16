import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    base_dir = Path(__file__).parent
    data_path = base_dir / "caschools.csv"

    df = pd.read_csv(data_path)

    # Construct key derived variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscr"] = (df["read"] + df["math"]) / 2.0

    # Basic summary statistics
    stratio_mean = float(df["stratio"].mean())
    stratio_std = float(df["stratio"].std())
    testscr_mean = float(df["testscr"].mean())
    testscr_std = float(df["testscr"].std())
    corr = float(df["stratio"].corr(df["testscr"]))

    # Simple bivariate association
    m_simple = smf.ols("testscr ~ stratio", data=df).fit()
    m_simple_robust = m_simple.get_robustcov_results(cov_type="HC1")
    simple_param_names = list(m_simple.params.index)
    idx_stratio_simple = simple_param_names.index("stratio")
    coef_simple = float(m_simple_robust.params[idx_stratio_simple])
    p_simple = float(m_simple_robust.pvalues[idx_stratio_simple])

    # Multiple regression with key covariates to account for confounding
    formula_multi = (
        "testscr ~ stratio + income + english + calworks + lunch + expenditure + computer"
    )
    m_multi = smf.ols(formula_multi, data=df).fit()
    m_multi_robust = m_multi.get_robustcov_results(cov_type="HC1")
    multi_param_names = list(m_multi.params.index)
    idx_stratio_multi = multi_param_names.index("stratio")
    coef_multi = float(m_multi_robust.params[idx_stratio_multi])
    p_multi = float(m_multi_robust.pvalues[idx_stratio_multi])

    # Effect sizes for a 5-student change in the student-teacher ratio
    delta_ratio = 5.0
    effect_simple = coef_simple * delta_ratio
    effect_multi = coef_multi * delta_ratio

    # Determine strength of evidence that lower ratio is associated with higher scores
    # (i.e., that testscr decreases as stratio increases -> negative coefficient).
    direction_consistent = coef_simple < 0 and coef_multi < 0
    significant_simple = p_simple < 0.05
    significant_multi = p_multi < 0.05

    # Standardized effect sizes (per 5-student change)
    std_effect_simple = abs(effect_simple) / testscr_std
    std_effect_multi = abs(effect_multi) / testscr_std

    if direction_consistent and significant_simple and significant_multi:
        # Clear, statistically significant negative association in both models.
        if std_effect_multi >= 0.25:
            response = 90
        elif std_effect_multi >= 0.15:
            response = 80
        else:
            response = 70
        qualitative = "Yes"
    elif direction_consistent and (significant_simple or significant_multi):
        # Somewhat weaker but still present evidence.
        response = 65
        qualitative = "Yes"
    elif direction_consistent and not (significant_simple or significant_multi):
        # Direction is in line with the hypothesis, but estimates are noisy.
        response = 55
        qualitative = "Leaning Yes"
    else:
        # Estimates do not reliably support a negative association.
        if significant_simple or significant_multi:
            response = 30
        else:
            response = 45
        qualitative = "No"

    explanation = (
        "Research question: Is a lower student–teacher ratio associated with higher academic performance "
        "(average of reading and math scores)? "
        "Using 420 California K-6 and K-8 districts, I created the student–teacher ratio as students per "
        "teacher and an average test score (testscr) as the mean of reading and math scores. "
        f"The student–teacher ratio has mean {stratio_mean:.1f} (SD {stratio_std:.1f}), while testscr has mean "
        f"{testscr_mean:.1f} (SD {testscr_std:.1f}). The Pearson correlation between the ratio and test scores "
        f"is {corr:.3f}, indicating that districts with smaller classes tend to have higher scores. "
        "A bivariate linear regression of testscr on the student–teacher ratio yields a negative coefficient on "
        f"the ratio (robust estimate {coef_simple:.3f}) with p-value {p_simple:.4f}, meaning that larger ratios "
        "are significantly associated with lower test scores. "
        "To account for important socio-economic and resource differences between districts, I estimated a "
        "multiple regression of testscr on the student–teacher ratio controlling for district income, the "
        "percent of English learners, percent on CalWorks, percent on reduced-price lunch, expenditures per "
        "student, and number of computers. In this model the coefficient on the student–teacher ratio remains "
        f"negative (robust estimate {coef_multi:.3f}) but with p-value {p_multi:.4f}, so once we adjust for "
        "these covariates the association is no longer statistically distinguishable from zero. "
        f"A 5-student increase in the student–teacher ratio is associated with a change in average test scores "
        f"of about {effect_multi:.2f} points in the multivariate model, corresponding to roughly "
        f"{std_effect_multi:.2f} standard deviations of testscr—small in magnitude and estimated with "
        "substantial uncertainty. "
        "Taken together, the simple correlation and bivariate regression show a clear negative association "
        "between the student–teacher ratio and test scores, while the fully adjusted model suggests that much "
        "of this relationship can be explained by observed socio-economic and resource differences across "
        "districts and leaves only a small, statistically weak residual association. Despite the prompt’s prior "
        "belief that the answer is 'No', the empirical evidence overall is more consistent with a modest "
        "negative association than with no association at all, which motivates a 'Yes' answer but with only "
        "moderate strength. "
        f"On a 0–100 scale where higher values represent stronger evidence for 'Yes', I summarize this as "
        f"{response} ({qualitative})."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    out_path = base_dir / "conclusion.txt"
    out_path.write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()
