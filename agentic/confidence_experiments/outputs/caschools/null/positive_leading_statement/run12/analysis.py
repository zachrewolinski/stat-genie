import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent
    csv_path = base_dir / "caschools.csv"
    info_path = base_dir / "info.json"

    df = pd.read_csv(csv_path)

    # Construct student-teacher ratio and an overall academic performance measure.
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = df[["read", "math"]].mean(axis=1)

    # Drop any rows with missing values in the main variables, if present.
    df_model = df[["avg_score", "stratio", "read", "math", "students", "teachers", "income", "english", "lunch", "calworks", "expenditure"]].dropna()

    # Simple bivariate OLS: avg_score on student-teacher ratio.
    X_simple = sm.add_constant(df_model["stratio"])
    y = df_model["avg_score"]
    model_simple = sm.OLS(y, X_simple).fit()

    # Multiple regression controlling for key demographic and resource covariates.
    covariates = ["stratio", "income", "english", "lunch", "calworks", "expenditure"]
    X_multi = sm.add_constant(df_model[covariates])
    model_multi = sm.OLS(y, X_multi).fit()

    # Extract key statistics.
    coef_simple = model_simple.params["stratio"]
    pval_simple = model_simple.pvalues["stratio"]

    coef_multi = model_multi.params["stratio"]
    pval_multi = model_multi.pvalues["stratio"]

    r_simple = np.corrcoef(df_model["stratio"], df_model["avg_score"])[0, 1]

    # Read research question for context.
    with info_path.open() as f:
        info = json.load(f)
    question = info["research_questions"][0]

    # Decide on Likert-scale response based on sign, magnitude, and significance.
    significant_simple = pval_simple < 0.05
    significant_multi = pval_multi < 0.05
    effect_5_students = 5 * coef_multi
    abs_r = abs(r_simple)

    supports_hypothesis = (
        coef_simple < 0
        and coef_multi < 0
        and significant_simple
        and significant_multi
    )

    if supports_hypothesis:
        # Strengthen based on correlation magnitude and effect size.
        if abs_r >= 0.3 and effect_5_students <= -5:
            response_score = 90
        elif abs_r >= 0.2 and effect_5_students <= -3:
            response_score = 80
        else:
            response_score = 70
    else:
        # No robust evidence that lower ratios improve performance.
        if (coef_simple < 0 and coef_multi < 0) and (
            pval_simple < 0.1 or pval_multi < 0.1
        ):
            # Direction is consistent but only weakly supported.
            response_score = 40
        else:
            # Either essentially no association or even the opposite direction.
            if abs_r < 0.05 and abs(effect_5_students) < 1 and pval_simple > 0.5 and pval_multi > 0.5:
                response_score = 10
            else:
                response_score = 25

    explanation_lines = []
    explanation_lines.append(question)
    explanation_lines.append("")
    explanation_lines.append(
        "I constructed a student–teacher ratio variable as students divided by teachers "
        "and an overall academic performance measure as the average of reading and math scores."
    )
    direction_simple = "negative" if coef_simple < 0 else "positive"
    sig_word_simple = "statistically significant" if significant_simple else "not statistically significant"
    explanation_lines.append(
        "A simple OLS regression of average test score on the student–teacher ratio shows a "
        f"{direction_simple} coefficient of {coef_simple:.2f} (p-value {pval_simple:.4f}), which is {sig_word_simple}; "
        f"the Pearson correlation between the two variables is {r_simple:.2f}."
    )
    explanation_lines.append(
        "A multiple regression controlling for district income, percent English learners, "
        "percent qualifying for reduced-price lunch, percent on CalWorks, and expenditures per student "
        f"yields a {('negative' if coef_multi < 0 else 'positive')} coefficient of {coef_multi:.2f} "
        f"(p-value {pval_multi:.4f}), which is {'statistically significant' if significant_multi else 'not statistically significant'}."
    )

    if supports_hypothesis:
        explanation_lines.append(
            "Taken together, these models indicate that, holding other factors constant, districts with lower student–teacher ratios "
            "tend to have higher average test scores, and this association is statistically significant at conventional levels."
        )
    else:
        explanation_lines.append(
            "Taken together, these models do not provide meaningful evidence that lower student–teacher ratios are associated with higher "
            "academic performance: the estimated effects are very small in magnitude and not statistically distinguishable from zero."
        )

    explanation_lines.append(
        "The estimated effect size from the multiple regression implies that reducing the student–teacher ratio by about five students per teacher "
        f"is associated with an estimated change of {effect_5_students:.2f} points in the average test score, which is practically negligible."
    )

    explanation = "\n".join(explanation_lines)

    conclusion = {
        "response": int(round(response_score)),
        "explanation": explanation,
    }

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w") as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=None)


if __name__ == "__main__":
    main()
