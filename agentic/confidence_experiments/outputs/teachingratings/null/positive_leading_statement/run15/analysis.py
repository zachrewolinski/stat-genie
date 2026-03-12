import json
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "teachingratings.csv"
OUTPUT_PATH = "conclusion.txt"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Core relationship: beauty vs evaluation score
    r, p = stats.pearsonr(df["beauty"], df["eval"])

    model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

    # Controlled model to account for key instructor/course characteristics
    formula_ctrl = (
        "eval ~ beauty + age + students + "
        "C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)"
    )
    model_ctrl = smf.ols(formula_ctrl, data=df).fit(cov_type="HC3")

    coef_simple = float(model_simple.params["beauty"])
    p_simple = float(model_simple.pvalues["beauty"])
    coef_ctrl = float(model_ctrl.params["beauty"])
    p_ctrl = float(model_ctrl.pvalues["beauty"])

    sd_beauty = float(df["beauty"].std())
    sd_eval = float(df["eval"].std())
    std_beta_ctrl = coef_ctrl * sd_beauty / sd_eval

    # Convert evidence into a Likert-style response
    response = 50
    if coef_ctrl > 0 and p_ctrl < 0.05:
        response = 70
        if p_ctrl < 0.01:
            response += 10
        if abs(std_beta_ctrl) >= 0.2:
            response += 5
        if abs(std_beta_ctrl) >= 0.3:
            response += 5
        response = min(response, 95)
        conclusion_sentence = (
            "Because the controlled association is positive and statistically significant, "
            "the evidence supports a 'Yes' answer that beauty affects teaching productivity "
            "as reflected in student ratings, with a moderate effect size."
        )
    elif coef_ctrl < 0 and p_ctrl < 0.05:
        response = 30
        conclusion_sentence = (
            "Because the controlled association is statistically significant but negative, "
            "the evidence does not support a positive beauty effect on student ratings."
        )
    else:
        response = 20 if abs(std_beta_ctrl) < 0.05 else 40
        conclusion_sentence = (
            "Because the controlled association is not statistically significant and the effect size is near "
            "zero, the evidence does not support a meaningful relationship between beauty and student ratings."
        )

    explanation = (
        "I tested whether instructor beauty is associated with student instructional ratings (eval). "
        f"The Pearson correlation between beauty and eval is r={r:.3f} (p={p:.3g}). "
        f"A simple OLS regression gives a beauty coefficient of {coef_simple:.3f} (p={p_simple:.3g}). "
        "In a controlled model adjusting for age, course size (students), and categorical controls "
        "(gender, minority status, credits, division, native English, tenure), the beauty coefficient is "
        f"{coef_ctrl:.3f} (p={p_ctrl:.3g}), with a standardized effect of {std_beta_ctrl:.3f}. "
        f"{conclusion_sentence}"
    )

    result = {
        "response": int(response),
        "explanation": explanation,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
