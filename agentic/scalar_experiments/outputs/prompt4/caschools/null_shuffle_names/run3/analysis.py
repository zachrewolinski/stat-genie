import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Use metadata from info.json to map shuffled column names
    # english -> total enrollment (students)
    # students -> number of teachers
    # district -> average reading score
    # expenditure -> average math score
    # school -> percent qualifying for CalWorks
    # computer -> percent qualifying for reduced-price lunch
    # grades -> expenditure per student
    # income -> district average income (in USD 1,000)
    # rownames -> percent of English learners
    df = df.copy()
    df["students_total"] = df["english"]
    df["teachers_total"] = df["students"]
    df["stratio"] = df["students_total"] / df["teachers_total"]
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["testscr"] = (df["read_score"] + df["math_score"]) / 2.0

    # Socioeconomic and demographic controls
    df["pct_calworks"] = df["school"]
    df["pct_lunch"] = df["computer"]
    df["pct_ell"] = df["rownames"]
    df["exp_per_student"] = df["grades"]

    # Drop rows with missing data in variables of interest (if any)
    subset_cols = [
        "stratio",
        "testscr",
        "income",
        "pct_ell",
        "pct_lunch",
        "pct_calworks",
        "exp_per_student",
    ]
    df_model = df.dropna(subset=subset_cols)

    # Simple correlation between student-teacher ratio and test scores
    corr = float(df_model["stratio"].corr(df_model["testscr"]))

    # Simple linear regression
    model_simple = smf.ols("testscr ~ stratio", data=df_model).fit()

    # Adjusted linear regression with key controls
    formula_adj = (
        "testscr ~ stratio + income + pct_ell + pct_lunch + pct_calworks + exp_per_student"
    )
    model_adj = smf.ols(formula_adj, data=df_model).fit()

    coef_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])
    r2_simple = float(model_simple.rsquared)

    coef_adj = float(model_adj.params["stratio"])
    p_adj = float(model_adj.pvalues["stratio"])
    r2_adj = float(model_adj.rsquared)

    # Translate statistical evidence into a 0-100 Likert-style response
    #  - Strong negative, highly significant association -> strong "Yes"
    #  - Moderate negative association or weaker significance -> moderate "Yes"
    #  - Little or no association -> around 50 (uncertain)
    #  - Strong positive association -> low score (evidence against)
    if coef_adj < 0 and p_adj < 1e-6 and abs(corr) >= 0.3:
        response = 90
    elif coef_adj < 0 and p_adj < 0.001:
        response = 80
    elif coef_adj < 0 and p_adj < 0.01:
        response = 70
    elif coef_adj < 0 and p_adj < 0.05:
        response = 60
    elif coef_adj < 0:
        response = 55
    elif coef_adj > 0 and p_adj < 0.05:
        response = 20
    else:
        response = 50

    direction_word = "lower" if coef_adj < 0 else "higher" if coef_adj > 0 else "no clear"

    explanation = (
        "Research question: Is a lower student-teacher ratio associated with higher academic performance?\n\n"
        "Using the provided California school districts dataset (420 districts), I reconstructed the variables "
        "based on the metadata: total enrollment came from the 'english' column, number of teachers from "
        "the 'students' column, and I defined the student-teacher ratio as enrollment divided by teachers. "
        "Academic performance was measured as the average of the reading and math scores, drawn from the "
        "'district' and 'expenditure' columns, respectively.\n\n"
        f"The simple Pearson correlation between the student-teacher ratio and the average test score was {corr:.3f}, "
        "showing the overall direction and strength of the association across districts. "
        f"A simple linear regression of average test score on the student-teacher ratio produced a coefficient of "
        f"{coef_simple:.3f} (p-value = {p_simple:.3g}, R-squared = {r2_simple:.3f}), indicating how much the mean test "
        "score changes when the number of students per teacher increases by one.\n\n"
        "To account for key confounding factors, I estimated an adjusted regression including district income, "
        "percent of English learners, percent of students in CalWorks, percent on reduced-price lunch, and per-pupil "
        "expenditure. In this adjusted model, the coefficient on the student-teacher ratio was "
        f"{coef_adj:.3f} (p-value = {p_adj:.3g}, R-squared = {r2_adj:.3f}). A negative coefficient means that districts "
        "with fewer students per teacher tend to have higher test scores, even after controlling for these socioeconomic "
        "and demographic differences.\n\n"
        f"Given the sign, magnitude, and statistical significance of the estimated association in both the simple and "
        f"adjusted models, there is {('strong' if response >= 80 else 'moderate' if response >= 60 else 'weak' if response >= 55 else 'little')} "
        f"evidence that {direction_word} student-teacher ratios are linked to better academic performance in this dataset. "
        "However, these results are observational and do not by themselves establish a causal effect of changing class size on "
        "test scores; they only quantify the strength and direction of the association after adjusting for observed covariates."
    )

    conclusion = {"response": int(response), "explanation": explanation}

    # Write conclusion as JSON to conclusion.txt with no extra text
    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

