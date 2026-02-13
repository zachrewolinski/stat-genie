import json

import pandas as pd
from scipy import stats
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    df = pd.read_csv("caschools.csv")

    # Based on info.json metadata:
    # - "english" is total enrollment
    # - "students" is number of teachers
    # - "district" is average reading score
    # - "expenditure" is average math score
    df = df.copy()
    df = df[(df["english"] > 0) & (df["students"] > 0)].reset_index(drop=True)

    df["student_teacher_ratio"] = df["english"] / df["students"]
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]
    df["test_score"] = df[["read_score", "math_score"]].mean(axis=1)

    # Drop any remaining missing values in variables of interest
    covariates = ["income", "school", "computer", "rownames", "grades", "county"]
    analysis_cols = ["student_teacher_ratio", "test_score"] + covariates
    df = df.dropna(subset=analysis_cols)

    # Simple correlation between student-teacher ratio and performance
    corr, corr_p = stats.pearsonr(df["student_teacher_ratio"], df["test_score"])

    # Linear regression with controls
    X = df[["student_teacher_ratio"] + covariates]
    X = sm.add_constant(X)
    y = df["test_score"]
    model = sm.OLS(y, X).fit()

    coef = float(model.params["student_teacher_ratio"])
    se = float(model.bse["student_teacher_ratio"])
    pval = float(model.pvalues["student_teacher_ratio"])
    r2 = float(model.rsquared)

    # Decide Yes/No based on direction and significance of association
    if pval < 0.05 and coef < 0:
        response = "Yes"
    else:
        response = "No"

    # Map evidence strength and confidence from p-value
    if pval < 1e-4:
        strength = 90
        confidence = 90
    elif pval < 1e-2:
        strength = 80
        confidence = 85
    elif pval < 0.05:
        strength = 65
        confidence = 75
    else:
        strength = 40
        confidence = 60

    if abs(corr) < 0.05:
        corr_sentence = (
            "The Pearson correlation between student–teacher ratio and average test score was "
            f"{corr:.3f} (p={corr_p:.3g}), which is very close to zero and not statistically "
            "significant, suggesting little linear relationship between the two variables."
        )
    else:
        corr_direction = "negative" if corr < 0 else "positive"
        corr_sentence = (
            "The Pearson correlation between student–teacher ratio and average test score was "
            f"{corr:.3f} (p={corr_p:.3g}), indicating a {corr_direction} linear association between "
            "the two variables."
        )

    if pval < 0.05:
        if coef < 0:
            reg_sentence = (
                "In the regression, the coefficient on student–teacher ratio was "
                f"{coef:.3f} (SE={se:.3f}, p={pval:.3g}, R²={r2:.3f}), indicating that districts with "
                "lower student–teacher ratios tend to have higher average test scores even after "
                "adjusting for these covariates."
            )
        else:
            reg_sentence = (
                "In the regression, the coefficient on student–teacher ratio was "
                f"{coef:.3f} (SE={se:.3f}, p={pval:.3g}, R²={r2:.3f}), indicating that districts with "
                "higher student–teacher ratios tend to have higher average test scores even after "
                "adjusting for these covariates."
            )
    else:
        reg_sentence = (
            "In the regression, the coefficient on student–teacher ratio was "
            f"{coef:.3f} (SE={se:.3f}, p={pval:.3g}, R²={r2:.3f}), which is very small in magnitude "
            "and not statistically significant, so the model does not provide evidence of a meaningful "
            "association between student–teacher ratio and average test scores after adjustment."
        )

    if response == "Yes":
        conclusion_sentence = (
            "Taken together, the descriptive and regression analyses support the conclusion that a "
            "lower student–teacher ratio is associated with higher academic performance in this dataset."
        )
    else:
        conclusion_sentence = (
            "Taken together, the descriptive and regression analyses do not provide evidence that a "
            "lower student–teacher ratio is associated with higher academic performance in this dataset."
        )

    explanation = (
        "Using the caschools.csv data (N={n} districts), I constructed a student–teacher ratio "
        "variable as total enrollment divided by the number of teachers, based on the metadata in "
        "info.json which indicate that the 'english' column is total enrollment and 'students' is "
        "the number of teachers. Academic performance was summarized as the average of the reading "
        "and math scores (the 'district' and 'expenditure' columns). "
        "{corr_sentence} "
        "I then fit a linear regression of average test score on student–teacher ratio, controlling for "
        "district average income, percentages in CalWorks and reduced-price lunch programs, percent English "
        "learners, expenditure per student, and number of computers. "
        "{reg_sentence} "
        "{conclusion_sentence}"
    ).format(
        n=df.shape[0],
        corr_sentence=corr_sentence,
        reg_sentence=reg_sentence,
        conclusion_sentence=conclusion_sentence,
    )

    output = {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()
