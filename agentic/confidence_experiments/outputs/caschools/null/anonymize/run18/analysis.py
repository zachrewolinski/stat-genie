import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Compute student-teacher ratio and combined test score
    df["student_teacher_ratio"] = df["feature6"] / df["feature7"]
    df["test_score"] = (df["feature14"] + df["feature15"]) / 2.0
    return df


def simple_association(df: pd.DataFrame):
    x = df["student_teacher_ratio"]
    y = df["test_score"]

    # Correlation
    corr = x.corr(y)

    # Simple linear regression: test_score ~ student_teacher_ratio
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    coef = model.params["student_teacher_ratio"]
    p_value = model.pvalues["student_teacher_ratio"]
    r_squared = model.rsquared

    return {
        "corr": float(corr),
        "coef": float(coef),
        "p_value": float(p_value),
        "r_squared": float(r_squared),
        "n": int(model.nobs),
    }


def adjusted_association(df: pd.DataFrame):
    """
    Multiple regression controlling for key demographics and resources.
    test_score ~ student_teacher_ratio + income + expenditure + english + lunch + calworks
    """
    y = df["test_score"]
    predictors = [
        "student_teacher_ratio",
        "feature12",  # income
        "feature11",  # expenditure per student
        "feature13",  # percent English learners
        "feature9",  # percent reduced-price lunch
        "feature8",  # percent CalWorks
    ]

    X = df[predictors].copy()
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    coef = model.params["student_teacher_ratio"]
    p_value = model.pvalues["student_teacher_ratio"]
    r_squared = model.rsquared

    return {
        "coef": float(coef),
        "p_value": float(p_value),
        "r_squared": float(r_squared),
        "n": int(model.nobs),
    }


def choose_likert(simple_res, adjusted_res) -> int:
    """
    Map evidence strength to a 0–100 Likert score,
    where higher means stronger evidence that lower ratios
    are associated with higher performance (i.e., a negative coefficient).
    """
    coef_simple = simple_res["coef"]
    p_simple = simple_res["p_value"]
    r_simple = simple_res["r_squared"]

    coef_adj = adjusted_res["coef"]
    p_adj = adjusted_res["p_value"]
    r_adj = adjusted_res["r_squared"]

    # We require the relationship to be negative and statistically significant
    # in at least the simple model.
    strong_negative = coef_simple < 0 and coef_adj < 0
    sig_simple = p_simple < 0.01
    sig_adj = p_adj < 0.05

    # Rough effect size via R^2 from simple model
    if strong_negative and sig_simple and sig_adj and r_simple >= 0.15:
        return 90
    if strong_negative and sig_simple and sig_adj:
        return 80
    if strong_negative and sig_simple:
        return 70
    if strong_negative and (p_simple < 0.1):
        return 60

    # Weak or no consistent evidence
    if coef_simple < 0 or coef_adj < 0:
        return 40
    return 20


def build_explanation(simple_res, adjusted_res, likert: int) -> str:
    lines = []
    lines.append(
        "I examined whether districts with lower student–teacher ratios "
        "tend to have higher academic performance in the California K–6/K–8 data."
    )
    lines.append(
        "Student–teacher ratio was computed as total enrollment (feature6) divided "
        "by the number of teachers (feature7), and academic performance was measured "
        "as the average of the district reading and math scores "
        "( (feature14 + feature15) / 2 )."
    )

    # Simple association summary
    lines.append(
        f"In a simple linear regression of average test score on the student–teacher ratio "
        f"(n={simple_res['n']} districts), the Pearson correlation was {simple_res['corr']:.3f}, "
        f"the slope coefficient was {simple_res['coef']:.3f} points per additional student per teacher, "
        f"with p-value {simple_res['p_value']:.3g} and R-squared {simple_res['r_squared']:.3f}."
    )

    # Adjusted association summary
    lines.append(
        "I then fit a multiple regression that controlled for district income (feature12), "
        "expenditure per student (feature11), percentage of English learners (feature13), "
        "percentage qualifying for reduced-price lunch (feature9), and percentage on CalWorks (feature8). "
        f"In this adjusted model, the coefficient on the student–teacher ratio was "
        f"{adjusted_res['coef']:.3f} with p-value {adjusted_res['p_value']:.3g} "
        f"and model R-squared {adjusted_res['r_squared']:.3f} (n={adjusted_res['n']})."
    )

    if likert >= 80:
        conclusion = (
            "Both the simple and adjusted models show a statistically significant "
            "negative association: districts with more students per teacher tend to "
            "have lower average test scores, even after accounting for key demographic "
            "and resource variables. This provides strong evidence in this dataset that "
            "lower student–teacher ratios are associated with higher academic performance."
        )
    elif likert >= 60:
        conclusion = (
            "The relationship is negative and statistically significant in at least the simple model, "
            "suggesting that districts with fewer students per teacher tend to perform somewhat better, "
            "though the effect size and robustness are more moderate."
        )
    elif likert >= 40:
        conclusion = (
            "The estimated relationship is generally negative but only weakly supported statistically, "
            "so the evidence for a meaningful association between lower student–teacher ratios and higher "
            "performance is limited in this dataset."
        )
    else:
        conclusion = (
            "The analyses do not show consistent or statistically reliable evidence that lower "
            "student–teacher ratios are associated with higher academic performance in this dataset."
        )

    lines.append(conclusion)

    lines.append(
        f"On a 0–100 scale summarizing how strongly the data support a 'Yes' answer "
        f"to the research question, I assign a score of {likert}."
    )

    return " ".join(lines)


def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "caschools.csv"

    df = load_data(csv_path)

    simple_res = simple_association(df)
    adjusted_res = adjusted_association(df)
    likert = choose_likert(simple_res, adjusted_res)
    explanation = build_explanation(simple_res, adjusted_res, likert)

    output = {"response": int(likert), "explanation": explanation}

    out_path = base_dir / "conclusion.txt"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

