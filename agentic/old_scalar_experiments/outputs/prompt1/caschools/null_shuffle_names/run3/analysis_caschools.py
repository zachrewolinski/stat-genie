import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # According to info.json:
    # - "english" is total enrollment.
    # - "students" is number of teachers.
    # - "district" is average reading score.
    # - "expenditure" is average math score.
    df = df.copy()
    df["enrollment"] = df["english"]
    df["n_teachers"] = df["students"]
    df["read_score"] = df["district"]
    df["math_score"] = df["expenditure"]

    # Student–teacher ratio and overall test score (average of reading and math).
    df["stratio"] = df["enrollment"] / df["n_teachers"]
    df["testscr"] = (df["read_score"] + df["math_score"]) / 2.0

    # Drop any obviously invalid or missing ratios.
    df = df.replace([pd.NA, pd.NaT], pd.NA)
    df = df[df["n_teachers"] > 0]
    df = df.dropna(subset=["stratio", "testscr"])

    return df


def analyze_relationship(df: pd.DataFrame) -> dict:
    """
    Run a simple linear regression of test scores on student–teacher ratio.
    Returns key statistics for interpretation.
    """
    x = df["stratio"]
    y = df["testscr"]

    # Add constant for OLS.
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    coef_stratio = model.params["stratio"]
    pvalue_stratio = model.pvalues["stratio"]
    r_squared = model.rsquared

    corr = df["stratio"].corr(df["testscr"])

    return {
        "n_obs": int(model.nobs),
        "coef_stratio": float(coef_stratio),
        "pvalue_stratio": float(pvalue_stratio),
        "r_squared": float(r_squared),
        "corr": float(corr),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "caschools.csv"

    df = load_data(csv_path)
    stats = analyze_relationship(df)

    # Print stats for human inspection when needed.
    print("N observations:", stats["n_obs"])
    print("Correlation(stratio, testscr):", stats["corr"])
    print("OLS coef on stratio:", stats["coef_stratio"])
    print("p-value for stratio:", stats["pvalue_stratio"])
    print("R-squared:", stats["r_squared"])

    # Decide binary answer based on sign and significance of association.
    # Lower ratio corresponds to higher performance if:
    # - correlation is negative, and
    # - OLS slope is negative and statistically significant at 5% level.
    negative_corr = stats["corr"] < 0
    negative_slope = stats["coef_stratio"] < 0
    significant = stats["pvalue_stratio"] < 0.05

    if negative_corr and negative_slope and significant:
        response = "Yes"
        explanation = (
            "Using data on 420 California school districts, I computed the student–teacher "
            "ratio as total enrollment divided by the number of teachers and an overall test "
            "score as the average of reading and math scores. "
            f"A simple OLS regression of the overall test score on the student–teacher ratio shows a "
            f"negative and statistically significant association (correlation {stats['corr']:.3f}, "
            f"slope {stats['coef_stratio']:.3f}, p-value {stats['pvalue_stratio']:.3g}, "
            f"R-squared {stats['r_squared']:.3f}), indicating that districts with lower "
            "student–teacher ratios tend to have higher academic performance."
        )
    else:
        response = "No"
        explanation = (
            "Using data on 420 California school districts, I computed the student–teacher "
            "ratio as total enrollment divided by the number of teachers and an overall test "
            "score as the average of reading and math scores. "
            f"In this dataset, the relationship between the student–teacher ratio and test scores is "
            f"very weak and statistically indistinguishable from zero (correlation {stats['corr']:.3f}, "
            f"slope {stats['coef_stratio']:.3f}, p-value {stats['pvalue_stratio']:.3g}, "
            f"R-squared {stats['r_squared']:.3f}), so there is no clear evidence that lower "
            "student–teacher ratios are associated with higher academic performance."
        )

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
