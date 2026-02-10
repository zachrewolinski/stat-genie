import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    data_path = pathlib.Path("caschools.csv")
    df = pd.read_csv(data_path)

    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["avg_score"] = (df["read"] + df["math"]) / 2.0

    # Keep only rows with non-missing values on variables of interest
    base_cols = ["avg_score", "stratio"]
    control_candidates = ["income", "lunch", "calworks", "english", "computer", "expenditure"]
    controls = [c for c in control_candidates if c in df.columns]

    used_cols = base_cols + controls
    df_model = df[used_cols].dropna()

    # Simple correlation between student-teacher ratio and performance
    corr = df_model[["stratio", "avg_score"]].corr().loc["stratio", "avg_score"]

    # Linear regression of performance on student-teacher ratio + controls
    X = df_model[["stratio"] + controls]
    X = sm.add_constant(X)
    y = df_model["avg_score"]

    model = sm.OLS(y, X).fit()
    coef = model.params["stratio"]
    se = model.bse["stratio"]
    t_stat = coef / se if se != 0 else 0.0

    # Map evidence to Likert (-100..100) answering:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    # Negative coefficient means higher ratio -> lower scores, i.e. lower ratios help.
    direction = 1.0 if coef < 0 else -1.0 if coef > 0 else 0.0

    # Use t-stat magnitude as strength of evidence measure.
    # Cap at |t|=4 corresponding to |score|=100 and scale linearly below that.
    t_mag = abs(t_stat)
    strength = min(t_mag / 4.0, 1.0) * 100.0
    likert_score = int(round(direction * strength))

    # Basic sanity: if correlation contradicts regression sign strongly, shrink confidence
    if direction != 0.0 and np.sign(corr) == np.sign(coef):
        # correlation and regression agree
        adjusted_score = likert_score
    else:
        # disagreement or weak signal: downweight by half
        adjusted_score = int(round(likert_score * 0.5))

    # Ensure within bounds [-100, 100]
    adjusted_score = max(-100, min(100, adjusted_score))

    # Print a brief analysis summary to stdout
    print("Correlation between student-teacher ratio and average score:", corr)
    print("OLS coefficient on student-teacher ratio:", coef)
    print("t-statistic for student-teacher ratio:", t_stat)
    print("Derived Likert scalar (before bounds):", likert_score)
    print("Final Likert scalar used:", adjusted_score)

    # Write the scalar conclusion to conclusion.txt as required
    conclusion_path = pathlib.Path("conclusion.txt")
    conclusion_path.write_text(f"{adjusted_score}\n")


if __name__ == "__main__":
    main()

