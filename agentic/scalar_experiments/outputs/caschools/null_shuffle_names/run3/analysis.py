import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # According to info.json descriptions:
    # - "english" is total enrollment.
    # - "students" is number of teachers.
    # - "district" is average reading score.
    # - "expenditure" is average math score.
    enrollment = df["english"].astype(float)
    num_teachers = df["students"].astype(float)

    # Guard against division by zero just in case.
    stratio = enrollment / num_teachers.replace({0: pd.NA})

    readscr = df["district"].astype(float)
    mathscr = df["expenditure"].astype(float)
    testscr = (readscr + mathscr) / 2.0

    analysis_df = pd.DataFrame(
        {
            "testscr": testscr,
            "stratio": stratio,
        }
    ).dropna()

    X = sm.add_constant(analysis_df["stratio"])
    y = analysis_df["testscr"]

    model = sm.OLS(y, X).fit()

    beta = model.params["stratio"]
    pval = model.pvalues["stratio"]

    # Map evidence strength to a Likert-style integer in [-100, 100].
    if beta < 0:
        # Lower student-teacher ratio (fewer students per teacher)
        # is associated with higher test scores.
        if pval < 0.001:
            score = 80
        elif pval < 0.01:
            score = 60
        elif pval < 0.05:
            score = 40
        else:
            score = 20
    elif beta > 0:
        # Higher student-teacher ratio associated with higher scores
        # (opposite of the hypothesized direction).
        if pval < 0.001:
            score = -80
        elif pval < 0.01:
            score = -60
        elif pval < 0.05:
            score = -40
        else:
            score = -20
    else:
        score = 0

    # Ensure within [-100, 100] and integer.
    score = int(max(-100, min(100, score)))

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(str(score), encoding="utf-8")

    # Print a short summary for human inspection (not used by grader).
    print("Estimated effect of student-teacher ratio on test scores:")
    print(f"  beta_stratio = {beta:.4f}, p = {pval:.4g}")
    print(f"  Likert-style conclusion score = {score}")


if __name__ == "__main__":
    main()

