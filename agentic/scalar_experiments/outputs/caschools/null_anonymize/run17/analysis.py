import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio: enrollment (feature6) / teachers (feature7)
    df = df.copy()
    df["stratio"] = df["feature6"] / df["feature7"]

    # Academic performance: average of reading and math scores
    df["testscr"] = (df["feature14"] + df["feature15"]) / 2.0

    # Simple linear regression: testscr ~ stratio
    x = sm.add_constant(df["stratio"])
    y = df["testscr"]
    model = sm.OLS(y, x, missing="drop").fit()

    slope = model.params["stratio"]
    pvalue = model.pvalues["stratio"]
    r2 = model.rsquared

    # Map evidence to Likert-style scalar from -100 to 100
    # Negative slope (higher ratio -> lower scores) supports the research question.
    score = 0

    if pvalue < 0.001:
        base = 90
    elif pvalue < 0.01:
        base = 75
    elif pvalue < 0.05:
        base = 60
    elif pvalue < 0.1:
        base = 40
    else:
        base = 10

    # Strengthen if model fit is reasonably high
    if r2 > 0.4:
        base += 10
    elif r2 > 0.2:
        base += 5

    if slope < 0:
        score = base
    elif slope > 0:
        score = -base
    else:
        score = 0

    # Clip to [-100, 100] and round to integer
    score_int = int(max(-100, min(100, round(score))))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(score_int))


if __name__ == "__main__":
    main()

