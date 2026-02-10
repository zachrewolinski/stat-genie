import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # student-teacher ratio = total enrollment / number of teachers
    df = df.copy()
    df["stratio"] = df["feature6"] / df["feature7"]

    # use average of reading and math scores as overall performance
    df["avg_score"] = (df["feature14"] + df["feature15"]) / 2.0

    # drop rows with missing or non-finite values
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["stratio", "avg_score"])

    # simple linear regression of avg_score on stratio
    X = sm.add_constant(df["stratio"])
    y = df["avg_score"]
    model = sm.OLS(y, X).fit()

    coef = model.params["stratio"]
    t_value = model.tvalues["stratio"]

    # We expect a negative coefficient if lower ratios (smaller stratio) are beneficial.
    # Map effect strength and significance into a Likert scalar:
    # strong, highly significant negative association -> near +100
    # weak / non-significant -> near 0
    # strong positive -> negative values.
    if coef < 0:
        base = min(1.0, abs(t_value) / 5.0)
        scalar = int(round(40 + 60 * base))
    else:
        base = min(1.0, abs(t_value) / 5.0)
        scalar = int(round(-40 - 60 * base))

    # ensure within [-100, 100]
    scalar = max(-100, min(100, scalar))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

