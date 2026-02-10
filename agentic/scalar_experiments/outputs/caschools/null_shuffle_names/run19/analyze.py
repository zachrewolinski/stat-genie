import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Based on the metadata in info.json, the shuffled column names map as:
    # - "english": total enrollment
    # - "students": number of teachers (FTE)
    # - "district": average reading score
    # - "expenditure": average math score
    #
    # Construct student-teacher ratio (students per teacher) and an overall
    # academic performance measure as the average of reading and math scores.
    df = df.copy()
    df["stratio"] = df["english"] / df["students"]
    df["testscr"] = (df["district"] + df["expenditure"]) / 2.0

    # Drop rows with missing or nonsensical ratios.
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna(subset=["stratio", "testscr"])

    # Basic correlation
    corr = df["stratio"].corr(df["testscr"])

    # Simple OLS regression of test scores on class size.
    X = sm.add_constant(df["stratio"])
    model = sm.OLS(df["testscr"], X).fit()

    print("Number of districts:", len(df))
    print("Mean test score:", df["testscr"].mean())
    print("Mean student-teacher ratio:", df["stratio"].mean())
    print("Correlation (stratio, testscr):", corr)
    print("\nOLS results: testscr ~ stratio")
    print(model.summary())


if __name__ == "__main__":
    main()

