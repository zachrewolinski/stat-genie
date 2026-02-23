import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")
    df = df.rename(
        columns={
            "feature6": "enrollment",
            "feature7": "teachers",
            "feature8": "calworks_pct",
            "feature9": "lunch_pct",
            "feature10": "computers",
            "feature11": "expenditure",
            "feature12": "avg_income",
            "feature13": "english_pct",
            "feature14": "read_score",
            "feature15": "math_score",
        }
    )
    df["stratio"] = df["enrollment"] / df["teachers"]
    df["avg_score"] = df[["read_score", "math_score"]].mean(axis=1)

    # Focus on a more plausible range of class sizes
    subset = df[(df["stratio"] >= 5) & (df["stratio"] <= 40)].copy()

    print("Restricted sample size:", len(subset))
    print("Descriptive statistics (restricted stratio 5-40)")
    print(subset[["stratio", "avg_score"]].describe())
    print()

    print("Correlation (restricted):")
    print(subset[["stratio", "avg_score"]].corr())
    print()

    X = sm.add_constant(subset["stratio"])
    y = subset["avg_score"]
    model = sm.OLS(y, X).fit()
    print("OLS restricted: avg_score ~ stratio (5-40)")
    print(model.summary())
    print()


if __name__ == "__main__":
    main()

