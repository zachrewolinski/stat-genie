import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # According to info.json, the column named "age" actually encodes
    # frequency of extramarital affairs, and the column named
    # "religiousness" is a yes/no indicator for whether there are
    # children in the marriage.
    affairs_freq = df["age"]

    # Binary indicator: any extramarital affair in the past year
    df["affair_any"] = (affairs_freq > 0).astype(int)

    # Children indicator from the yes/no column "religiousness"
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop rows with missing mapping, if any
    df = df.dropna(subset=["has_children"])

    # Group-level summaries
    summary = (
        df.groupby("has_children")["affair_any"]
        .agg(["mean", "count"])
        .rename(index={0: "no_children", 1: "has_children"})
    )

    # Logistic regression: probability of any affair ~ has_children
    X = sm.add_constant(df["has_children"])
    y = df["affair_any"]
    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    print("Group-wise probability of any affair:")
    print(summary)
    print("\nLogistic regression: affair_any ~ has_children")
    print(result.summary())


if __name__ == "__main__":
    main()

