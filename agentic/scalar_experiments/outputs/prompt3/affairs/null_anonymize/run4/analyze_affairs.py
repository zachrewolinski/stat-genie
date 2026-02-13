import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DATA_PATH = Path("affairs.csv")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def basic_summaries(df: pd.DataFrame) -> None:
    # affair frequency summary by children
    df["has_affair"] = df["feature2"] > 0
    grp = df.groupby("feature6", observed=True)

    print("Counts by children:")
    print(grp["feature1"].count())
    print("\nMean affair frequency (feature2) by children:")
    print(grp["feature2"].mean())
    print("\nProportion with any affair by children:")
    print(grp["has_affair"].mean())


def logistic_regression_affair_any(df: pd.DataFrame) -> None:
    df = df.copy()
    df["has_affair"] = df["feature2"] > 0
    df["children_yes"] = (df["feature6"] == "yes").astype(int)

    # Covariates: gender, age, years married, religiousness, education, occupation, marriage rating
    df["is_male"] = (df["feature3"] == "male").astype(int)

    X = df[
        [
            "children_yes",
            "is_male",
            "feature4",
            "feature5",
            "feature7",
            "feature8",
            "feature9",
            "feature10",
        ]
    ]
    X = sm.add_constant(X)
    y = df["has_affair"].astype(int)

    model = sm.Logit(y, X)
    res = model.fit(disp=False)
    print("\nLogistic regression for any affair (has_affair):")
    print(res.summary())

    # compute odds ratio for having children
    children_coef = res.params["children_yes"]
    children_p = res.pvalues["children_yes"]
    children_or = float(np.exp(children_coef))
    print(f"\nChildren coefficient (log-odds): {children_coef:.4f}")
    print(f"Children odds ratio: {children_or:.4f}")
    print(f"P-value for children effect: {children_p:.4g}")


def main() -> None:
    df = load_data()
    basic_summaries(df)
    logistic_regression_affair_any(df)


if __name__ == "__main__":
    main()

