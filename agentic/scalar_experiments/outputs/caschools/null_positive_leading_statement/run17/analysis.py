import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_metadata():
    info_path = Path("info.json")
    with info_path.open() as f:
        return json.load(f)


def load_data():
    df = pd.read_csv("caschools.csv")
    # Construct key variables
    df["stratio"] = df["students"] / df["teachers"]
    df["testscore"] = (df["read"] + df["math"]) / 2.0
    return df


def summarize_relationship(df: pd.DataFrame):
    # Correlation between student–teacher ratio and test scores
    corr = df["stratio"].corr(df["testscore"])

    # Simple bivariate regression
    X_simple = sm.add_constant(df["stratio"])
    model_simple = sm.OLS(df["testscore"], X_simple).fit()

    # Regression with common covariates mentioned in the metadata
    covariates = ["income", "english", "lunch", "calworks", "expenditure", "computer"]
    X_controls = sm.add_constant(df[["stratio"] + covariates])
    model_controls = sm.OLS(df["testscore"], X_controls).fit()

    return corr, model_simple, model_controls


def main():
    metadata = load_metadata()
    print("Research question:")
    for q in metadata.get("research_questions", []):
        print("-", q)

    df = load_data()
    print(f"\nNumber of observations: {len(df)}")

    corr, model_simple, model_controls = summarize_relationship(df)

    print("\nCorrelation between student–teacher ratio (stratio) and test score:")
    print(corr)

    print("\nSimple OLS: testscore ~ stratio")
    print(model_simple.summary())

    print("\nOLS with controls: testscore ~ stratio + income + english + lunch + calworks + expenditure + computer")
    print(model_controls.summary())


if __name__ == "__main__":
    main()

