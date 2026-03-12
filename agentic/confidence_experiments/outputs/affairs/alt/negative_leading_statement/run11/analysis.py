import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics: affair rates by children status
    rates = (
        df.groupby("children")["has_affair"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "affair_rate"})
    )
    print("Affair rates by children status:")
    print(rates)
    print()

    # Logistic regression: probability of any affair vs. covariates
    # Encode categorical variables manually to keep full control over design.
    df_model = df.copy()
    df_model["children_yes"] = (df_model["children"] == "yes").astype(int)
    df_model["female"] = (df_model["gender"] == "female").astype(int)

    X = df_model[
        [
            "children_yes",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
            "female",
        ]
    ]
    X = sm.add_constant(X, has_constant="add")
    y = df_model["has_affair"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Logistic regression results (has_affair ~ children + controls):")
    print(result.summary())
    print()

    # Print effect size and significance for children explicitly
    coef = float(result.params["children_yes"])
    pval = float(result.pvalues["children_yes"])
    odds_ratio = float(np.exp(coef))

    print(f"children_yes coefficient (log-odds): {coef:.4f}")
    print(f"children_yes p-value: {pval:.4g}")
    print(f"children_yes odds ratio: {odds_ratio:.4f}")


if __name__ == "__main__":
    main()
