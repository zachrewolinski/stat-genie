import pandas as pd
import statsmodels.api as sm
import numpy as np


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital intercourse in past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Encode children indicator: 1 = children present, 0 = no children
    df["has_children"] = (df["feature6"].str.lower() == "yes").astype(int)

    # Basic descriptive statistics: rate of any affair by children status
    child_group_rates = df.groupby("has_children")["any_affair"].mean()
    child_group_counts = df["has_children"].value_counts().sort_index()

    print("Prevalence of any extramarital affair by children status:")
    for has_children, rate in child_group_rates.items():
        count = child_group_counts.get(has_children, 0)
        label = "children in marriage" if has_children == 1 else "no children in marriage"
        print(f"  {label}: {rate:.3f} (n={count})")

    # Add covariates
    # Gender as binary indicator (1 = male, 0 = female)
    df["is_male"] = (df["feature3"].str.lower() == "male").astype(int)

    covariates = [
        "has_children",
        "is_male",
        "feature4",  # age
        "feature5",  # years married
        "feature7",  # religiousness
        "feature8",  # education
        "feature9",  # occupation
        "feature10",  # marriage rating
    ]

    X = df[covariates]
    X = sm.add_constant(X, prepend=True)
    y = df["any_affair"]

    logit_model = sm.Logit(y, X, missing="drop")
    result = logit_model.fit(disp=False)

    print("\nLogistic regression of any affair on children and covariates")
    print(result.summary())

    params = result.params
    pvalues = result.pvalues
    odds_ratios = params.apply(lambda x: float(np.exp(x)))

    print("\nKey coefficient for having children:")
    print(f"  Coefficient (has_children): {params['has_children']:.4f}")
    print(f"  Odds ratio (has_children): {odds_ratios['has_children']:.3f}")
    print(f"  p-value (has_children): {pvalues['has_children']:.4f}")


if __name__ == "__main__":
    main()
