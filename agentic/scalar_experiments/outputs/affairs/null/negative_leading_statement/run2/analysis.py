import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for engaging in any extramarital affair.
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"] == "yes").astype(int)

    # Descriptive statistics by children status.
    prop_any_affair = df.groupby("children")["any_affair"].mean()
    mean_affairs = df.groupby("children")["affairs"].mean()
    counts = df["children"].value_counts()

    # Logistic regression for any affair, controlling for covariates.
    formula = (
        "any_affair ~ children_yes + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    model = smf.logit(formula, data=df).fit(disp=False)
    params = model.params
    pvalues = model.pvalues

    children_coef = float(params["children_yes"])
    children_p = float(pvalues["children_yes"])

    # Print a concise summary that the analyst (agent) can interpret.
    print("Descriptive statistics by children status:")
    for group in ["no", "yes"]:
        print(
            f"  children={group}: n={int(counts.get(group, 0))}, "
            f"prop_any_affair={prop_any_affair.get(group, float('nan')):.3f}, "
            f"mean_affairs={mean_affairs.get(group, float('nan')):.3f}"
        )

    print("\nLogistic regression (any_affair ~ ...):")
    print(f"  children_yes coefficient: {children_coef:.4f}")
    print(f"  children_yes p-value:     {children_p:.4g}")
    print(f"  pseudo R-squared:         {model.prsquared:.4f}")


if __name__ == "__main__":
    main()

