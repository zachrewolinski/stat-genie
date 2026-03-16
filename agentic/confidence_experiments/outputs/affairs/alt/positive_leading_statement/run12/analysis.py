import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic recoding
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Children as categorical (yes/no)
    # Summary statistics by children
    summary_affairs = df.groupby("children")["affairs"].agg(["mean", "std", "count"])
    summary_any = df.groupby("children")["has_affair"].mean()

    print("Mean number of affairs by children:")
    print(summary_affairs)
    print("\nProportion with any affair by children:")
    print(summary_any)

    # Linear regression for affair count
    formula_ols = (
        "affairs ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    model_ols = smf.ols(formula_ols, data=df).fit()
    print("\nOLS regression on affair counts:")
    print(model_ols.summary())

    # Logistic regression for any affair
    formula_logit = (
        "has_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    model_logit = smf.logit(formula_logit, data=df).fit(disp=0)
    print("\nLogistic regression on any affair (has_affair):")
    print(model_logit.summary())


if __name__ == "__main__":
    main()

