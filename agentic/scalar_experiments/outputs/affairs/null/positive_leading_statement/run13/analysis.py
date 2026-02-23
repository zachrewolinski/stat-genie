import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic derived variables
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    print("N observations:", len(df))
    print("\nChildren value counts:")
    print(df["children"].value_counts())

    print("\nAffairs summary overall:")
    print(df["affairs"].describe())

    print("\nAffairs summary by children:")
    print(df.groupby("children")["affairs"].describe())

    # Proportion with any affairs by children
    prop_any = df.groupby("children")["any_affair"].mean()
    print("\nProportion with any affair (>0) by children:")
    print(prop_any)

    # Two-sample t-test on affair counts by children status
    affairs_yes = df.loc[df["children"] == "yes", "affairs"]
    affairs_no = df.loc[df["children"] == "no", "affairs"]

    ttest = stats.ttest_ind(affairs_no, affairs_yes, equal_var=False)
    print("\nT-test (affairs_no vs affairs_yes):")
    print(ttest)

    # Logistic regression: any affair ~ children
    logit1 = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("\nLogistic regression: any_affair ~ C(children)")
    print(logit1.summary())
    print(
        "\nOdds ratio for children=yes vs no:",
        float(np.exp(logit1.params.get("C(children)[T.yes]", np.nan))),
    )

    # Logistic regression with controls
    logit2 = smf.logit(
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "rating + education + occupation",
        data=df,
    ).fit(disp=False)
    print(
        "\nLogistic regression with controls: "
        "any_affair ~ children + age + yearsmarried + religiousness + "
        "rating + education + occupation"
    )
    print(logit2.summary())
    print(
        "\nAdjusted odds ratio for children=yes vs no:",
        float(np.exp(logit2.params.get("C(children)[T.yes]", np.nan))),
    )


if __name__ == "__main__":
    main()

