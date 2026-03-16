import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    print("Number of observations:", len(df))
    print()

    # Descriptive statistics by children status
    mean_affairs = df.groupby("children")["affairs"].mean()
    prop_any = df.groupby("children")["any_affair"].mean()

    print("Mean number of affairs by children status:")
    print(mean_affairs)
    print()

    print("Proportion with any affair by children status:")
    print(prop_any)
    print()

    # Logistic regression: probability of any affair ~ children only
    logit_children = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("Logistic regression with children only:")
    print(logit_children.summary())
    print()

    # Logistic regression with key controls
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    logit_full = smf.logit(formula, data=df).fit(disp=False)
    print("Logistic regression with controls:")
    print(logit_full.summary())
    print()

    # Odds ratios and confidence intervals for children effect in the full model
    params = logit_full.params
    conf_int = logit_full.conf_int()

    or_params = np.exp(params)
    or_conf = np.exp(conf_int)

    print("Odds ratios (95% CI) for children-related terms (full model):")
    mask = [idx for idx in or_params.index if "children" in idx]
    if mask:
        print(
            pd.concat(
                [
                    or_params.loc[mask].rename("odds_ratio"),
                    or_conf.loc[mask].rename(columns={0: "ci_lower", 1: "ci_upper"}),
                ],
                axis=1,
            )
        )
    else:
        print("No children-related terms found in the model.")


if __name__ == "__main__":
    main()

