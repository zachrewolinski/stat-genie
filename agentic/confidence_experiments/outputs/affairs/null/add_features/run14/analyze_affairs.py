import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair in the past year
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive statistics by children status
    desc_any = df.groupby("children")["had_affair"].agg(["mean", "count"])
    desc_count = df.groupby("children")["affairs"].agg(["mean", "median"])

    print("Proportion with any affair by children:")
    print(desc_any)
    print("\nAffair count distribution by children:")
    print(desc_count)

    # Logistic regression: any affair ~ children only
    logit_children = smf.logit("had_affair ~ C(children)", data=df).fit(disp=False)
    print("\nLogistic regression: had_affair ~ C(children)")
    print(logit_children.summary())

    # Odds ratio for having children vs not
    params = logit_children.params
    odds_ratios = np.exp(params)
    print("\nOdds ratios (children-only model):")
    print(odds_ratios)

    # Logistic regression with standard covariates from the Fair affairs data
    formula_controls = (
        "had_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    logit_controls = smf.logit(formula_controls, data=df).fit(disp=False)
    print("\nLogistic regression with controls:")
    print(logit_controls.summary())

    params_ctrl = logit_controls.params
    odds_ratios_ctrl = np.exp(params_ctrl)
    print("\nOdds ratios (controlled model):")
    print(odds_ratios_ctrl)


if __name__ == "__main__":
    main()

