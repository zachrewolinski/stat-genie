import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for any extramarital affair in past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    print("Basic sample information")
    print("------------------------")
    print(f"Number of observations: {len(df)}")
    print("\nChildren distribution:")
    print(df["children"].value_counts())

    # Group comparisons by children status
    print("\nAffair counts by children status")
    print("--------------------------------")
    group_counts = df.groupby("children")["affairs"].agg(["mean", "median"])
    print(group_counts)

    print("\nProbability of any affair by children status")
    print("-------------------------------------------")
    prop_any = df.groupby("children")["affair_any"].mean()
    print(prop_any)

    # Logistic regression for any affair, controlling for other factors
    print("\nLogistic regression: any affair ~ children + controls")
    print("-----------------------------------------------------")
    formula = (
        "affair_any ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=0)
    print(logit_model.summary())

    # Odds ratios for easier interpretation
    print("\nOdds ratios (exp(coefficients)) with 95% CI")
    print("-------------------------------------------")
    params = logit_model.params
    conf = logit_model.conf_int()
    odds_ratios = np.exp(params)
    conf_or = np.exp(conf)
    or_table = pd.DataFrame(
        {
            "odds_ratio": odds_ratios,
            "ci_lower": conf_or[0],
            "ci_upper": conf_or[1],
        }
    )
    print(or_table)

    # Poisson regression for affair counts, as an alternative check
    print("\nPoisson regression: affair count ~ children + controls")
    print("-------------------------------------------------------")
    pois_formula = (
        "affairs ~ C(children) + age + yearsmarried + "
        "religiousness + education + occupation + rating + C(gender)"
    )
    pois_model = smf.poisson(pois_formula, data=df).fit(disp=0)
    print(pois_model.summary())


if __name__ == "__main__":
    main()
