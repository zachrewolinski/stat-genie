import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children status
    mean_counts = df.groupby("children")["affairs"].mean()
    prop_any = df.groupby("children")["any_affair"].mean()

    print("Mean affair count by children status:")
    print(mean_counts)
    print("\nProportion with any affair by children status:")
    print(prop_any)
    print()

    # Simple logistic regression: any affair on children only
    simple_model = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("Simple logistic regression: any_affair ~ C(children)")
    print(simple_model.summary())
    print()

    # Logistic regression with standard controls used in analyses of this dataset
    formula = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + C(occupation) + rating"
    )
    full_model = smf.logit(formula, data=df).fit(disp=False)

    print("Full logistic regression with controls:")
    print(full_model.summary())
    print()

    # Odds ratios and confidence intervals for easier interpretation
    params = full_model.params
    conf_int = full_model.conf_int()

    odds_ratios = np.exp(params)
    or_ci_lower = np.exp(conf_int[0])
    or_ci_upper = np.exp(conf_int[1])

    print("Odds ratios (full model):")
    for name in odds_ratios.index:
        print(
            f"{name}: OR={odds_ratios[name]:.3f}, "
            f"95% CI=({or_ci_lower[name]:.3f}, {or_ci_upper[name]:.3f})"
        )


if __name__ == "__main__":
    main()

