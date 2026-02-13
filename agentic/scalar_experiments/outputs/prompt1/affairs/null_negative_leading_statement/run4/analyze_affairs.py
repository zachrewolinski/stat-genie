import math

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any affair vs none
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Quick descriptive stats by children
    desc = df.groupby("children")["any_affair"].agg(["mean", "count"])
    print("Affair rate by children (mean of any_affair):")
    print(desc)

    # Logistic regression of any_affair on children, controlling for key covariates
    # children is categorical 'yes'/'no'; treat 'no' as reference via C(children)
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness + "
        "education + occupation + rating + C(gender)"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("\nLogit results:")
    print(model.summary())

    # Extract coefficient for children[T.yes]
    params = model.params
    conf_int = model.conf_int()
    coef = params.get("C(children)[T.yes]")
    ci_low, ci_high = conf_int.loc["C(children)[T.yes]"]

    print("\nEffect of having children (C(children)[T.yes]):")
    print(f"Coef: {coef:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # Also compute odds ratio
    or_children = math.exp(coef)
    or_low = math.exp(ci_low)
    or_high = math.exp(ci_high)
    print(f"Odds ratio: {or_children:.3f}, 95% CI: [{or_low:.3f}, {or_high:.3f}]")


if __name__ == "__main__":
    main()
