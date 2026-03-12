import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic summaries by children status
    grouped = df.groupby("children")["affairs"]
    summary = grouped.agg(["mean", "std", "count"])
    any_affair = df.assign(any_affair=(df["affairs"] > 0).astype(int))
    prop_any = any_affair.groupby("children")["any_affair"].mean()

    print("Affairs summary by children:")
    print(summary)
    print("\nProportion with any affair by children:")
    print(prop_any)

    # Logistic regression for any affair ~ children + controls
    formula_logit = "any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)"
    logit_model = smf.logit(formula_logit, data=any_affair).fit(disp=False)
    print("\nLogit results for any_affair:")
    print(logit_model.summary())

    # Poisson regression for affair count ~ children + controls
    formula_pois = "affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)"
    pois_model = smf.poisson(formula_pois, data=df).fit(disp=False)
    print("\nPoisson results for affairs count:")
    print(pois_model.summary())


if __name__ == "__main__":
    main()

