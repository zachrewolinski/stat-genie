import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic sample description
    print("N observations:", len(df))
    print("\nChildren value counts:")
    print(df["children"].value_counts(dropna=False))

    # Binary indicator for any extramarital affairs in the past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    print("\nMean number of affairs by children:")
    print(df.groupby("children")["affairs"].mean())

    print("\nProportion with any affairs by children:")
    print(df.groupby("children")["any_affair"].mean())

    # Logistic regression for probability of any affair
    logit_formula = (
        "any_affair ~ C(children) + gender + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)
    print("\nLogit results for any_affair ~ children + controls")
    print(logit_model.summary())

    # Poisson regression for counts of affairs (including zeros)
    poisson_formula = (
        "affairs ~ C(children) + gender + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )
    poisson_model = smf.glm(
        poisson_formula,
        data=df,
        family=sm.families.Poisson(),
    ).fit()
    print("\nPoisson GLM results for affairs ~ children + controls")
    print(poisson_model.summary())

    # Isolate children coefficients for quick inspection
    print("\nLogit children coefficients:")
    print(logit_model.params.filter(like="C(children)"))
    print(logit_model.pvalues.filter(like="C(children)"))

    print("\nPoisson children coefficients:")
    print(poisson_model.params.filter(like="C(children)"))
    print(poisson_model.pvalues.filter(like="C(children)"))


if __name__ == "__main__":
    main()

