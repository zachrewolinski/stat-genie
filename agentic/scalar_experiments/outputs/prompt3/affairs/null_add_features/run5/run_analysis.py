import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Basic sanity checks
    print("Rows, columns:", df.shape)
    print("Columns:", list(df.columns))

    # Descriptive stats by children
    grouped = df.groupby("children")["affairs"]
    means = grouped.mean()
    stds = grouped.std()
    counts = grouped.count()

    print("\nMean number of affairs by children:")
    for grp in means.index:
        print(
            f"  children={grp}: mean={means[grp]:.3f}, "
            f"std={stds[grp]:.3f}, n={counts[grp]}"
        )

    # Binary indicator of any affairs
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    prop_any = df.groupby("children")["any_affair"].mean()
    print("\nProportion with any affair in past year by children:")
    for grp in prop_any.index:
        print(f"  children={grp}: proportion={prop_any[grp]:.3f}")

    # Simple logistic regression: any_affair ~ children
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=0)
    print("\nLogit model: any_affair ~ C(children)")
    print(logit_simple.summary())

    # Logistic regression with basic controls
    formula_controls = (
        "any_affair ~ C(children) + age + yearsmarried + "
        "religiousness + education + rating + C(gender)"
    )
    logit_controls = smf.logit(formula_controls, data=df).fit(disp=0)
    print("\nLogit model with controls:")
    print(logit_controls.summary())

    # Print key coefficient and p-value for children effect
    params = logit_controls.params
    pvalues = logit_controls.pvalues
    coef_children = params.get("C(children)[T.yes]", np.nan)
    p_children = pvalues.get("C(children)[T.yes]", np.nan)
    print(
        "\nChildren effect in controlled logit model "
        "(relative to children = no):"
    )
    print(f"  coef = {coef_children:.4f}, p-value = {p_children:.4g}")


if __name__ == "__main__":
    main()

