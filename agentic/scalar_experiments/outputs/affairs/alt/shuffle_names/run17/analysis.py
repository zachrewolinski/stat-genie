import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # According to the metadata in info.json, the column named "age"
    # actually encodes the frequency of extramarital intercourse in
    # the past year (0 = none, >0 = some affairs).
    df["has_affair"] = (df["age"] > 0).astype(int)

    # The column named "religiousness" is described as a factor
    # indicating whether there are children in the marriage ("yes"/"no").
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Basic descriptives: affair rates with vs without children
    print("Counts of has_children (1=yes, 0=no):")
    print(df["has_children"].value_counts(dropna=False))
    print()

    overall_rate = df["has_affair"].mean()
    print(f"Overall affair prevalence (any vs none): {overall_rate:.3f}")
    print()

    tab = pd.crosstab(df["has_children"], df["has_affair"], normalize="index")
    print("Affair rate by children status (row-normalized):")
    print(tab)
    print()

    # Unadjusted logistic regression: has_affair ~ has_children
    print("Logistic regression: has_affair ~ has_children")
    model1 = smf.logit("has_affair ~ has_children", data=df).fit(disp=False)
    print(model1.summary())
    print()

    # Adjusted model adding other available covariates from the dataset.
    # Column names are somewhat shuffled relative to their semantic
    # meaning, but this still serves to check robustness of the
    # children effect when controlling for other factors.
    formula_adj = (
        "has_affair ~ has_children + children + affairs + "
        "C(gender) + occupation + yearsmarried + rating"
    )
    print("Adjusted logistic regression:")
    print(f"Formula: {formula_adj}")
    model2 = smf.logit(formula_adj, data=df).fit(disp=False)
    print(model2.summary())
    print()

    # Extract odds ratio and 95% CI for the children effect
    params = model2.params
    conf_int = model2.conf_int()
    if "has_children" in params.index:
        log_odds = params["has_children"]
        ci_low, ci_high = conf_int.loc["has_children"]
        or_est = np.exp(log_odds)
        or_low, or_high = np.exp([ci_low, ci_high])
        print("Effect of having children (adjusted model):")
        print(f"  log-odds coef = {log_odds:.3f}")
        print(f"  95% CI (log-odds): [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  odds ratio = {or_est:.3f}")
        print(f"  95% CI (OR): [{or_low:.3f}, {or_high:.3f}]")


if __name__ == "__main__":
    main()

