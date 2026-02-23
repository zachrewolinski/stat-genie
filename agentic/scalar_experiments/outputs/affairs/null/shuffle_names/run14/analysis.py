import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    df["had_affair"] = (df["age"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    print("Value counts for affairs frequency code (column 'age'):")
    print(df["age"].value_counts().sort_index(), "\n")

    print("Value counts for children in marriage (column 'religiousness'):")
    print(df["religiousness"].value_counts(dropna=False), "\n")

    crosstab = pd.crosstab(df["has_children"], df["had_affair"])
    crosstab.index = ["no_children", "has_children"]
    crosstab.columns = ["no_affair", "affair"]
    print("Contingency table of children vs any affair:")
    print(crosstab, "\n")

    chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)
    print(f"Chi-squared test: chi2={chi2:.3f}, dof={dof}, p-value={p_value:.4g}\n")

    model_simple = smf.logit("had_affair ~ has_children", data=df).fit(disp=False)
    print("Logistic regression: had_affair ~ has_children")
    print(model_simple.summary(), "\n")

    model_full = smf.logit(
        "had_affair ~ has_children + gender + children + occupation + rating + yearsmarried + affairs",
        data=df,
    ).fit(disp=False)
    print(
        "Logistic regression with controls:"
        " had_affair ~ has_children + gender + children + occupation + rating + yearsmarried + affairs"
    )
    print(model_full.summary())


if __name__ == "__main__":
    main()

