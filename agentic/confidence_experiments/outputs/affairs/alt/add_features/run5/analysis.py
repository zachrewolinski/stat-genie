import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Create a binary outcome: any affair vs none
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Basic descriptives for the key relationship
    print("=== Descriptive stats: affairs by children ===")
    ct = pd.crosstab(df["children"], df["affair_any"], margins=True)
    print(ct)
    print()

    # Chi-square test of independence for the 2x2 table (excluding margins)
    table = ct.iloc[0:2, 0:2].to_numpy()
    chi2, p_chi2, dof, expected = stats.chi2_contingency(table)
    print("=== Chi-square test: children vs any affair ===")
    print(f"chi2={chi2:.3f}, dof={dof}, p-value={p_chi2:.4f}")
    print("Expected counts:")
    print(expected)
    print()

    # Logistic regression with children as main predictor,
    # adjusting for standard covariates in this dataset.
    # Treat categorical variables with C().
    formula = (
        "affair_any ~ C(children) + C(gender) + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )

    print("=== Fitting logistic regression ===")
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    print(model.summary())

    # Display the odds ratio for having children (yes vs no)
    params = model.params
    conf_int = model.conf_int()
    odds_ratios = params.map(lambda x: float(np.exp(x)))
    print("\n=== Odds ratios ===")
    for name, or_val in odds_ratios.items():
        ci_low = float(np.exp(conf_int.loc[name, 0]))
        ci_high = float(np.exp(conf_int.loc[name, 1]))
        print(f"{name:25s} OR={or_val:6.3f}  95% CI=({ci_low:6.3f}, {ci_high:6.3f})")


if __name__ == "__main__":
    main()
