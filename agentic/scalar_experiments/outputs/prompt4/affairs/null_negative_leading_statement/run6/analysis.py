import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator: any extramarital affair in the past year
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    print("Total observations:", len(df))
    print("\nChildren value counts:")
    print(df["children"].value_counts())

    # Unadjusted difference in affair rates by children status
    print("\nAffair prevalence by children status (unadjusted):")
    print(df.groupby("children")["affair_any"].mean())

    # Encode predictors
    df["children_yes"] = (df["children"] == "yes").astype(int)
    df["gender_male"] = (df["gender"] == "male").astype(int)

    # Unadjusted logistic regression: affair_any ~ children_yes
    try:
        X1 = sm.add_constant(df[["children_yes"]])
        model1 = sm.Logit(df["affair_any"], X1).fit(disp=False)
        coef1 = float(model1.params["children_yes"])
        pval1 = float(model1.pvalues["children_yes"])
        or1 = float(np.exp(coef1))
        print("\nUnadjusted logistic regression (affair_any ~ children_yes):")
        print(f"  log-odds coefficient for children_yes: {coef1:.3f}")
        print(f"  odds ratio for children_yes: {or1:.3f}")
        print(f"  p-value: {pval1:.4f}")
    except Exception as exc:  # pragma: no cover - defensive
        print("\nUnadjusted logistic regression failed:")
        print(repr(exc))
        coef1 = np.nan
        pval1 = np.nan
        or1 = np.nan

    # Adjusted logistic regression with demographic and relationship covariates
    try:
        predictors = [
            "children_yes",
            "gender_male",
            "age",
            "yearsmarried",
            "religiousness",
            "education",
            "occupation",
            "rating",
        ]
        X2 = sm.add_constant(df[predictors])
        model2 = sm.Logit(df["affair_any"], X2).fit(disp=False)
        coef2 = float(model2.params["children_yes"])
        pval2 = float(model2.pvalues["children_yes"])
        or2 = float(np.exp(coef2))
        print("\nAdjusted logistic regression (affair_any ~ children + covariates):")
        print(f"  log-odds coefficient for children_yes: {coef2:.3f}")
        print(f"  adjusted odds ratio for children_yes: {or2:.3f}")
        print(f"  p-value: {pval2:.4f}")
    except Exception as exc:  # pragma: no cover - defensive
        print("\nAdjusted logistic regression failed:")
        print(repr(exc))


if __name__ == "__main__":
    main()

