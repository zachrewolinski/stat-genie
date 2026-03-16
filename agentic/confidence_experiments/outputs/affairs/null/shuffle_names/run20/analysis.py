import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # According to info.json, the column named "age" actually encodes the
    # frequency of extramarital sexual intercourse in the past year:
    # 0 = none, 1 = once, 2 = twice, 3 = 3 times, 7 = 4–10 times,
    # 12 = monthly/weekly/daily.
    # The column "religiousness" is described as:
    # factor. Are there children in the marriage? (yes/no)
    df = df.copy()
    df["affair_freq"] = df["age"]

    # Binary indicator of any extramarital affair.
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    # Children in marriage: religiousness column is actually the children yes/no factor.
    df["has_children"] = (df["religiousness"].astype(str).str.lower() == "yes").astype(int)

    # Drop any rows with missing values in these key fields, if present.
    sub = df[["any_affair", "has_children"]].dropna()

    # Simple descriptive statistics.
    overall_rate = sub["any_affair"].mean()
    rate_children = sub.loc[sub["has_children"] == 1, "any_affair"].mean()
    rate_no_children = sub.loc[sub["has_children"] == 0, "any_affair"].mean()

    # Logistic regression for any affair ~ has_children.
    model = smf.logit("any_affair ~ has_children", data=sub)
    result = model.fit(disp=False)

    coef = result.params["has_children"]
    se = result.bse["has_children"]
    p_value = result.pvalues["has_children"]
    odds_ratio = float(np.exp(coef))

    ci_low, ci_high = np.exp(result.conf_int().loc["has_children"])

    print("N (used in model):", int(result.nobs))
    print("Overall any-affair rate: {:.3f}".format(overall_rate))
    print("Any-affair rate (has children): {:.3f}".format(rate_children))
    print("Any-affair rate (no children): {:.3f}".format(rate_no_children))
    print()
    print("Logistic regression: any_affair ~ has_children")
    print("Coefficient (has_children): {:.4f}".format(coef))
    print("Std. error: {:.4f}".format(se))
    print("p-value: {:.4g}".format(p_value))
    print("Odds ratio (has_children vs none): {:.3f}".format(odds_ratio))
    print("95% CI for odds ratio: ({:.3f}, {:.3f})".format(ci_low, ci_high))


if __name__ == "__main__":
    main()

