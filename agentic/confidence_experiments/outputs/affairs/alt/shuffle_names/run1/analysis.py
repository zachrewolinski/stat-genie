import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # According to info.json, the "age" column actually encodes
    # frequency of extramarital affairs in the past year, and
    # "religiousness" encodes whether there are children in the marriage.
    df["affair_freq"] = df["age"]
    df["has_affair"] = (df["affair_freq"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Drop any rows with missing recodes (if present)
    df = df.dropna(subset=["has_children", "has_affair", "affair_freq"])

    print("Head of recoded data:")
    print(df[["affair_freq", "has_affair", "has_children"]].head(), "\n")

    # Descriptive statistics
    print("Counts by children status:")
    print(df["has_children"].value_counts().rename(index={0: "no_children", 1: "has_children"}), "\n")

    print("Affair frequency by children status (mean, median, std, count):")
    desc = df.groupby("has_children")["affair_freq"].agg(["mean", "median", "std", "count"])
    desc.index = desc.index.map({0: "no_children", 1: "has_children"})
    print(desc, "\n")

    # Binary outcome: any affair vs none
    ct = pd.crosstab(df["has_children"], df["has_affair"])
    ct.index = ct.index.map({0: "no_children", 1: "has_children"})
    ct.columns = ["no_affair", "any_affair"]
    print("Contingency table (children status x any affair):")
    print(ct, "\n")

    # Chi-squared test of independence
    chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
    print(f"Chi-squared test: chi2={chi2:.3f}, df={dof}, p={p_chi2:.4g}")

    # Logistic regression: probability of any affair ~ children (unadjusted)
    model = smf.logit("has_affair ~ has_children", data=df).fit(disp=False)
    coef = model.params["has_children"]
    p_val = model.pvalues["has_children"]
    odds_ratio = float(np.exp(coef))
    ci_low, ci_high = model.conf_int().loc["has_children"]
    or_ci_low, or_ci_high = np.exp(ci_low), np.exp(ci_high)

    print("\nLogistic regression: has_affair ~ has_children")
    print(f"  Coefficient (has_children): {coef:.4f}")
    print(f"  p-value: {p_val:.4g}")
    print(f"  Odds ratio: {odds_ratio:.3f}")
    print(f"  95% CI for OR: [{or_ci_low:.3f}, {or_ci_high:.3f}]")

    # Predicted probabilities for each group
    prob_no_children = float(model.predict({"has_children": [0]})[0])
    prob_has_children = float(model.predict({"has_children": [1]})[0])
    print(f"  Predicted P(any affair | no children): {prob_no_children:.3f}")
    print(f"  Predicted P(any affair | has children): {prob_has_children:.3f}")


if __name__ == "__main__":
    main()

