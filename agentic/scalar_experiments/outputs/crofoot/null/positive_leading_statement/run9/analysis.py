import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Construct relative group size and contest location metrics
    df["rel_size"] = (df["n_focal"] - df["n_other"]) / (df["n_focal"] + df["n_other"])
    # Positive loc_diff => focal group is closer to the center of its home range
    df["loc_diff"] = df["dist_other"] - df["dist_focal"]
    df["focal_larger"] = (df["n_focal"] > df["n_other"]).astype(int)
    df["focal_home"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    print("Summary of key predictors:")
    print(df[["win", "rel_size", "loc_diff"]].describe())
    print()

    # Logistic regression: probability of focal win as a function of relative size and location
    print("Logistic regression without interaction:")
    model = smf.logit("win ~ rel_size + loc_diff", data=df).fit(disp=False)
    print(model.summary())

    # Also check a model with the interaction between relative size and location
    model_int = smf.logit("win ~ rel_size * loc_diff", data=df).fit(disp=False)
    print()
    print("Model with interaction:")
    print(model_int.summary())

    # Simple 2x2 tables and chi-square tests
    print()
    print("Contingency: focal larger vs win")
    table_size = pd.crosstab(df["focal_larger"], df["win"])
    print(table_size)
    chi2, p, _, _ = stats.chi2_contingency(table_size)
    print(f"Chi-square p-value (size): {p:.4f}")

    print()
    print("Contingency: focal home vs win")
    table_home = pd.crosstab(df["focal_home"], df["win"])
    print(table_home)
    chi2, p, _, _ = stats.chi2_contingency(table_home)
    print(f"Chi-square p-value (home): {p:.4f}")


if __name__ == "__main__":
    main()

