import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "affairs.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    df = df.copy()
    df = df[df["affairs"].notna() & df["children"].notna()]

    # Binary indicator: any affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Group summary by children
    group = df.groupby("children")
    summary = group["affairs"].agg(["count", "mean", "median"])
    summary["prop_any_affair"] = group["has_affair"].mean()

    print("Group summary by children (affairs levels):")
    print(summary)
    print()

    # Regression controls
    controls = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender"]
    model_df = df.dropna(subset=controls + ["children", "has_affair", "affairs"]).copy()

    # Logistic regression: any affair
    logit_model = smf.logit(
        "has_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)",
        data=model_df,
    ).fit(disp=False)

    print("Logit model (any affair):")
    print(logit_model.summary())
    print()

    # OLS on affairs count (for effect direction; not causal)
    ols_model = smf.ols(
        "affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)",
        data=model_df,
    ).fit()

    print("OLS model (affairs count):")
    print(ols_model.summary())


if __name__ == "__main__":
    main()
