import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("panda_nuts.csv")

    # Basic cleaning
    df = df.copy()
    df = df[df["seconds"].notna() & df["nuts_opened"].notna()]
    df = df[df["seconds"] > 0]

    # Efficiency: nuts opened per second
    df["efficiency"] = df["nuts_opened"] / df["seconds"]

    # Normalize categories
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().str.strip()
    if "help" in df.columns:
        df["help"] = df["help"].astype(str).str.lower().str.strip()

    # Model: efficiency ~ age + sex + help
    model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()

    # Simple group means for context
    group_means = (
        df.groupby(["sex", "help"], dropna=False)["efficiency"]
        .mean()
        .reset_index()
        .sort_values(["sex", "help"])
    )

    print("Rows used:", len(df))
    print("Efficiency mean:", df["efficiency"].mean())
    print("\nGroup means (efficiency):")
    print(group_means.to_string(index=False))
    print("\nOLS summary:")
    print(model.summary())

    # Extract p-values for the main predictors
    pvals = model.pvalues
    coef = model.params

    result = {
        "age_p": pvals.get("age"),
        "sex_p": pvals.get("C(sex)[T.m]"),
        "help_p": pvals.get("C(help)[T.y]"),
        "age_coef": coef.get("age"),
        "sex_coef": coef.get("C(sex)[T.m]"),
        "help_coef": coef.get("C(help)[T.y]"),
    }

    print("\nKey effects (coef, p-value):")
    for k in ["age", "sex", "help"]:
        print(k, result[f"{k}_coef"], result[f"{k}_p"]) 


if __name__ == "__main__":
    main()
