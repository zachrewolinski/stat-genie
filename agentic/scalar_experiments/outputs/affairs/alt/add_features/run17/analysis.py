import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary outcome: any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Indicator for having children
    df["children_ind"] = df["children"].str.lower().eq("yes").astype(int)

    covariates = [
        "children_ind",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    model_df = df[["any_affair"] + covariates].dropna()
    y = model_df["any_affair"]
    X = sm.add_constant(model_df[covariates])

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print(result.summary())

    # Effect of having children on odds of any affair
    params = result.params
    conf_int = result.conf_int()

    children_coef = params["children_ind"]
    children_p = result.pvalues["children_ind"]
    children_or = float(np.exp(children_coef))
    children_ci_low, children_ci_high = np.exp(conf_int.loc["children_ind"])

    print(
        "\nEffect of having children (logit, outcome = any affair > 0):"
    )
    print(
        f"  Coefficient (log-odds): {children_coef:.4f}\n"
        f"  Odds ratio: {children_or:.3f}\n"
        f"  95% CI for OR: ({children_ci_low:.3f}, {children_ci_high:.3f})\n"
        f"  p-value: {children_p:.4g}"
    )

    prevalence = (
        df.groupby("children")["any_affair"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "prevalence"})
    )
    print("\nPrevalence of any affair by children status:")
    print(prevalence)


if __name__ == "__main__":
    main()

