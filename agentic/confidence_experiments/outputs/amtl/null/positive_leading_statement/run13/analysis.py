import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Binary indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Ensure categorical encoding for tooth class
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Construct aggregated binomial response: [successes, failures]
    successes = df["num_amtl"].to_numpy()
    failures = (df["sockets"] - df["num_amtl"]).to_numpy()
    endog = np.column_stack([successes, failures])

    # Design matrix for predictors
    exog = patsy.dmatrix(
        "is_human + age + prob_male + C(tooth_class)",
        data=df,
        return_type="dataframe",
    )

    model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()

    is_human_coef = model.params["is_human"]
    is_human_pvalue = model.pvalues["is_human"]
    is_human_or = float(np.exp(is_human_coef))
    ci_low, ci_high = np.exp(model.conf_int().loc["is_human"]).tolist()

    print(
        "Binomial regression (aggregated): "
        "AMTL counts ~ is_human + age + prob_male + tooth_class"
    )
    print()
    print(model.summary())
    print()
    print("=== Key effect: modern humans vs non-human primates ===")
    print(f"Coefficient (log-odds) for is_human: {is_human_coef:.4f}")
    print(f"P-value for is_human: {is_human_pvalue:.4g}")
    print(f"Odds ratio for AMTL (humans vs non-humans): {is_human_or:.3f}")
    print(f"95% CI for odds ratio: [{ci_low:.3f}, {ci_high:.3f}]")


if __name__ == "__main__":
    main()
