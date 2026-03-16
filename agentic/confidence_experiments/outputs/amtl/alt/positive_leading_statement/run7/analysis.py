import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Create human vs. non-human indicator
    df["is_human"] = df["genus"].astype(str).str.startswith("Homo").astype(int)

    # Proportion of missing teeth for binomial regression
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Binomial regression with logit link, using sockets as binomial trials
    model = smf.glm(
        "amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    coef = model.params["is_human"]
    p_value = model.pvalues["is_human"]
    conf_int = model.conf_int().loc["is_human"].tolist()
    odds_ratio = float(np.exp(coef))

    # Average marginal effect of being human on predicted AMTL probability
    df0 = df.copy()
    df0["is_human"] = 0
    df1 = df.copy()
    df1["is_human"] = 1

    pred0 = model.predict(df0)
    pred1 = model.predict(df1)

    mean_prob_nonhuman = float(pred0.mean())
    mean_prob_human = float(pred1.mean())
    diff_prob = mean_prob_human - mean_prob_nonhuman

    print("Binomial GLM (logit) for AMTL proportion")
    print("Outcome: num_amtl / sockets")
    print("Predictors: is_human, age, prob_male, tooth_class")
    print()
    print(f"is_human coef (log-odds): {coef:.4f}")
    print(f"is_human odds ratio: {odds_ratio:.3f}")
    print(f"is_human 95% CI (log-odds): [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")
    print(f"is_human p-value: {p_value:.4g}")
    print()
    print(f"Avg predicted AMTL prob (non-human): {mean_prob_nonhuman:.4f}")
    print(f"Avg predicted AMTL prob (human):     {mean_prob_human:.4f}")
    print(f"Difference (human - non-human):      {diff_prob:.4f}")


if __name__ == "__main__":
    main()

