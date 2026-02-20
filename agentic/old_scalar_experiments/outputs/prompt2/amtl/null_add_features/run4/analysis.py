import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: str = "amtl.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic cleaning: keep only valid rows
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0)]
    df = df[df["num_amtl"] <= df["sockets"]]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth for grouped binomial regression
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    return df


def fit_binomial_model(df: pd.DataFrame):
    # Binomial regression with logit link on proportions with frequency weights
    # Controls for age, sex (prob_male), and tooth class.
    formula = "prop_missing ~ is_human + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def summarize_effect(df: pd.DataFrame, result):
    # Extract coefficient and inference for human vs non-human
    coef = result.params["is_human"]
    se = result.bse["is_human"]
    p_value = result.pvalues["is_human"]

    # 95% Wald confidence interval on log-odds scale
    z = 1.96
    ci_low = coef - z * se
    ci_high = coef + z * se

    # Convert to odds ratios for interpretability
    import math

    or_est = math.exp(coef)
    or_low = math.exp(ci_low)
    or_high = math.exp(ci_high)

    # Predicted probabilities at representative covariate values
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    common_class = df["tooth_class"].mode().iat[0]

    design_nonhuman = {
        "is_human": 0,
        "age": mean_age,
        "prob_male": mean_prob_male,
        "tooth_class": common_class,
    }
    design_human = {
        "is_human": 1,
        "age": mean_age,
        "prob_male": mean_prob_male,
        "tooth_class": common_class,
    }

    pred_nonhuman = result.predict(pd.DataFrame([design_nonhuman]))[0]
    pred_human = result.predict(pd.DataFrame([design_human]))[0]

    return {
        "coef": float(coef),
        "se": float(se),
        "p_value": float(p_value),
        "ci_logodds": (float(ci_low), float(ci_high)),
        "odds_ratio": float(or_est),
        "ci_or": (float(or_low), float(or_high)),
        "pred_prob_nonhuman": float(pred_nonhuman),
        "pred_prob_human": float(pred_human),
    }


def main():
    df = load_data("amtl.csv")

    # Basic descriptive comparison
    genus_stats = (
        df.assign(rate=lambda x: x["num_amtl"] / x["sockets"])
        .groupby("genus")["rate"]
        .agg(["mean", "count"])
        .reset_index()
    )

    print("AMTL rate by genus (unadjusted):")
    print(genus_stats.to_string(index=False))
    print()

    # Fit regression model
    result = fit_binomial_model(df)
    print(result.summary())

    effect = summarize_effect(df, result)

    print("\nEffect of being human (Homo sapiens) vs non-human primates:")
    print(
        f"  Log-odds coefficient: {effect['coef']:.3f} "
        f"(SE={effect['se']:.3f}, p={effect['p_value']:.3g})"
    )
    print(
        "  Odds ratio (humans vs non-humans): "
        f"{effect['odds_ratio']:.3f} "
        f"(95% CI {effect['ci_or'][0]:.3f}–{effect['ci_or'][1]:.3f})"
    )
    print(
        "  Predicted AMTL probability at mean age/sex and "
        f"{len(df['tooth_class'].unique())} tooth classes "
        f"(using {df['tooth_class'].mode().iat[0]} as representative):"
    )
    print(f"    Non-human primates: {effect['pred_prob_nonhuman']:.3f}")
    print(f"    Humans: {effect['pred_prob_human']:.3f}")


if __name__ == "__main__":
    main()

