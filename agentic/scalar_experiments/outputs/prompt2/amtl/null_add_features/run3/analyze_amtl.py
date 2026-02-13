import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    genera_of_interest = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(genera_of_interest)].copy()

    df = df[
        (df["sockets"] > 0)
        & (df["num_amtl"] >= 0)
        & (df["num_amtl"] <= df["sockets"])
    ].copy()

    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    )

    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    print(f"Rows used in model: {len(df)}")

    total_missing_h = df.loc[df["is_human"] == 1, "num_amtl"].sum()
    total_sockets_h = df.loc[df["is_human"] == 1, "sockets"].sum()
    rate_h = total_missing_h / total_sockets_h

    total_missing_nh = df.loc[df["is_human"] == 0, "num_amtl"].sum()
    total_sockets_nh = df.loc[df["is_human"] == 0, "sockets"].sum()
    rate_nh = total_missing_nh / total_sockets_nh

    print(f"Observed AMTL rate humans: {rate_h:.3f}")
    print(f"Observed AMTL rate non-humans: {rate_nh:.3f}")

    formula = "prop_missing ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print(result.summary())

    coef = result.params["is_human"]
    pval = result.pvalues["is_human"]

    df_h = df.copy()
    df_h["is_human"] = 1
    pred_h = result.predict(df_h).mean()

    df_nh = df.copy()
    df_nh["is_human"] = 0
    pred_nh = result.predict(df_nh).mean()

    print(f"Average predicted AMTL prob if human: {pred_h:.3f}")
    print(f"Average predicted AMTL prob if non-human: {pred_nh:.3f}")

    response = "Yes" if coef > 0 else "No"

    if pval < 1e-4:
        base_conf = 95
    elif pval < 1e-3:
        base_conf = 93
    elif pval < 1e-2:
        base_conf = 90
    elif pval < 5e-2:
        base_conf = 80
    elif pval < 1e-1:
        base_conf = 70
    else:
        base_conf = 55

    abs_coef = abs(coef)
    if abs_coef < 0.05:
        base_conf -= 15
    elif abs_coef < 0.1:
        base_conf -= 5

    confidence = int(max(0, min(100, round(base_conf))))

    explanation = (
        "Using a binomial logistic regression of the proportion of missing teeth "
        "(num_amtl out of sockets) on a binary indicator for humans versus non-human primates, "
        "controlling for age, probability of being male, and tooth class, humans had an observed "
        f"AMTL rate of {rate_h:.3f} compared with {rate_nh:.3f} in non-human primates, and an adjusted "
        f"average predicted AMTL probability of {pred_h:.3f} versus {pred_nh:.3f}; the human indicator "
        f"had coefficient {coef:.3f} (p={pval:.3g}), supporting the conclusion that humans "
        f"{'do' if response == 'Yes' else 'do not'} have higher AMTL frequencies than the non-human genera "
        "after accounting for these covariates."
    )

    conclusion = {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    with out_path.open("w") as f:
        json.dump(conclusion, f)

    print("Conclusion written to conclusion.txt")


if __name__ == "__main__":
    main()

