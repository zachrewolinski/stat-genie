import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "amtl.csv"

    df = pd.read_csv(data_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature7": "sex_est",
            "feature8": "genus",
        }
    )

    # Basic cleaning: ensure valid counts and required columns present
    df = df[
        (df["sockets"] > 0)
        & df["missing"].ge(0)
        & df["missing"].le(df["sockets"])
    ].copy()
    df = df.dropna(
        subset=["missing", "sockets", "age", "sex_est", "tooth_class", "genus"]
    )

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["prop_missing"] = df["missing"] / df["sockets"]

    # Binomial regression: missing proportion with sockets as binomial denominator
    model = smf.glm(
        formula="prop_missing ~ is_human + age + sex_est + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    # Extract effect for humans vs non-human primates
    coef = float(model.params["is_human"])
    pval = float(model.pvalues["is_human"])
    ci_low, ci_high = model.conf_int().loc["is_human"].astype(float)

    # Predicted AMTL probabilities for humans vs non-humans,
    # averaging over observed age, sex, and tooth-class distributions.
    base_covariates = df[["age", "sex_est", "tooth_class"]].copy()
    human_design = base_covariates.copy()
    human_design["is_human"] = 1
    nonhuman_design = base_covariates.copy()
    nonhuman_design["is_human"] = 0

    pred_human = float(model.predict(human_design).mean())
    pred_nonhuman = float(model.predict(nonhuman_design).mean())

    # Descriptive genus-level AMTL rates
    df["rate"] = df["prop_missing"]
    genus_summary = (
        df.groupby("genus")["rate"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )

    # Determine binary answer
    if coef > 0 and pval < 0.05:
        response = "Yes"
        direction_statement = (
            "The human indicator has a positive and statistically significant "
            "log-odds coefficient (p < 0.05), indicating higher AMTL frequencies "
            "for modern humans than for non-human primates after adjusting for "
            "age, sex, and tooth class."
        )
    else:
        response = "No"
        if coef > 0:
            direction_statement = (
                "Although the human indicator coefficient is positive, it is not "
                "statistically distinguishable from zero at conventional levels, "
                "so the model does not provide strong evidence that modern humans "
                "have higher AMTL frequencies than non-human primates once age, "
                "sex, and tooth class are taken into account."
            )
        else:
            direction_statement = (
                "The human indicator coefficient is not positive and statistically "
                "significant; instead, the fitted model suggests that modern humans "
                "do not have higher AMTL frequencies (and may even have similar or "
                "lower frequencies) compared with non-human primates after "
                "adjusting for age, sex, and tooth class."
            )

    # Build explanation tying model and descriptive patterns together
    genus_lines = []
    for genus, row in genus_summary.iterrows():
        genus_lines.append(
            f"{genus}: mean AMTL proportion {row['mean']:.3f} "
            f"across {int(row['count'])} tooth-class observations"
        )
    genus_text = "; ".join(genus_lines)

    explanation = (
        "I analyzed the AMTL dataset of 1,450 tooth-class observations using a "
        "binomial logistic regression model. The response variable was the "
        "proportion of missing teeth (number of missing teeth divided by the "
        "number of observable sockets), with the number of sockets used as the "
        "binomial denominator. The predictors were a binary indicator for modern "
        "humans versus non-human primates, estimated age at death, estimated sex, "
        "and tooth class (anterior, premolar, posterior) treated as a categorical "
        "factor. "
        f"In this model, the coefficient for the modern human indicator is "
        f"{coef:.3f} with a 95% confidence interval from {ci_low:.3f} to "
        f"{ci_high:.3f} (p = {pval:.3g}). "
        f"Average predicted AMTL probabilities from the fitted model are "
        f"{pred_human:.3f} for modern humans and {pred_nonhuman:.3f} for "
        "non-human primates when averaging over the observed distributions of "
        "age, sex, and tooth class. "
        f"Descriptively, mean AMTL proportions by genus are: {genus_text}. "
        + direction_statement
    )

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = base_path / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

