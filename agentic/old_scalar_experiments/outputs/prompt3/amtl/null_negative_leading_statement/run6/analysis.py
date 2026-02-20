import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic sanity filters for binomial modeling
    df = df.copy()
    # Drop rows with non-positive sockets
    df = df[df["sockets"] > 0]
    # Drop rows where missing teeth exceed observable sockets (data entry issues)
    df = df[df["num_amtl"] <= df["sockets"]]

    # Indicator for modern humans (anything with "Homo" in the genus label)
    df["human"] = df["genus"].str.contains("Homo", case=False, na=False)

    # Proportion of missing teeth for descriptive summaries
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    return df


def fit_binomial_glm(df: pd.DataFrame):
    # Aggregated binomial: successes = num_amtl, failures = sockets - num_amtl
    y = np.asarray(
        np.column_stack(
            [
                df["num_amtl"].to_numpy(),
                (df["sockets"] - df["num_amtl"]).to_numpy(),
            ]
        )
    )

    # Design matrix with intercept, human indicator, and covariates
    # Age and sex (prob_male) are continuous; tooth_class is categorical.
    X = dmatrix(
        "1 + human + age + prob_male + C(tooth_class)",
        df,
        return_type="dataframe",
    )

    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit()
    return res, X


def summarize_and_decide(df: pd.DataFrame, res) -> dict:
    # Descriptive: average AMTL proportion by genus
    genus_summary = df.groupby("genus")["prop_amtl"].agg(
        ["mean", "std", "count"]
    ).sort_values("mean", ascending=False)

    # Effect of being human vs non-human: statsmodels encodes the indicator
    # created from the boolean 'human' column typically as 'human[T.True]'.
    human_param_name = None
    for name in res.params.index:
        if "human" in name:
            human_param_name = name
            break

    if human_param_name is None:
        raise RuntimeError("Human effect not found in model parameters.")

    coef_human = res.params[human_param_name]
    se_human = res.bse[human_param_name]
    pval_human = res.pvalues[human_param_name]

    or_human = float(np.exp(coef_human))
    ci_low = float(np.exp(coef_human - 1.96 * se_human))
    ci_high = float(np.exp(coef_human + 1.96 * se_human))

    # Heuristic decision logic:
    # - If OR is clearly > 1 with CI mostly above 1 and small p-value, answer "Yes"
    # - If OR is around 1 or CI comfortably includes 1 and p-value is large, answer "No"
    #   (i.e., no strong evidence that humans have higher AMTL than non-human primates).
    if (or_human > 1.1) and (ci_low > 1.0) and (pval_human < 0.05):
        response = "Yes"
        strength = 80
        confidence = 80
    elif (or_human < 0.9) and (ci_high < 1.0) and (pval_human < 0.05):
        # Clear evidence that humans have LOWER frequencies than non-human primates
        response = "No"
        strength = 85
        confidence = 85
    else:
        # Inconclusive or marginal: treat as "No" to the specific claim
        # that humans have higher AMTL frequencies after adjustment.
        response = "No"
        strength = 65
        confidence = 70

    explanation = {
        "genus_mean_props": genus_summary.reset_index().to_dict(orient="records"),
        "human_odds_ratio": or_human,
        "human_odds_ratio_ci": [ci_low, ci_high],
        "human_p_value": float(pval_human),
        "model_summary": str(res.summary()),
        "notes": (
            "Binomial GLM of AMTL counts with logit link was fit to the number of "
            "missing teeth (num_amtl) out of observable sockets, with predictors "
            "for modern humans vs non-human genera, age at death, sex estimate "
            "(prob_male), and tooth class (anterior/posterior/premolar). "
            "Rows with data inconsistencies (num_amtl > sockets or sockets <= 0) "
            "were excluded from the analysis."
        ),
    }

    return {
        "response": response,
        "strength": strength,
        "confidence": confidence,
        "explanation": explanation,
    }


def main():
    df = load_and_clean("amtl.csv")

    print("Unique genera and counts:")
    print(df["genus"].value_counts(dropna=False))
    print("\nMean AMTL proportion by genus:")
    print(
        df.groupby("genus")["prop_amtl"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )

    res, _ = fit_binomial_glm(df)
    print("\nModel summary:")
    print(res.summary())

    conclusion = summarize_and_decide(df, res)

    # Write conclusion.json-like object to conclusion.txt as required
    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)

    print("\nConclusion written to conclusion.txt")


if __name__ == "__main__":
    main()
