import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Construct binomial response: missing teeth out of total (missing + present).
    df["total_teeth"] = df["feature3"] + df["feature4"]
    df = df[df["total_teeth"] > 0].copy()
    df["missing_rate"] = df["feature3"] / df["total_teeth"]

    # Indicator for modern humans vs all non-human primates.
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Binomial GLM: missing_rate ~ human + age + sex + tooth class, weighted by total teeth.
    formula = "missing_rate ~ is_human + feature5 + feature7 + C(feature1)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["total_teeth"],
    )
    result = model.fit()

    coef = result.params["is_human"]
    pvalue = result.pvalues["is_human"]
    ci_low, ci_high = result.conf_int().loc["is_human"]
    odds_ratio = float(np.exp(coef))
    ci_low_or = float(np.exp(ci_low))
    ci_high_or = float(np.exp(ci_high))

    print("Human coefficient (log-odds):", coef)
    print("p-value:", pvalue)
    print("Odds ratio (humans vs non-humans):", odds_ratio)
    print("95% CI for odds ratio:", (ci_low_or, ci_high_or))

    # Decide binary answer based on positive, statistically significant human effect.
    if (coef > 0) and (pvalue < 0.05):
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "I fit a binomial regression model for the proportion of missing teeth "
        "per specimen (missing teeth divided by total teeth of that class), "
        "using modern human status (Homo sapiens vs all non-human primates), age, "
        "estimated sex, and tooth class as predictors. The coefficient for modern "
        f"humans on the log-odds scale was {coef:.3f} (odds ratio {odds_ratio:.2f}, "
        f"95% CI {ci_low_or:.2f}–{ci_high_or:.2f}, p-value {pvalue:.3g}). "
        "Based on this model, I concluded that modern humans "
        + ("have higher" if response == "Yes" else "do not have higher")
        + " frequencies of antemortem tooth loss than the non-human primate genera "
        "after accounting for age, sex, and tooth class."
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

