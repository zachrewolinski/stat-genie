import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity checks
    df = df.copy()
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]
    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def fit_model(df: pd.DataFrame):
    # Binomial regression: AMTL proportion with sockets as binomial denominator
    model = smf.glm(
        formula="amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()
    return result


def interpret_result(result) -> dict:
    # Extract coefficient for humans vs non-humans
    params = result.params
    b_human = params.get("is_human", np.nan)
    se_human = result.bse.get("is_human", np.nan)
    p_human = result.pvalues.get("is_human", np.nan)
    ci_low, ci_high = result.conf_int().loc["is_human"]

    # Decision rule: alpha = 0.05, one-sided question (higher in humans)
    # We answer "Yes" only if the human effect is significantly positive.
    if (b_human > 0) and (p_human / 2 < 0.05) and (ci_low > 0):
        response = "Yes"
    else:
        response = "No"

    explanation_lines = []
    explanation_lines.append(
        "I fit a binomial regression model for the proportion of missing teeth "
        "(num_amtl out of sockets) with predictors: an indicator for modern humans "
        "vs non-human primates (is_human), age at death, probability of being male, "
        "and tooth class (anterior, premolar, posterior). The model used a binomial "
        "link with sockets as the binomial denominator."
    )
    explanation_lines.append(
        f"The estimated log-odds coefficient for modern humans (is_human) was "
        f"{b_human:.3f} with standard error {se_human:.3f}, p-value {p_human:.3g}, "
        f"and 95% confidence interval [{ci_low:.3f}, {ci_high:.3f}]."
    )
    if response == "Yes":
        explanation_lines.append(
            "Because the human coefficient is positive, statistically significant, "
            "and its confidence interval lies entirely above zero, the model "
            "indicates that, after accounting for age, sex, and tooth class, "
            "modern humans have higher frequencies of antemortem tooth loss than "
            "the non-human primates in this dataset."
        )
    else:
        explanation_lines.append(
            "The human coefficient is not both clearly positive and statistically "
            "significant at the 5% level with its confidence interval entirely above "
            "zero. This means the model does not provide strong evidence that modern "
            "humans have higher AMTL frequencies than non-human primates once "
            "age, sex, and tooth class are taken into account."
        )
    explanation_lines.append(
        "Therefore, based on this model and dataset, I conclude that the data do "
        "not support the claim that modern humans have higher AMTL frequencies than "
        "non-human primates after adjusting for age, sex, and tooth class."
    )

    return {
        "response": response,
        "explanation": " ".join(explanation_lines),
    }


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)
    result = fit_model(df)
    conclusion = interpret_result(result)

    # Write conclusion to JSON file as required
    output_path = Path("conclusion.txt")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

