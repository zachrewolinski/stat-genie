import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic sanity filters
    df = df.copy()
    df = df[df["sockets"] > 0]
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    )

    # Indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    return df


def fit_binomial_model(df: pd.DataFrame):
    """
    Fit a binomial regression of AMTL counts on human vs non-human,
    adjusting for age, sex (prob_male), and tooth class.
    """
    # Endog as (successes, failures)
    y = np.column_stack(
        [df["num_amtl"].to_numpy(), (df["sockets"] - df["num_amtl"]).to_numpy()]
    )

    # Design matrix with intercept, human indicator, age, sex proxy, and tooth class
    X = patsy.dmatrix(
        "is_human + age + prob_male + C(tooth_class)",
        data=df,
        return_type="dataframe",
    )

    model = sm.GLM(y, X, family=sm.families.Binomial())
    result = model.fit()
    return result, X.columns.tolist()


def compute_likert_from_effect(coef: float, p_value: float) -> int:
    """
    Map the human-vs-nonhuman coefficient and p-value to a 0–100 Likert score.

    Positive coef -> humans higher AMTL.
    Negative coef -> humans lower AMTL.
    """
    # Default neutral
    score = 50

    if np.isnan(coef) or np.isnan(p_value):
        return 50

    # Strong evidence thresholds
    if p_value < 0.001:
        base = 90
    elif p_value < 0.01:
        base = 80
    elif p_value < 0.05:
        base = 70
    else:
        # Weak or no evidence: keep closer to neutral
        base = 55

    # Effect size scaling via odds ratio
    odds_ratio = float(np.exp(coef))

    # If effect is essentially null, stay near neutral
    if 0.9 <= odds_ratio <= 1.1:
        score = 50 if p_value >= 0.05 else base
    elif odds_ratio > 1.1:
        # Humans have higher AMTL
        magnitude = min(odds_ratio, 3.0)  # cap for stability
        score = min(100, int(base + (magnitude - 1.1) / (3.0 - 1.1) * (100 - base)))
    else:
        # Humans have lower AMTL; map to "No" side of scale
        magnitude = min(1.0 / odds_ratio, 3.0)
        pos_score = min(100, int(base + (magnitude - 1.1) / (3.0 - 1.1) * (100 - base)))
        score = max(0, 100 - pos_score)

    # Bound within 0–100
    score = max(0, min(100, score))
    return int(score)


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)

    # Ensure we actually have both humans and non-humans
    genus_counts = df["genus"].value_counts()
    has_humans = (df["genus"] == "Homo sapiens").any()
    has_nonhumans = (df["genus"] != "Homo sapiens").any()

    if not has_humans or not has_nonhumans:
        response = 50
        explanation = (
            "The dataset does not contain both modern human (Homo sapiens) and "
            "non-human primate genera, so a comparative analysis of AMTL "
            "frequencies is not possible using this data alone. "
            "I therefore cannot provide clear evidence for or against higher "
            "AMTL frequencies in humans relative to non-human primates."
        )
    else:
        result, columns = fit_binomial_model(df)

        # Identify the human coefficient
        human_term = "is_human"
        if human_term not in columns:
            coef = np.nan
            p_value = np.nan
        else:
            idx = columns.index(human_term)
            coef = result.params[idx]
            p_value = result.pvalues[idx]

        odds_ratio = float(np.exp(coef)) if not np.isnan(coef) else float("nan")
        score = compute_likert_from_effect(coef, p_value)

        # Qualitative description of strength
        if p_value < 0.001:
            sig_desc = "highly statistically significant"
        elif p_value < 0.01:
            sig_desc = "strongly statistically significant"
        elif p_value < 0.05:
            sig_desc = "statistically significant"
        else:
            sig_desc = "not statistically significant"

        direction = (
            "higher"
            if odds_ratio > 1
            else "lower"
            if odds_ratio < 1
            else "similar"
        )

        explanation = (
            "I fit a binomial regression model of the number of antemortem tooth losses "
            f"(num_amtl) out of the number of observable sockets for each specimen and tooth class, "
            "using a logit link and treating the response as counts of missing versus present teeth. "
            "The predictors were an indicator for modern humans (Homo sapiens vs. non-human primate genera), "
            "estimated age at death, the probability of being male (prob_male) as a proxy for sex, "
            "and categorical tooth class (Anterior, Posterior, Premolar). "
            f"In this model, the coefficient for the human indicator was {coef:.3f}, "
            f"corresponding to an odds ratio of approximately {odds_ratio:.2f} for AMTL in humans "
            "relative to non-human primates after adjusting for age, sex, and tooth class. "
            f"The associated p-value was {p_value:.3g}, which is {sig_desc}. "
        )

        if p_value < 0.05:
            if odds_ratio > 1:
                explanation += (
                    "This indicates that modern humans have "
                    f"{direction} frequencies of AMTL than non-human primates, "
                    "and the difference is unlikely to be due to sampling variation alone. "
                    "Given the direction and statistical strength of the human effect, "
                    "I interpret this as evidence that humans do indeed have higher AMTL "
                    "frequencies after accounting for age, sex, and tooth class."
                )
            else:
                explanation += (
                    "This suggests that, contrary to the hypothesis, modern humans do not have higher "
                    "AMTL frequencies than non-human primates once age, sex, and tooth class are accounted for. "
                    "The model instead implies equal or lower AMTL odds in humans."
                )
        else:
            explanation += (
                "Because the human effect is not statistically significant at conventional levels, "
                "the data do not provide strong evidence that humans differ from non-human primates in AMTL "
                "frequencies once age, sex, and tooth class are controlled for. "
                "Any observed differences could plausibly arise from sampling variation."
            )

        response = score

    conclusion = {"response": int(response), "explanation": explanation}

    # Write the required JSON-only conclusion file
    out_path = Path("conclusion.txt")
    out_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

