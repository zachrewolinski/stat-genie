import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data():
    data_path = Path("amtl.csv")
    info_path = Path("info.json")

    df = pd.read_csv(data_path)
    with info_path.open() as f:
        info = json.load(f)
    return df, info


def infer_missing_and_sockets(df: pd.DataFrame):
    """
    Infer which columns correspond to the number of missing teeth (amtl_count)
    and the number of observable sockets (sockets_total).
    """
    # Interpretation A: genus = missing, age = sockets
    missing_a = df["genus"]
    sockets_a = df["age"]
    invalid_a = (missing_a > sockets_a).sum()

    # Interpretation B: age = missing, genus = sockets
    missing_b = df["age"]
    sockets_b = df["genus"]
    invalid_b = (missing_b > sockets_b).sum()

    if invalid_a == 0 and invalid_b > 0:
        missing_col, sockets_col = "genus", "age"
    elif invalid_b == 0 and invalid_a > 0:
        missing_col, sockets_col = "age", "genus"
    else:
        # Fall back to the metadata description if both have violations or both are clean.
        # Metadata states that "genus" is the number missing and "age" is the number of sockets.
        missing_col, sockets_col = "genus", "age"

    return missing_col, sockets_col, invalid_a, invalid_b


def fit_binomial_model(df: pd.DataFrame, missing_col: str, sockets_col: str):
    # Drop any rows with non-positive socket counts
    df = df.copy()
    df = df[df[sockets_col] > 0].copy()

    # Define response as a 2-column array: [successes, failures]
    successes = df[missing_col].astype(float)
    failures = (df[sockets_col] - df[missing_col]).astype(float)

    # Guard against impossible values
    valid = (successes >= 0) & (failures >= 0)
    df = df[valid].copy()
    successes = successes[valid]
    failures = failures[valid]

    endog = np.column_stack([successes, failures])

    # Covariates
    df["is_human"] = (df["tooth_class"] == "Homo sapiens").astype(int)
    df["tooth_class_cat"] = df["sockets"].astype("category")

    # Age at death and sex proxy
    df["age_at_death"] = df["pop"].astype(float)
    df["sex_code"] = df["stdev_age"].astype(float)

    # Binomial GLM with logit link
    formula = "endog ~ is_human + age_at_death + sex_code + tooth_class_cat"
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=None,
    )
    result = model.fit()
    return result


def map_to_likert(coef: float, p_value: float) -> int:
    """
    Map the human effect size and significance to a 0-100 Likert scale.
    """
    if p_value >= 0.05:
        # No strong evidence for a difference.
        # Center around 50 with mild adjustment by effect direction.
        base = 50
        adjustment = max(-15, min(15, coef * 5))
        return int(round(base + adjustment))

    # Statistically significant effect: scale by magnitude.
    # Clamp coefficient to a reasonable range to avoid extreme values.
    clipped = max(-2.5, min(2.5, coef))
    # Positive coefficients (higher AMTL for humans) map above 50,
    # negative coefficients map below 50.
    score = 50 + (clipped / 2.5) * 50
    return int(round(max(0, min(100, score))))


def build_explanation(info, result, human_coef_name: str, missing_col: str, sockets_col: str,
                      invalid_a: int, invalid_b: int) -> str:
    rq = info.get("research_questions", [""])[0]
    coef = result.params[human_coef_name]
    pval = result.pvalues[human_coef_name]

    # Compute approximate average AMTL frequency for humans vs non-humans
    df, _ = load_data()
    df["is_human"] = (df["tooth_class"] == "Homo sapiens").astype(int)
    df["missing"] = df[missing_col].astype(float)
    df["sockets_total"] = df[sockets_col].astype(float)
    valid = (df["sockets_total"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["sockets_total"])
    df = df[valid]

    human = df[df["is_human"] == 1]
    nonhuman = df[df["is_human"] == 0]

    human_rate = (human["missing"].sum() / human["sockets_total"].sum()) if len(human) > 0 else np.nan
    nonhuman_rate = (nonhuman["missing"].sum() / nonhuman["sockets_total"].sum()) if len(nonhuman) > 0 else np.nan

    explanation = []
    explanation.append(rq)
    explanation.append("")
    explanation.append(
        f"I modeled the proportion of sockets missing (AMTL) using a binomial regression "
        f"with a logit link, where each row contributed the number of missing teeth and "
        f"the number of observable sockets for a given specimen and tooth class."
    )
    explanation.append(
        f"The response used `{missing_col}` as the count of missing teeth and `{sockets_col}` "
        f"as the total number of observable sockets. Under these interpretations, the number of "
        f"rows with impossible counts (missing > sockets) was {invalid_a} when assuming "
        f"genus = missing, age = sockets and {invalid_b} when assuming "
        f"age = missing, genus = sockets; I followed the metadata description, which assigns "
        f"missing teeth to the `{missing_col}` column."
    )
    explanation.append(
        "Predictors in the model included an indicator for modern humans versus non-human primates, "
        "estimated age at death, a sex proxy, and tooth-class category (anterior, posterior, premolar)."
    )
    explanation.append(
        f"The coefficient for the human indicator ({human_coef_name}) was {coef:.3f} with a "
        f"p-value of {pval:.4g}."
    )

    if not np.isnan(human_rate) and not np.isnan(nonhuman_rate):
        explanation.append(
            f"Across the dataset, the overall AMTL frequency was "
            f"{human_rate:.3%} for modern humans and {nonhuman_rate:.3%} for non-human primates."
        )

    if pval < 0.05 and coef > 0:
        explanation.append(
            "Because the human indicator is positive and statistically significant, this analysis "
            "supports the conclusion that modern humans have higher AMTL frequencies than the "
            "non-human primate genera in this sample, after adjusting for age, sex, and tooth class."
        )
    elif pval < 0.05 and coef < 0:
        explanation.append(
            "Because the human indicator is negative and statistically significant, this analysis "
            "suggests that modern humans have lower AMTL frequencies than the non-human primate "
            "genera in this sample, after adjusting for age, sex, and tooth class."
        )
    else:
        explanation.append(
            "The human indicator was not statistically significant at the 0.05 level, so the data "
            "do not provide strong evidence that AMTL frequencies differ between modern humans and "
            "the non-human primate genera once age, sex, and tooth class are accounted for."
        )

    return "\n".join(explanation)


def main():
    df, info = load_data()
    missing_col, sockets_col, invalid_a, invalid_b = infer_missing_and_sockets(df)

    result = fit_binomial_model(df, missing_col, sockets_col)

    # Identify the coefficient name for the human indicator
    human_coef_name = "is_human"
    if human_coef_name not in result.params.index:
        # Fallback: look for any parameter that starts with "is_human"
        candidates = [name for name in result.params.index if name.startswith("is_human")]
        if not candidates:
            raise RuntimeError("Could not find human indicator coefficient in the model.")
        human_coef_name = candidates[0]

    coef = result.params[human_coef_name]
    pval = result.pvalues[human_coef_name]

    response_score = map_to_likert(coef, pval)
    explanation = build_explanation(info, result, human_coef_name, missing_col, sockets_col, invalid_a, invalid_b)

    conclusion = {
        "response": int(response_score),
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
