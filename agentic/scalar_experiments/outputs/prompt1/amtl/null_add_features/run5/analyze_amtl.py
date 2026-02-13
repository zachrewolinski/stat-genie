import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Keep rows with valid socket counts and required fields
    df = df[df["sockets"] > 0].copy()
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Proportion of missing teeth within each tooth_class/specimen entry
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans (Homo sapiens) vs non-human primates
    genus_clean = df["genus"].astype(str).str.strip()
    df["is_human"] = (genus_clean == "Homo sapiens").astype(int)

    # Center age to improve numerical stability
    df["age_c"] = df["age"] - df["age"].mean()

    # Binomial GLM with logit link; weights are the number of observable sockets
    formula = "prop_amtl ~ is_human + age_c + prob_male + C(tooth_class)"
    family = sm.families.Binomial()

    try:
        model = smf.glm(formula=formula, data=df, family=family, freq_weights=df["sockets"])
    except TypeError:
        # Fallback in case this version of statsmodels uses 'weights' instead
        model = smf.glm(formula=formula, data=df, family=family, weights=df["sockets"])

    result = model.fit()

    coef_human = float(result.params.get("is_human", np.nan))
    p_human = float(result.pvalues.get("is_human", np.nan))

    # Weighted mean AMTL proportions for descriptive comparison
    def weighted_mean(group: pd.DataFrame) -> float:
        return float(np.average(group["prop_amtl"], weights=group["sockets"]))

    human = df[df["is_human"] == 1]
    nonhuman = df[df["is_human"] == 0]

    mean_human = weighted_mean(human) if not human.empty else float("nan")
    mean_nonhuman = weighted_mean(nonhuman) if not nonhuman.empty else float("nan")

    odds_ratio = float(np.exp(coef_human)) if np.isfinite(coef_human) else float("nan")

    try:
        ci_low, ci_high = result.conf_int().loc["is_human"]
        ci_low = float(ci_low)
        ci_high = float(ci_high)
    except Exception:
        ci_low, ci_high = float("nan"), float("nan")

    # Decision rule: humans have higher AMTL if the human coefficient is positive
    # and statistically significant at alpha = 0.05.
    if np.isfinite(coef_human) and coef_human > 0 and p_human < 0.05:
        response = "Yes"
    else:
        response = "No"

    explanation = (
        "I modeled the proportion of antemortem tooth loss (num_amtl / sockets) per "
        "specimen–tooth-class entry using a binomial logistic regression with a logit link. "
        "The outcome was the proportion of missing teeth, with the number of observable "
        "sockets used as binomial weights. Predictors included an indicator for modern humans "
        "(Homo sapiens vs. non-human primates), centered age at death, probability of being "
        "male, and tooth class (Anterior, Posterior, Premolar). "
        f"Weighted mean AMTL was approximately {mean_human:.3f} for modern humans and "
        f"{mean_nonhuman:.3f} for non-human primates. "
        f"In the regression model, the coefficient for humans was {coef_human:.3f}, "
        f"corresponding to an odds ratio of about {odds_ratio:.2f}, with a 95% confidence "
        f"interval of [{ci_low:.2f}, {ci_high:.2f}] and p-value {p_human:.4f}. "
        "Based on this model, after accounting for age, sex, and tooth class, "
        "modern humans "
        + ("do" if response == "Yes" else "do not")
        + " show significantly higher frequencies of antemortem tooth loss than the "
        "non-human primate genera (Pan, Pongo, Papio)."
    )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

