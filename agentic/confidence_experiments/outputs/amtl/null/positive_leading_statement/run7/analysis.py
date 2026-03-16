import json
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm


def expand_to_sockets(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        num_amtl = int(row["num_amtl"])
        num_present = sockets - num_amtl

        base = {
            "age": float(row["age"]),
            "prob_male": float(row["prob_male"]),
            "tooth_class": row["tooth_class"],
            "genus": row["genus"],
        }

        for _ in range(num_amtl):
            r = dict(base)
            r["amtl"] = 1
            rows.append(r)

        for _ in range(num_present):
            r = dict(base)
            r["amtl"] = 0
            rows.append(r)

    return pd.DataFrame(rows)


def fit_model(df_sockets: pd.DataFrame):
    df_sockets = df_sockets.copy()
    df_sockets["is_human"] = (df_sockets["genus"] == "Homo sapiens").astype(int)
    df_sockets["age_c"] = df_sockets["age"] - df_sockets["age"].mean()
    df_sockets["prob_male_c"] = df_sockets["prob_male"] - df_sockets["prob_male"].mean()

    df_sockets = pd.get_dummies(df_sockets, columns=["tooth_class"], drop_first=True)

    y = df_sockets["amtl"].astype(int)
    predictor_cols = ["is_human", "age_c", "prob_male_c"]
    predictor_cols.extend(c for c in df_sockets.columns if c.startswith("tooth_class_"))

    X = df_sockets[predictor_cols]
    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result, X


def build_explanation(result, X) -> Dict:
    params = result.params
    pvalues = result.pvalues
    conf_int = result.conf_int()

    coef_human = float(params["is_human"])
    pvalue_human = float(pvalues["is_human"])
    ci_low, ci_high = conf_int.loc["is_human"]

    odds_ratio = float(np.exp(coef_human))
    or_low = float(np.exp(ci_low))
    or_high = float(np.exp(ci_high))

    X_human = X.copy()
    X_nonhuman = X.copy()
    X_human["is_human"] = 1
    X_nonhuman["is_human"] = 0

    pred_human = float(result.predict(X_human).mean())
    pred_nonhuman = float(result.predict(X_nonhuman).mean())
    diff = pred_human - pred_nonhuman

    if coef_human > 0 and pvalue_human < 0.001:
        response_int = 95
    elif coef_human > 0 and pvalue_human < 0.01:
        response_int = 90
    elif coef_human > 0 and pvalue_human < 0.05:
        response_int = 80
    elif coef_human > 0 and pvalue_human < 0.1:
        response_int = 65
    elif coef_human > 0:
        response_int = 55
    elif coef_human < 0 and pvalue_human < 0.05:
        response_int = 10
    else:
        response_int = 40

    if response_int > 50:
        answer_label = "Yes"
    else:
        answer_label = "No"

    if pvalue_human < 0.05:
        if coef_human > 0:
            effect_interp = (
                "This indicates that, after adjusting for age, sex, and tooth class, "
                "modern human tooth sockets have statistically significantly higher odds "
                "of showing AMTL than sockets from the non-human primate genera."
            )
        else:
            effect_interp = (
                "This indicates that, after adjusting for age, sex, and tooth class, "
                "modern human tooth sockets have statistically significantly lower odds "
                "of showing AMTL than sockets from the non-human primate genera."
            )
    else:
        if coef_human > 0:
            effect_interp = (
                "Although the point estimate suggests slightly higher odds of AMTL for "
                "modern humans, the confidence interval includes an odds ratio of 1 and "
                "the effect is not statistically significant, so the data do not provide "
                "strong evidence that humans differ from the non-human primate genera."
            )
        else:
            effect_interp = (
                "The point estimate suggests slightly lower odds of AMTL for modern humans, "
                "but the confidence interval includes an odds ratio of 1 and the effect is "
                "far from statistically significant, so the data do not provide strong "
                "evidence that humans differ from the non-human primate genera."
            )

    if answer_label == "Yes":
        conclusion_sentence = (
            "Taken together, these results support a 'Yes' answer to the research "
            "question that modern humans have higher AMTL frequencies than the "
            "non-human primate genera considered."
        )
    else:
        conclusion_sentence = (
            "Taken together, these results do not support the claim that modern humans "
            "have higher AMTL frequencies than the non-human primate genera considered; "
            "the data are more consistent with similar or even slightly lower AMTL "
            "frequencies in humans."
        )

    conclusion_sentence += (
        f" I therefore answer '{answer_label}' and assign a confidence rating of "
        f"{response_int} on a 0–100 Likert scale, where 0 represents a strong 'No' and "
        "100 represents a strong 'Yes'."
    )

    explanation = (
        "I analyzed the antemortem tooth loss (AMTL) data using a binomial logistic "
        "regression model at the level of individual tooth sockets. Each row in the "
        "original dataset was expanded so that every observable socket contributed a "
        "binary outcome (1 = missing due to AMTL, 0 = present). The model estimated the "
        "log-odds of AMTL as a function of an indicator for modern humans versus "
        "non-human primates, age at death, the probability of being male, and dummy "
        "variables for tooth class (anterior/posterior/premolar).\n\n"
        f"In this model, the coefficient for the modern human indicator (Homo sapiens "
        f"vs. Pan/Papio/Pongo) was {coef_human:.3f}, corresponding to an odds ratio of "
        f"{odds_ratio:.2f} with a 95% confidence interval from {or_low:.2f} to "
        f"{or_high:.2f} (p = {pvalue_human:.3g}). "
        f"{effect_interp}\n\n"
        f"Using the fitted model, the mean predicted probability of AMTL for sockets from "
        f"non-human primates was {pred_nonhuman:.3f}, whereas for sockets from modern "
        f"humans it was {pred_human:.3f} (difference = {diff:.3f}). "
        "These model-based predicted probabilities reflect the adjusted comparison after "
        "controlling for age, sex, and tooth class.\n\n"
        f"{conclusion_sentence}"
    )

    return {"response": int(response_int), "explanation": explanation}


def main():
    df = pd.read_csv("amtl.csv")
    df_sockets = expand_to_sockets(df)
    result, X = fit_model(df_sockets)
    conclusion = build_explanation(result, X)

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
