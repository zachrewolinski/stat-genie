import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Basic derived variables
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = df["genus"].str.contains("Homo", case=False, na=False).astype(int)
    df["failures"] = df["sockets"] - df["num_amtl"]

    # Sanity checks: drop any rows with non-positive sockets just in case
    df = df[df["sockets"] > 0].copy()

    # Descriptive statistics: observed AMTL rates by genus
    genus_summary = (
        df.groupby("genus")
        .agg(
            n_rows=("specimen", "size"),
            mean_amtl_rate=("amtl_rate", "mean"),
        )
        .reset_index()
    )

    # Binomial regression: AMTL (successes vs failures) as a function of
    # human vs non-human, age, sex proxy, and tooth class.
    formula = "num_amtl + failures ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    coef_human = float(result.params["is_human"])
    pval_human = float(result.pvalues["is_human"])
    ci_low, ci_high = map(float, result.conf_int().loc["is_human"])

    # Average predicted AMTL probabilities if everyone were non-human vs human
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0
    df_human = df.copy()
    df_human["is_human"] = 1

    pred_nonhuman = result.predict(df_nonhuman)
    pred_human = result.predict(df_human)

    mean_pred_nonhuman = float(pred_nonhuman.mean())
    mean_pred_human = float(pred_human.mean())
    diff_mean = mean_pred_human - mean_pred_nonhuman
    ratio_mean = mean_pred_human / mean_pred_nonhuman if mean_pred_nonhuman > 0 else np.nan

    # Decide on Yes/No and Likert-scale strength based on direction,
    # statistical significance, and effect size.
    ci_positive = ci_low > 0
    ci_negative = ci_high < 0

    if coef_human > 0 and ci_positive and pval_human < 0.001:
        response = 95
        qualitative = "a very strong 'Yes'"
        answer_text = "Yes"
    elif coef_human > 0 and ci_positive and pval_human < 0.01:
        response = 90
        qualitative = "a strong 'Yes'"
        answer_text = "Yes"
    elif coef_human > 0 and ci_positive and pval_human < 0.05:
        response = 80
        qualitative = "a clear 'Yes'"
        answer_text = "Yes"
    elif coef_human > 0 and pval_human < 0.1 and diff_mean > 0:
        response = 65
        qualitative = "a tentative 'Yes'"
        answer_text = "Yes"
    elif coef_human > 0 and diff_mean > 0:
        response = 55
        qualitative = "a weak, non-significant tendency toward 'Yes'"
        answer_text = "Yes"
    elif coef_human < 0 and ci_negative and pval_human < 0.001:
        response = 5
        qualitative = "a very strong 'No'"
        answer_text = "No"
    elif coef_human < 0 and ci_negative and pval_human < 0.01:
        response = 10
        qualitative = "a strong 'No'"
        answer_text = "No"
    elif coef_human < 0 and ci_negative and pval_human < 0.05:
        response = 20
        qualitative = "a clear 'No'"
        answer_text = "No"
    elif coef_human < 0 and pval_human < 0.1 and diff_mean < 0:
        response = 35
        qualitative = "a tentative 'No'"
        answer_text = "No"
    elif coef_human < 0 and diff_mean < 0:
        response = 40
        qualitative = "a weak, non-significant tendency toward 'No'"
        answer_text = "No"
    else:
        # Effect very close to zero or highly uncertain.
        response = 50
        qualitative = "an essentially indeterminate answer"
        answer_text = "Uncertain"

    response = int(response)

    # Build a human-readable explanation that summarizes data patterns
    # and the model results.
    genus_lines = []
    for _, row in genus_summary.iterrows():
        genus_lines.append(
            f"{row['genus']}: n={int(row['n_rows'])}, mean AMTL rate={row['mean_amtl_rate']:.3f}"
        )
    genus_text = "; ".join(genus_lines)

    direction = "higher" if diff_mean > 0 else "lower"

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies of "
        "antemortem tooth loss (AMTL) than non-human primate genera (Pan, Pongo, Papio), "
        "after accounting for age, sex, and tooth class?\n\n"
        "Data and descriptive patterns:\n"
        f"- Dataset contains {len(df)} tooth-class-by-specimen records with AMTL counts and numbers of observable sockets.\n"
        f"- Genera represented with mean AMTL rates (num_amtl / sockets) are: {genus_text}.\n\n"
        "Modeling approach:\n"
        "- Fitted a binomial logistic regression using AMTL counts (num_amtl) vs non-missing sockets "
        "(sockets - num_amtl) as the response.\n"
        "- Predictors included an indicator for modern humans vs non-human primates (is_human), "
        "continuous age at death, a continuous probability-of-male sex estimate (prob_male), and "
        "categorical tooth class (Anterior, Posterior, Premolar).\n"
        "- Cluster-robust standard errors were used at the specimen level to account for non-independence "
        "of multiple tooth classes from the same individual.\n\n"
        "Key results for the human vs non-human contrast:\n"
        f"- Coefficient for the human indicator (is_human) on the log-odds scale: {coef_human:.3f} "
        f"(p = {pval_human:.3g}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]).\n"
        f"- Average predicted AMTL probability per tooth if all observations were non-human: {mean_pred_nonhuman:.3f}.\n"
        f"- Average predicted AMTL probability per tooth if all observations were human: {mean_pred_human:.3f}.\n"
        f"- This corresponds to humans having {direction} AMTL frequencies by about {abs(diff_mean):.3f} "
        f"on the probability scale (ratio human/non-human ≈ {ratio_mean:.2f}).\n\n"
        "Interpretation and conclusion:\n"
        f"- Based on the sign and statistical significance of the human indicator, along with the size of the "
        f"predicted difference in AMTL probabilities, the data support {qualitative} answer to the question "
        f"\"Do modern humans have higher AMTL frequencies than non-human primates after accounting for age, sex, "
        "and tooth class?\".\n"
        f"- Final qualitative answer: {answer_text}.\n"
        f"- Corresponding quantitative Likert-scale response (0 = strong 'No', 100 = strong 'Yes'): {response}."
    )

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

