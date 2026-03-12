import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load dataset
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Define indicator for modern humans
    df["is_human"] = df["genus"].isin(["Homo", "Homo sapiens"]).astype(int)

    # Response: binomial counts (successes = num_amtl, failures = sockets - num_amtl)
    successes = df["num_amtl"].to_numpy()
    failures = (df["sockets"] - df["num_amtl"]).to_numpy()
    endog = np.column_stack([successes, failures])

    # Predictors: human indicator, age, sex (prob_male), and tooth class (categorical)
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)
    X = pd.concat(
        [
            df[["is_human", "age", "prob_male"]],
            tooth_dummies,
        ],
        axis=1,
    )
    X = sm.add_constant(X, has_constant="add")

    # Fit binomial regression model
    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()

    # Extract effect of being human vs non-human
    coef = float(result.params["is_human"])
    pvalue = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    # Predicted probabilities at mean age and sex, anterior teeth (baseline tooth class)
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())

    base_row = {col: 0.0 for col in X.columns}
    base_row["const"] = 1.0
    base_row["age"] = mean_age
    base_row["prob_male"] = mean_prob_male

    rows = []
    for is_human in (0.0, 1.0):
        row = base_row.copy()
        row["is_human"] = is_human
        rows.append(row)

    pred_df = pd.DataFrame(rows)[X.columns]
    preds = result.predict(pred_df)
    prob_nonhuman = float(preds.iloc[0])
    prob_human = float(preds.iloc[1])
    diff_prob = prob_human - prob_nonhuman

    # Map evidence to a 0-100 Likert-style response
    if pvalue < 0.001:
        if coef > 0:
            response = 95 if odds_ratio >= 2.0 else 90
        else:
            response = 5 if odds_ratio <= 0.5 else 10
    elif pvalue < 0.01:
        response = 85 if coef > 0 else 15
    elif pvalue < 0.05:
        response = 75 if coef > 0 else 25
    else:
        if coef > 0:
            response = 55
        elif coef < 0:
            response = 45
        else:
            response = 50

    response = int(max(0, min(100, response)))

    direction = "higher" if coef > 0 else "lower" if coef < 0 else "no clear difference in"
    signif_text = (
        "strongly statistically significant (p < 0.001)"
        if pvalue < 0.001
        else "statistically significant (0.001 ≤ p < 0.01)"
        if pvalue < 0.01
        else "statistically significant (0.01 ≤ p < 0.05)"
        if pvalue < 0.05
        else "not statistically significant (p ≥ 0.05)"
    )

    n_rows = int(df.shape[0])
    n_specimens = int(df["specimen"].nunique())

    explanation = (
        "I analyzed the antemortem tooth loss (AMTL) dataset of modern humans and three non-human primate "
        f"genera (Pan, Pongo, Papio), with {n_rows} tooth-class observations from {n_specimens} specimens. "
        "For each specimen and tooth class, I modeled the number of missing teeth out of the number of observable "
        "sockets using a binomial regression (GLM with binomial family and logit link). The predictors were an "
        "indicator for modern humans versus non-human primates, age at death, estimated sex (probability of male), "
        "and tooth class (anterior vs posterior/premolar).\n\n"
        f"The estimated coefficient for the human indicator is {coef:.3f}, corresponding to an odds ratio of "
        f"{odds_ratio:.2f} and a {signif_text} p-value of {pvalue:.3g}. Holding age, sex, and tooth class at their "
        f"average or reference values, the model predicts a probability of AMTL of {prob_nonhuman:.3f} for "
        f"non-human primates and {prob_human:.3f} for modern humans, a difference of {diff_prob:.3f} in absolute "
        "probability.\n\n"
        "Given the direction and statistical strength of the human indicator, this analysis "
        + (
            "supports the conclusion that modern humans have higher frequencies of antemortem tooth loss than "
            "non-human primates, even after accounting for age, sex, and tooth class."
            if coef > 0 and pvalue < 0.05
            else "does not provide strong evidence that modern humans have higher AMTL frequencies than non-human "
            "primates after accounting for age, sex, and tooth class."
        )
        + f" On a 0–100 scale where 0 is a strong 'No' and 100 is a strong 'Yes', this evidence corresponds to a "
        f"value of {response}, reflecting a {'strong' if response >= 85 else 'moderate' if response >= 70 else 'weak or ambiguous'} "
        "answer to the research question."
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

