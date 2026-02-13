import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only columns needed for the research question
    cols = [
        "num_amtl",
        "sockets",
        "age",
        "prob_male",
        "tooth_class",
        "genus",
    ]
    df = df[cols].dropna()
    # Binary indicator: modern human vs non-human primate
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def fit_binomial_glm(df: pd.DataFrame):
    # Proportion of missing teeth with number of sockets as weights
    y = df["num_amtl"] / df["sockets"]

    # Design matrix: human indicator, age, sex proxy, and tooth class
    X = df[["is_human", "age", "prob_male"]].copy()
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)
    X = pd.concat([X, tooth_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()
    return model, X


def summarize_human_effect(model) -> dict:
    params = model.params
    conf_int = model.conf_int()
    pvalues = model.pvalues

    human_coef = float(params["is_human"])
    human_ci_low = float(conf_int.loc["is_human", 0])
    human_ci_high = float(conf_int.loc["is_human", 1])
    human_p = float(pvalues["is_human"])
    human_or = float(np.exp(human_coef))

    # Simple strength-of-evidence mapping to a 0–100 Likert scale
    if human_coef > 0:
        if human_p < 0.001:
            response = 95
        elif human_p < 0.01:
            response = 85
        elif human_p < 0.05:
            response = 75
        elif human_p < 0.1:
            response = 65
        else:
            response = 55
    else:
        if human_p < 0.001:
            response = 5
        elif human_p < 0.01:
            response = 15
        elif human_p < 0.05:
            response = 25
        elif human_p < 0.1:
            response = 35
        else:
            response = 45

    return {
        "response": int(response),
        "human_coef": human_coef,
        "human_ci_low": human_ci_low,
        "human_ci_high": human_ci_high,
        "human_p": human_p,
        "human_or": human_or,
    }


def make_explanation(summary: dict) -> str:
    response = summary["response"]
    coef = summary["human_coef"]
    ci_low = summary["human_ci_low"]
    ci_high = summary["human_ci_high"]
    p = summary["human_p"]
    or_human = summary["human_or"]

    if response > 50:
        qualitative = "yes, there is evidence that modern humans (Homo sapiens) have higher frequencies of antemortem tooth loss than non-human primates"
    elif response < 50:
        qualitative = "no, there is not clear evidence that modern humans have higher frequencies of antemortem tooth loss than non-human primates"
    else:
        qualitative = "the data are equivocal about whether modern humans have higher frequencies of antemortem tooth loss than non-human primates"

    explanation = (
        "I modeled the probability that a tooth was lost antemortem using a binomial "
        "generalized linear model on the amtl.csv dataset. For each specimen–tooth-class "
        "combination, I treated the number of missing teeth (num_amtl) out of the number of "
        "observable sockets (sockets) as a binomial outcome and fit a logistic regression with "
        "a binary indicator for modern humans (genus = Homo sapiens vs. Pan/Pongo/Papio), "
        "controlling for estimated age at death (age), probability of being male (prob_male), "
        "and tooth class (Anterior/Posterior/Premolar).\n\n"
        f"In this model, the coefficient for the human indicator (Homo sapiens vs. non-human primates) "
        f"is {coef:.3f} on the log-odds scale, corresponding to an odds ratio of about {or_human:.2f}. "
        f"The 95% confidence interval for this coefficient is [{ci_low:.3f}, {ci_high:.3f}], "
        f"with a p-value of {p:.4g}. This means that, after adjusting for age, sex, and tooth class, "
        "modern humans have "
        + ("higher" if coef > 0 else "lower or similar")
        + " odds of antemortem tooth loss than non-human primates, "
        + ("and the effect is statistically reliable." if p < 0.05 else "but the statistical evidence for this difference is weak or uncertain.")
        + "\n\n"
        f"Mapping this evidence onto a 0–100 Likert scale where 0 is a strong 'no' and 100 is a strong 'yes', "
        f"I assign a value of {response:d}, which reflects the direction and strength of the estimated human effect "
        "in the regression model while accounting for the available covariates."
    )

    return explanation


def main() -> None:
    df = load_data("amtl.csv")
    model, _ = fit_binomial_glm(df)
    summary = summarize_human_effect(model)
    explanation = make_explanation(summary)

    conclusion = {
        "response": int(summary["response"]),
        "explanation": explanation,
    }

    out_path = Path("conclusion.txt")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)

    # Print a brief confirmation for the user (does not affect conclusion.txt content)
    print(f"Wrote conclusion.txt with response={conclusion['response']}")


if __name__ == "__main__":
    main()

