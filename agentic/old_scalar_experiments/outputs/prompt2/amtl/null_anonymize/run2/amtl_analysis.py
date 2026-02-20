import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_analysis() -> dict:
    """Run binomial regression to assess human vs non-human AMTL frequencies."""
    info_path = Path("info.json")
    data_path = Path("amtl.csv")

    with info_path.open("r") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    # Basic cleaning: require valid socket counts and missing counts
    df = df[df["feature4"] > 0].copy()
    df = df[(df["feature3"] >= 0) & (df["feature3"] <= df["feature4"])].copy()

    # Indicator for modern humans vs non-human primates (numeric 0/1)
    df["is_human"] = df["feature8"].astype(str).str.contains("Homo").astype(int)

    # Proportion of missing teeth for the tooth class
    df["prop_missing"] = df["feature3"] / df["feature4"]

    # Predictors: human indicator, age at death, sex estimate, tooth class dummies
    tooth_dummies = pd.get_dummies(
        df["feature1"], prefix="tooth_class", drop_first=True
    )

    X = pd.concat(
        [
            df[["is_human", "feature5", "feature7"]].rename(
                columns={"feature5": "age", "feature7": "sex_est"}
            ),
            tooth_dummies,
        ],
        axis=1,
    )
    X = sm.add_constant(X, has_constant="add")
    # Ensure all predictors are numeric
    X = X.apply(pd.to_numeric)

    y = df["prop_missing"].astype(float)
    weights = df["feature4"].astype(float)

    # Binomial regression with logit link on proportions with frequency weights
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()

    human_coef = float(result.params["is_human"])
    human_p = float(result.pvalues["is_human"])

    # Predicted AMTL frequencies per tooth site for humans vs non-humans
    X_nonhuman = X.copy()
    X_nonhuman["is_human"] = 0
    X_human = X.copy()
    X_human["is_human"] = 1

    pred_nonhuman = result.predict(X_nonhuman)
    pred_human = result.predict(X_human)

    avg_nonhuman = float(np.average(pred_nonhuman, weights=weights))
    avg_human = float(np.average(pred_human, weights=weights))

    # Determine yes/no answer from direction and significance
    if (avg_human > avg_nonhuman) and (human_p < 0.05):
        response = "Yes"
    else:
        response = "No"

    # Map p-value to a heuristic confidence score
    if human_p < 1e-8:
        base_conf = 95
    elif human_p < 1e-4:
        base_conf = 90
    elif human_p < 1e-2:
        base_conf = 85
    elif human_p < 5e-2:
        base_conf = 75
    else:
        base_conf = 60

    # Adjust confidence if the estimated difference is small
    diff = avg_human - avg_nonhuman
    if abs(diff) < 0.01:
        base_conf = min(base_conf, 70)

    confidence = int(max(0, min(100, round(base_conf))))

    explanation = (
        f"Research question: {question} "
        f"I analyzed the AMTL dataset using a binomial regression model with a logit link, "
        f"modeling the proportion of missing teeth (number missing / observable sockets) for each specimen "
        f"and tooth class as the outcome. The predictors in the model were a binary indicator for modern humans "
        f"(Homo) versus non-human primates (Pan, Papio, Pongo), estimated age at death, estimated sex, and tooth "
        f"class (anterior, posterior, premolar) represented with dummy variables "
        f"(N={len(df)} rows, totaling approximately {int(weights.sum())} observable tooth positions). "
        f"The regression coefficient for the human indicator was {human_coef:.3f} on the log-odds scale "
        f"(p-value {human_p:.2e}), indicating "
        f"{'higher' if human_coef > 0 else 'lower' if human_coef < 0 else 'no clear change in'} AMTL odds "
        f"for modern humans relative to non-human primates after adjusting for age, sex, and tooth class. "
        f"Using the fitted model, the predicted AMTL frequency per tooth site, averaged over the observed "
        f"distribution of ages, sexes, and tooth classes, was {avg_human:.3f} for humans and "
        f"{avg_nonhuman:.3f} for non-human primates. "
        f"Because the adjusted human AMTL frequency is "
        f"{'higher' if avg_human > avg_nonhuman else 'not higher'} than that of non-human primates and the "
        f"human effect is {'statistically significant' if human_p < 0.05 else 'not strongly supported statistically'}, "
        f"I conclude that modern humans {'' if response == 'Yes' else 'do not '}have higher AMTL frequencies than "
        f"the non-human primate genera in this sample once age, sex, and tooth class are taken into account."
    )

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    result = run_analysis()
    with open("conclusion.txt", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
