import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic preprocessing and variable creation
    df = df.copy()
    df = df[df["feature4"] > 0]  # keep only rows with observable sockets

    df["missing"] = df["feature3"].astype(float)
    df["sockets"] = df["feature4"].astype(float)
    df["prop_missing"] = df["missing"] / df["sockets"]

    # Predictor variables
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)
    df["age"] = df["feature5"].astype(float)
    df["sex_est"] = df["feature7"].astype(float)
    df["tooth_class"] = df["feature1"].astype("category")

    # Binomial regression on proportions with number of trials as frequency weights
    model = smf.glm(
        formula="prop_missing ~ is_human + age + sex_est + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    is_human_coef = float(result.params["is_human"])
    is_human_p = float(result.pvalues["is_human"])
    is_human_ci_low, is_human_ci_high = result.conf_int().loc["is_human"].tolist()
    is_human_or = float(np.exp(is_human_coef))

    humans_higher = is_human_coef > 0 and is_human_p < 0.05
    response = "Yes" if humans_higher else "No"

    explanation = (
        "I fit a binomial regression model for the proportion of missing teeth "
        "(number missing out of observable sockets) with predictors for human vs. "
        "non-human genus, age at death, estimated sex, and tooth class. "
        f"The coefficient for the human indicator (Homo sapiens vs. all non-human primates) "
        f"was {is_human_coef:.3f} on the log-odds scale (odds ratio ≈ {is_human_or:.2f}), "
        f"with p-value {is_human_p:.3g} and 95% confidence interval "
        f"[{is_human_ci_low:.3f}, {is_human_ci_high:.3f}]. "
        "A positive and statistically significant coefficient indicates that, after "
        "accounting for age, sex, and tooth class, modern humans have higher odds of "
        "antemortem tooth loss compared to the non-human primate genera in this dataset."
    )

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)

    # Also print a brief summary to stdout for inspection
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()

