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

    # Basic cleaning: ensure valid counts and trials and remove impossible cases
    df = df.copy()
    df = df[(df["sockets"] > 0) & (df["num_amtl"] >= 0)]
    # Remove rows where the number of missing teeth exceeds the number of sockets
    df = df[df["num_amtl"] <= df["sockets"]]

    # Binary indicator for modern humans vs non-human primates
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Expand to per-socket Bernoulli outcomes instead of aggregated binomial counts
    df = df.reset_index(drop=True)
    df["sockets"] = df["sockets"].astype(int)
    df["num_amtl"] = df["num_amtl"].astype(int)

    df_long = df.loc[df.index.repeat(df["sockets"])].copy()
    # For each original row, mark the first num_amtl sockets as missing (1), the rest as present (0)
    df_long["amtl"] = (
        df_long.groupby(level=0).cumcount() < df_long["num_amtl"]
    ).astype(int)

    # Binomial regression on individual sockets
    # Predictors: human vs non-human, age, sex (prob_male), tooth class
    formula = "amtl ~ human + age + prob_male + C(tooth_class)"

    model = smf.glm(
        formula=formula,
        data=df_long,
        family=sm.families.Binomial(),
    )
    result = model.fit()

    # Extract human coefficient and related statistics
    params = result.params
    bse = result.bse
    human_coef = params.get("human", np.nan)
    human_se = bse.get("human", np.nan)

    # z- and p-values are available from the summary frame
    summary_frame = result.summary2().tables[1]
    human_row = summary_frame.loc["human"]
    human_z = float(human_row["z"])
    human_p = float(human_row["P>|z|"])
    human_or = float(np.exp(human_coef))

    # Predicted AMTL probabilities for a representative specimen
    mean_age = float(df_long["age"].mean())
    mean_prob_male = float(df_long["prob_male"].mean())

    # Use the most common tooth class as reference for predictions
    common_tooth_class = df_long["tooth_class"].mode().iloc[0]

    pred_df_human = pd.DataFrame(
        {
            "human": [1],
            "age": [mean_age],
            "prob_male": [mean_prob_male],
            "tooth_class": [common_tooth_class],
        }
    )
    pred_df_nonhuman = pred_df_human.copy()
    pred_df_nonhuman["human"] = 0

    pred_human = float(result.predict(pred_df_human)[0])
    pred_nonhuman = float(result.predict(pred_df_nonhuman)[0])

    # Decision rule: statistically significant positive human effect (alpha=0.05)
    if (human_coef > 0) and (human_p < 0.05):
        response = "Yes"
    else:
        response = "No"

    explanation_lines = []
    explanation_lines.append(
        "I modeled antemortem tooth loss (AMTL) at the level of individual tooth sockets, "
        "treating each socket as a Bernoulli trial (1 = missing, 0 = present)."
    )
    explanation_lines.append(
        "The predictors in a binomial GLM with logit link were: "
        "an indicator for modern humans (Homo sapiens) versus non-human primates (Pan, Pongo, Papio), "
        "age at death (continuous), probability of being male (prob_male, as a proxy for sex), "
        "and tooth class (categorical: anterior/posterior/premolar)."
    )
    explanation_lines.append(
        f"The estimated coefficient for the human indicator was {human_coef:.3f} "
        f"(SE {human_se:.3f}, z = {human_z:.2f}, p = {human_p:.3g}), "
        f"corresponding to an odds ratio of {human_or:.2f} for AMTL in humans "
        "relative to non-human primates, holding age, sex, and tooth class constant."
    )
    explanation_lines.append(
        "Using the fitted model, I predicted AMTL probabilities for a representative specimen "
        f"with age and probability of being male set to their sample means and tooth class set to "
        f"the most common category ({common_tooth_class}). "
        f"The predicted AMTL probability for humans was {pred_human:.3f}, "
        f"compared to {pred_nonhuman:.3f} for non-human primates under the same conditions."
    )
    if response == "Yes":
        explanation_lines.append(
            "Because the human coefficient is positive and statistically significant at the 0.05 level, "
            "this analysis supports the conclusion that modern humans have higher AMTL frequencies "
            "than non-human primates after accounting for age, sex, and tooth class."
        )
    else:
        explanation_lines.append(
            "Because the human coefficient is not both positive and statistically significant at the 0.05 level, "
            "this analysis does not provide strong evidence that modern humans have higher AMTL frequencies "
            "than non-human primates after accounting for age, sex, and tooth class."
        )

    explanation = " ".join(explanation_lines)

    conclusion = {"response": response, "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
