import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing",
            "feature4": "sockets",
            "feature5": "age",
            "feature6": "age_unc",
            "feature7": "sex_est",
            "feature8": "genus",
            "feature9": "region",
        }
    )
    # Basic cleaning and type enforcement
    df = df.copy()
    df["missing"] = df["missing"].astype(int)
    df["sockets"] = df["sockets"].astype(int)
    df = df[(df["sockets"] > 0) & (df["missing"] >= 0) & (df["missing"] <= df["sockets"])]

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex_est"] = pd.to_numeric(df["sex_est"], errors="coerce")
    df = df.dropna(subset=["age", "sex_est"])

    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    return df


def expand_to_sockets(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        sockets = int(row["sockets"])
        missing = int(row["missing"])
        present = sockets - missing
        base = {
            "tooth_class": row["tooth_class"],
            "age": float(row["age"]),
            "sex_est": float(row["sex_est"]),
            "is_human": int(row["is_human"]),
        }
        if missing > 0:
            rows.extend({**base, "is_missing": 1} for _ in range(missing))
        if present > 0:
            rows.extend({**base, "is_missing": 0} for _ in range(present))
    socket_df = pd.DataFrame(rows)
    return socket_df


def fit_model(socket_df: pd.DataFrame):
    formula = "is_missing ~ is_human + age + sex_est + C(tooth_class)"
    model = smf.glm(formula=formula, data=socket_df, family=sm.families.Binomial())
    result = model.fit()
    return result


def infer_answer(result) -> dict:
    coef_human = result.params.get("is_human", np.nan)
    pvalue_human = result.pvalues.get("is_human", np.nan)

    odds_ratio = float(np.exp(coef_human)) if np.isfinite(coef_human) else np.nan

    if np.isnan(coef_human) or np.isnan(pvalue_human):
        response = "No"
        confidence = 20
        explanation = (
            "The regression model could not reliably estimate the effect of being "
            "human on the probability of antemortem tooth loss."
        )
        return {
            "response": response,
            "confidence": confidence,
            "explanation": explanation,
        }

    if odds_ratio > 1 and pvalue_human < 0.05:
        response = "Yes"
        if pvalue_human < 0.001:
            confidence = 95
        elif pvalue_human < 0.01:
            confidence = 90
        else:
            confidence = 80
    elif odds_ratio < 1 and pvalue_human < 0.05:
        response = "No"
        if pvalue_human < 0.001:
            confidence = 95
        elif pvalue_human < 0.01:
            confidence = 90
        else:
            confidence = 80
    else:
        response = "Yes" if odds_ratio > 1 else "No"
        confidence = 50

    # Compute adjusted predicted probabilities for humans vs non-humans
    socket_df = result.model.data.frame
    mean_age = float(socket_df["age"].mean())
    mean_sex = float(socket_df["sex_est"].mean())
    common_tooth_class = socket_df["tooth_class"].mode().iloc[0]

    pred_data_human = pd.DataFrame(
        {
            "is_human": [1],
            "age": [mean_age],
            "sex_est": [mean_sex],
            "tooth_class": [common_tooth_class],
        }
    )
    pred_data_nonhuman = pred_data_human.copy()
    pred_data_nonhuman["is_human"] = 0

    prob_human = float(result.predict(pred_data_human)[0])
    prob_nonhuman = float(result.predict(pred_data_nonhuman)[0])

    explanation_lines = [
        "I fitted a binomial (logistic) regression model at the individual tooth-socket level,",
        "with the binary outcome indicating whether a socket showed antemortem tooth loss.",
        "Predictors included an indicator for modern humans (Homo sapiens vs. non-human primates),",
        "estimated age at death, estimated sex, and tooth class (anterior/posterior/premolar).",
        f"The estimated coefficient for the human indicator corresponds to an odds ratio of approximately {odds_ratio:.2f},",
        f"with a p-value of {pvalue_human:.3g}, indicating that this effect is "
        f"{'statistically significant' if pvalue_human < 0.05 else 'not statistically significant'} at the 0.05 level.",
        f"Holding age, sex, and tooth class constant at their typical values, the model predicts a probability of AMTL of about {prob_human:.3f} for humans",
        f"and about {prob_nonhuman:.3f} for non-human primates.",
        "Based on the direction and statistical strength of this effect, I concluded that modern humans "
        f"{'do' if response == 'Yes' else 'do not'} have higher frequencies of antemortem tooth loss than the non-human primates considered,",
        "after accounting for age, sex, and tooth class.",
    ]

    explanation = " ".join(explanation_lines)

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main():
    csv_path = Path("amtl.csv")
    df = load_data(csv_path)
    socket_df = expand_to_sockets(df)
    result = fit_model(socket_df)
    conclusion = infer_answer(result)

    output_path = Path("conclusion.txt")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

