import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial


def main():
    df = pd.read_csv("amtl.csv")

    # Basic cleaning and derived variables
    df = df.copy()
    df["human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Ensure counts are valid
    df = df[df["sockets"] > 0].copy()
    df["num_amtl"] = df["num_amtl"].clip(lower=0)
    df["num_amtl"] = np.minimum(df["num_amtl"], df["sockets"])

    # Binomial GLM with successes/failures to avoid proportion edge cases
    failures = df["sockets"] - df["num_amtl"]
    endog = np.column_stack([df["num_amtl"].to_numpy(), failures.to_numpy()])

    exog = df[["human", "age", "prob_male"]].copy()
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)
    exog = pd.concat([exog, tooth_dummies], axis=1)
    exog = sm.add_constant(exog, has_constant="add")

    model = sm.GLM(endog, exog, family=Binomial()).fit()

    coef = float(model.params.get("human", np.nan))
    pval = float(model.pvalues.get("human", np.nan))

    # Predicted probabilities at mean covariates for context
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    mode_tooth = df["tooth_class"].mode().iloc[0]

    pred_base = {
        "const": 1.0,
        "human": 0,
        "age": mean_age,
        "prob_male": mean_prob_male,
    }
    for col in tooth_dummies.columns:
        pred_base[col] = 1.0 if col == f"tooth_{mode_tooth}" else 0.0

    pred_df = pd.DataFrame([pred_base, {**pred_base, "human": 1}])
    preds = model.predict(pred_df)
    pred_diff = float(preds.iloc[1] - preds.iloc[0])

    # Map evidence to Likert scale (-100, 100)
    if np.isnan(coef) or np.isnan(pval):
        score = 0
    else:
        if pval <= 0.01:
            evidence = 1.0
        elif pval <= 0.05:
            evidence = 0.8
        elif pval <= 0.1:
            evidence = 0.6
        elif pval <= 0.2:
            evidence = 0.4
        else:
            evidence = 0.2

        magnitude = min(1.0, abs(coef) / 1.0)
        score = int(round(100 * evidence * magnitude))
        if coef < 0:
            score = -score

    # Clamp to [-100, 100]
    score = max(-100, min(100, score))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(score)))

    # Optional: write a small diagnostics file for transparency
    with open("analysis_notes.txt", "w", encoding="utf-8") as f:
        f.write("GLM binomial with successes/failures\n")
        f.write(f"coef_human={coef:.6f}\n")
        f.write(f"pval_human={pval:.6g}\n")
        f.write(f"pred_diff={pred_diff:.6f}\n")


if __name__ == "__main__":
    main()
