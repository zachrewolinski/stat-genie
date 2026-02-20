import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])

    # Create human vs non-human indicator
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Descriptive summaries
    summary = (
        df.groupby("is_human")
        .agg(
            mean_prop_missing=("prop_missing", "mean"),
            std_prop_missing=("prop_missing", "std"),
            n=("prop_missing", "size"),
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
        )
        .reset_index()
    )
    print("Descriptive summary by human vs non-human (is_human=1 means Homo sapiens):")
    print(summary.to_string(index=False))
    print()

    # Binomial regression: proportion of missing teeth with number of sockets as weights
    # Predictors: human vs non-human, age, sex proxy (prob_male), and tooth class
    # Use Homo vs non-human indicator rather than individual genera to directly address the question
    df["tooth_class"] = df["tooth_class"].astype("category")

    formula = "prop_missing ~ is_human + age + prob_male + C(tooth_class)"

    # Add intercept via patsy-style formula using statsmodels
    y = df["prop_missing"]
    X = sm.add_constant(
        pd.get_dummies(
            df[["is_human", "age", "prob_male", "tooth_class"]],
            columns=["tooth_class"],
            drop_first=True,
        )
    )

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Binomial regression results (logit link), with sockets as frequency weights:")
    print(result.summary())
    print()

    # Extract coefficient and p-value for human vs non-human
    coef_human = result.params.get("is_human", np.nan)
    se_human = result.bse.get("is_human", np.nan)
    pvalue_human = result.pvalues.get("is_human", np.nan)

    print("Effect of being human (Homo sapiens) relative to non-human primates:")
    print(f"  Coefficient (log-odds): {coef_human:.4f}")
    print(f"  Std. error:            {se_human:.4f}")
    print(f"  p-value:               {pvalue_human:.4g}")
    print(f"  Odds ratio:            {np.exp(coef_human):.4f}")

    # Predicted probabilities at representative values
    # Choose median age, prob_male=0.5, and the most common tooth class
    median_age = df["age"].median()
    rep_prob_male = 0.5
    common_class = df["tooth_class"].mode()[0]

    # Construct design rows matching the dummy-encoded X structure
    base = {col: 0.0 for col in X.columns}
    base["const"] = 1.0
    base["age"] = median_age
    base["prob_male"] = rep_prob_male

    tooth_cols = [c for c in X.columns if c.startswith("tooth_class_")]
    for c in tooth_cols:
        base[c] = 0.0
    tooth_col_name = f"tooth_class_{common_class}"
    if tooth_col_name in base:
        base[tooth_col_name] = 1.0

    row_human = base.copy()
    row_human["is_human"] = 1.0
    row_nonhuman = base.copy()
    row_nonhuman["is_human"] = 0.0

    mat = pd.DataFrame([row_human, row_nonhuman], index=["human", "nonhuman"])
    preds = result.get_prediction(mat)
    pred_means = np.asarray(preds.predicted_mean)

    print()
    print("Predicted proportion of missing teeth at representative covariate values:")
    for label, value in zip(mat.index, pred_means):
        print(f"  {label}: {value:.4f}")


if __name__ == "__main__":
    main()
