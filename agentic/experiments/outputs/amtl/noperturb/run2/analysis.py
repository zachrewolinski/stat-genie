import math

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Binary indicator for modern humans vs non-human primates.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Prepare binomial response as successes/failures.
    endog = pd.DataFrame(
        {
            "successes": df["num_amtl"],
            "failures": df["sockets"] - df["num_amtl"],
        }
    )

    # Design matrix with baseline tooth_class = Anterior.
    exog = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
    )
    exog = sm.add_constant(exog, has_constant="add")

    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    res = model.fit()

    coef = res.params["is_human"]
    se = res.bse["is_human"]
    odds_ratio = math.exp(coef)
    ci_low = math.exp(coef - 1.96 * se)
    ci_high = math.exp(coef + 1.96 * se)
    p_value = res.pvalues["is_human"]

    print(res.summary())
    print("\nHuman vs non-human effect (logit scale):")
    print(f"coef={coef:.4f}, SE={se:.4f}, p={p_value:.3e}")
    print(f"odds_ratio={odds_ratio:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f})")

    # Predicted probability for Anterior tooth_class at mean age/sex.
    mean_age = df["age"].mean()
    mean_male = df["prob_male"].mean()
    pred_df = pd.DataFrame(
        {
            "const": [1.0, 1.0],
            "is_human": [0, 1],
            "age": [mean_age, mean_age],
            "prob_male": [mean_male, mean_male],
            "tooth_class_Posterior": [0, 0],
            "tooth_class_Premolar": [0, 0],
        }
    )
    pred_probs = res.predict(pred_df)
    print(
        "Predicted AMTL probability (Anterior, mean age/sex) "
        f"non-human={pred_probs.iloc[0]:.4f}, human={pred_probs.iloc[1]:.4f}"
    )


if __name__ == "__main__":
    main()
