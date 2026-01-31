import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():
    df = pd.read_csv("amtl.csv")

    # Basic cleaning
    needed = ["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"]
    df = df.dropna(subset=needed).copy()

    # Binary indicator for humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # One-hot encode tooth class (drop first as reference)
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)

    X = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            df[["age", "prob_male", "is_human"]],
            tooth_dummies,
        ],
        axis=1,
    )

    endog = np.column_stack([df["num_amtl"], df["sockets"] - df["num_amtl"]])

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    res = model.fit()

    coef = res.params["is_human"]
    pval = res.pvalues["is_human"]
    ci_low, ci_high = res.conf_int().loc["is_human"].tolist()

    odds_ratio = float(np.exp(coef))
    ci_or_low = float(np.exp(ci_low))
    ci_or_high = float(np.exp(ci_high))

    print(res.summary())
    print("\nHuman effect (is_human):")
    print(f"coef={coef:.4f}, p={pval:.4g}")
    print(f"odds_ratio={odds_ratio:.3f} (95% CI {ci_or_low:.3f}, {ci_or_high:.3f})")


if __name__ == "__main__":
    main()
