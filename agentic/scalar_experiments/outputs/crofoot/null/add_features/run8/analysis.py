import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors based on the research question
    # Relative group size: focal minus other (positive when focal is larger)
    df["rel_group_size"] = df["n_focal"] - df["n_other"]

    # Relative location: other distance minus focal distance (positive when focal is closer to its home range center)
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]

    # Standardize predictors to make coefficients comparable
    for col in ["rel_group_size", "rel_dist"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        df[f"z_{col}"] = (df[col] - mean) / std

    X = df[["z_rel_group_size", "z_rel_dist"]]
    X = sm.add_constant(X, has_constant="add")
    y = df["win"]

    logit_model = sm.Logit(y, X)
    logit_result = logit_model.fit(disp=False)

    print("=== Logistic regression: win ~ relative group size + relative location ===")
    print(logit_result.summary())

    # Odds ratios and 95% confidence intervals
    params = logit_result.params
    conf_int = logit_result.conf_int()
    odds_ratios = np.exp(params)
    or_ci = np.exp(conf_int)

    print("\n=== Odds ratios (per 1 SD increase) ===")
    for name in ["z_rel_group_size", "z_rel_dist"]:
        or_val = odds_ratios[name]
        ci_low, ci_high = or_ci.loc[name]
        p_val = logit_result.pvalues[name]
        print(
            f"{name}: OR={or_val:.3f}, 95% CI=({ci_low:.3f}, {ci_high:.3f}), p={p_val:.4f}"
        )

    # Predicted probabilities for interpretation
    def predict_prob(z_size: float, z_dist: float) -> float:
        X_new = pd.DataFrame(
            {"const": [1.0], "z_rel_group_size": [z_size], "z_rel_dist": [z_dist]}
        )
        return float(logit_result.predict(X_new)[0])

    base_prob = predict_prob(0.0, 0.0)
    size_up = predict_prob(1.0, 0.0)
    size_down = predict_prob(-1.0, 0.0)
    dist_up = predict_prob(0.0, 1.0)
    dist_down = predict_prob(0.0, -1.0)

    print("\n=== Predicted win probabilities (focal group) ===")
    print(f"At average size and neutral location (z=0,0): {base_prob:.3f}")
    print(
        f"Focal 1 SD larger (z_size=+1, z_dist=0): {size_up:.3f} "
        f"(vs. 1 SD smaller: {size_down:.3f})"
    )
    print(
        f"Contest 1 SD closer to focal home range (z_dist=+1, z_size=0): {dist_up:.3f} "
        f"(vs. 1 SD closer to other: {dist_down:.3f})"
    )


if __name__ == "__main__":
    main()

