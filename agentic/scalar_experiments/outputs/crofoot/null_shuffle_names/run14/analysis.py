import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Rename variables based on descriptions in info.json
    df = df.rename(
        columns={
            "m_focal": "focal_win",  # 1 if focal won, 0 otherwise
            "m_other": "focal_dist",  # distance of focal group from home range center
            "n_focal": "other_dist",  # distance of other group from home range center
            "f_other": "focal_size",  # number of individuals in focal group
            "win": "other_size",  # number of individuals in other group
        }
    )

    # Construct relative predictors
    df["size_diff"] = df["focal_size"] - df["other_size"]
    df["size_ratio"] = df["focal_size"] / df["other_size"]
    df["focal_closer"] = (df["focal_dist"] < df["other_dist"]).astype(int)
    df["dist_diff"] = df["other_dist"] - df["focal_dist"]

    # Center continuous predictors for stability
    for col in ["size_diff", "size_ratio", "dist_diff"]:
        df[f"c_{col}"] = df[col] - df[col].mean()

    # Logistic regression: does relative size and location predict focal win?
    formula = "focal_win ~ c_size_diff + c_dist_diff + c_size_diff:c_dist_diff"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    print(model.summary())

    # Extract key statistics
    params = model.params
    pvalues = model.pvalues

    size_effect = params["c_size_diff"]
    size_p = pvalues["c_size_diff"]

    loc_effect = params["c_dist_diff"]
    loc_p = pvalues["c_dist_diff"]

    print("\nEffect estimates:")
    print(f"  size_diff coef = {size_effect:.3f}, p = {size_p:.3f}")
    print(f"  dist_diff coef = {loc_effect:.3f}, p = {loc_p:.3f}")

    # Heuristic mapping from evidence strength to Likert-style scalar
    # Strong, positive and significant effects of both predictors -> strong "Yes".
    if size_p < 0.01 and loc_p < 0.01 and size_effect > 0 and loc_effect > 0:
        scalar = 90
    elif size_p < 0.05 and loc_p < 0.05 and size_effect > 0 and loc_effect > 0:
        scalar = 75
    elif size_p < 0.1 and loc_p < 0.1 and size_effect > 0 and loc_effect > 0:
        scalar = 50
    elif (size_p < 0.05 and size_effect > 0) or (loc_p < 0.05 and loc_effect > 0):
        scalar = 40
    else:
        scalar = 0

    print(f"\nChosen Likert scalar conclusion: {scalar}")

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(round(scalar))))


if __name__ == "__main__":
    main()

