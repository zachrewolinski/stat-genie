import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Outcome: focal group win indicator
    y = df["m_focal"]

    # Group sizes (based on info.json descriptions)
    # f_other: number of individuals in focal group
    # win: number of individuals in other group
    df["size_focal"] = df["f_other"]
    df["size_other"] = df["win"]

    # Relative group size metrics
    df["size_diff"] = df["size_focal"] - df["size_other"]
    df["size_ratio"] = df["size_focal"] / df["size_other"]

    # Contest location metrics
    # m_other: distance of focal group from center of its home range
    # n_focal: distance of other group from center of its home range
    df["dist_focal_center"] = df["m_other"]
    df["dist_other_center"] = df["n_focal"]

    # Relative location: positive when contest is closer to focal group's center
    df["rel_location"] = df["dist_other_center"] - df["dist_focal_center"]

    X = df[["size_ratio", "rel_location"]].copy()
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X).fit(disp=0)

    print("Number of contests:", len(df))
    print()
    print("Logistic regression: m_focal (win) ~ size_ratio + rel_location")
    print(logit_model.summary())
    print()
    print("Coefficients:")
    for name, coef, pval in zip(logit_model.params.index, logit_model.params.values, logit_model.pvalues.values):
        print(f"  {name:>12s}: coef = {coef: .4f}, p = {pval: .4g}")

    # Simple effect illustration at selected values
    size_grid = np.linspace(df["size_ratio"].min(), df["size_ratio"].max(), 5)
    loc_grid = np.linspace(df["rel_location"].min(), df["rel_location"].max(), 5)

    print()
    print("Predicted win probability across size_ratio (rel_location at mean):")
    mean_loc = df["rel_location"].mean()
    for s in size_grid:
        x_vec = [1.0, s, mean_loc]
        p = float(logit_model.predict([x_vec])[0])
        print(f"  size_ratio = {s: .2f} -> P(win) = {p: .3f}")

    print()
    print("Predicted win probability across rel_location (size_ratio at mean):")
    mean_size = df["size_ratio"].mean()
    for l in loc_grid:
        x_vec = [1.0, mean_size, l]
        p = float(logit_model.predict([x_vec])[0])
        print(f"  rel_location = {l: .1f} -> P(win) = {p: .3f}")


if __name__ == "__main__":
    main()

