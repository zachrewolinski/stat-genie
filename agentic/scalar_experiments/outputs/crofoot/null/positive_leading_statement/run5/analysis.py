import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Relative group size: positive values mean the focal group is larger
    df["rel_size"] = df["n_focal"] - df["n_other"]

    # Relative location advantage: positive values mean the focal group is closer
    # to the center of its home range than the other group is to its own
    df["loc_adv"] = df["dist_other"] - df["dist_focal"]

    print("Basic description")
    print(df[["win", "rel_size", "loc_adv"]].describe())
    print()

    # Logistic regression: probability that the focal group wins
    y = df["win"]
    X = df[["rel_size", "loc_adv"]]
    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    print("Logistic regression: win ~ rel_size + loc_adv")
    print(result.summary())
    print()

    # Odds ratios for easier interpretation
    params = result.params
    conf = result.conf_int()
    odds_ratios = np.exp(params)
    conf_or = np.exp(conf)

    print("Odds ratios (exp(coeff)) with 95% CI")
    for name in params.index:
        print(
            f"{name:>10s}: OR={odds_ratios[name]:.3f}, "
            f"95% CI=({conf_or.loc[name, 0]:.3f}, {conf_or.loc[name, 1]:.3f})"
        )

    print()

    # Simple contrasts for interpretation: small vs large group, home vs away
    # Use median splits for illustration
    size_median = df["rel_size"].median()
    loc_median = df["loc_adv"].median()

    def predict_prob(rel_size: float, loc_adv: float) -> float:
        row = pd.DataFrame({"const": [1.0], "rel_size": [rel_size], "loc_adv": [loc_adv]})
        return float(result.predict(row)[0])

    # Smaller vs larger group at neutral location
    p_small = predict_prob(size_median - 2, loc_median)
    p_large = predict_prob(size_median + 2, loc_median)

    # Focal at disadvantage vs advantage in location, for neutral size
    p_away = predict_prob(size_median, loc_median - 150)
    p_home = predict_prob(size_median, loc_median + 150)

    print("Illustrative predicted probabilities (not used directly in scoring)")
    print(f"Smaller vs larger group (neutral location): {p_small:.3f} vs {p_large:.3f}")
    print(f"Location disadvantage vs advantage (neutral size): {p_away:.3f} vs {p_home:.3f}")


if __name__ == "__main__":
    main()

