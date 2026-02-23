import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 if other group won.
    y = df["feature4"]

    # Relative group size: focal size minus other size (total individuals).
    df["rel_size"] = df["feature7"] - df["feature8"]

    # Contest location advantage: how much closer the focal group is to its home range center.
    # Distances are from each group's home-range center to the contest location.
    # If the focal group is closer, feature5 < feature6, so loc_adv > 0.
    df["loc_adv"] = df["feature6"] - df["feature5"]

    X = df[["rel_size", "loc_adv"]]
    X = sm.add_constant(X)

    # Cluster-robust standard errors by dyad (feature3) to account for repeated contests.
    clusters = df["feature3"]

    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": clusters})

    print("Logistic regression of focal win on relative size and location advantage")
    print(result.summary())

    # For additional intuition, show predicted probabilities at representative values.
    for rel_size in [-4, -2, 0, 2, 4]:
        for loc_adv in [-300, -100, 0, 100, 300]:
            x_row = {"const": 1.0, "rel_size": rel_size, "loc_adv": loc_adv}
            x_vec = [x_row[col] for col in result.params.index]
            p = result.predict([x_vec])[0]
            print(
                f"rel_size={rel_size:>3}, loc_adv={loc_adv:>4} -> "
                f"predicted P(focal win)={p:.3f}"
            )


if __name__ == "__main__":
    main()

