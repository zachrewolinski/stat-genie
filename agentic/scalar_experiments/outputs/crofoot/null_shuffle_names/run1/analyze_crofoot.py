import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Outcome: 1 if focal group won, 0 otherwise
    y = df["m_focal"]

    # Relative group size: focal size minus other size
    rel_group_size = df["f_other"] - df["win"]

    # Contest location: distances of each group from the center of its own home range
    # m_other: focal distance; n_focal: other distance (per info.json description)
    focal_dist = df["m_other"]
    other_dist = df["n_focal"]

    X = pd.DataFrame(
        {
            "rel_group_size": rel_group_size,
            "focal_dist": focal_dist,
            "other_dist": other_dist,
        }
    )
    X = sm.add_constant(X)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    print("Logistic regression results for probability focal group wins:")
    print(result.summary2())

    # Simple diagnostics to understand direction and strength of effects
    print("\nCoefficient estimates:")
    print(result.params)
    print("\nP-values:")
    print(result.pvalues)


if __name__ == "__main__":
    main()

