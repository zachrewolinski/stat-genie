import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Define key variables based on metadata
    # Outcome: focal group wins
    y = df["feature4"]

    # Relative group size: focal minus other
    df["rel_size"] = df["feature7"] - df["feature8"]

    # Contest location: indicator that focal group is closer to its home range center
    df["focal_closer"] = (df["feature5"] < df["feature6"]).astype(int)

    # Interaction term to test whether size advantage depends on location
    df["rel_size_x_focal_closer"] = df["rel_size"] * df["focal_closer"]

    print("Basic description of key variables:")
    print(df[["feature4", "rel_size", "focal_closer"]].describe())
    print()

    # Model 1: relative size only
    X1 = sm.add_constant(df[["rel_size"]])
    model1 = sm.GLM(y, X1, family=sm.families.Binomial())
    res1 = model1.fit(cov_type="cluster", cov_kwds={"groups": df["feature3"]})

    print("Model 1: Win ~ Relative group size")
    print(res1.summary())
    print()

    # Model 2: relative size + contest location
    X2 = sm.add_constant(df[["rel_size", "focal_closer"]])
    model2 = sm.GLM(y, X2, family=sm.families.Binomial())
    res2 = model2.fit(cov_type="cluster", cov_kwds={"groups": df["feature3"]})

    print("Model 2: Win ~ Relative group size + Focal closer to home center")
    print(res2.summary())
    print()

    # Model 3: add interaction between size and location
    X3 = sm.add_constant(df[["rel_size", "focal_closer", "rel_size_x_focal_closer"]])
    model3 = sm.GLM(y, X3, family=sm.families.Binomial())
    res3 = model3.fit(cov_type="cluster", cov_kwds={"groups": df["feature3"]})

    print(
        "Model 3: Win ~ Relative group size + Focal closer + "
        "Interaction(rel_size x focal_closer)"
    )
    print(res3.summary())
    print()

    # Predicted probabilities for interpretation using Model 2
    mean_rel_size = df["rel_size"].mean()
    grid = pd.DataFrame(
        {
            "const": 1.0,
            "rel_size": [
                mean_rel_size - 3,
                mean_rel_size,
                mean_rel_size + 3,
            ],
            "focal_closer": [0, 0, 0],
        }
    )
    grid2 = grid.copy()
    grid2["focal_closer"] = 1

    preds_far = res2.predict(grid)
    preds_home = res2.predict(grid2)

    print("Predicted win probabilities (Model 2):")
    print("Rows: smaller group, average, larger group; first away, then home.")
    print("Focal AWAY from home center:", np.round(preds_far.values, 3))
    print("Focal AT/CLOSER to home center:", np.round(preds_home.values, 3))


if __name__ == "__main__":
    main()

