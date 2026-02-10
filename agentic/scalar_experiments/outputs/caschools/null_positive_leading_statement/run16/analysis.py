import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio
    df["stratio"] = df["students"] / df["teachers"]

    # Outcome: average of reading and math scores
    df["avgscore"] = df[["read", "math"]].mean(axis=1)

    print("Rows, columns:", df.shape)
    print("\nCorrelation between STR and avgscore:")
    print(df["stratio"].corr(df["avgscore"]))

    # Simple bivariate regression
    X = sm.add_constant(df["stratio"])
    y = df["avgscore"]
    model_simple = sm.OLS(y, X).fit()
    print("\nSimple OLS: avgscore ~ STR")
    print(model_simple.summary())

    # Add key controls for confounding
    controls = ["income", "english", "calworks", "lunch", "computer", "expenditure"]
    Xc = sm.add_constant(df[["stratio"] + controls])
    model_ctrl = sm.OLS(y, Xc).fit()
    print("\nControlled OLS: avgscore ~ STR + controls")
    print(model_ctrl.summary())

    # Save a small summary of key coefficients
    summ = {
        "simple_stratio_coef": float(model_simple.params["stratio"]),
        "simple_stratio_p": float(model_simple.pvalues["stratio"]),
        "ctrl_stratio_coef": float(model_ctrl.params["stratio"]),
        "ctrl_stratio_p": float(model_ctrl.pvalues["stratio"]),
        "n_obs": int(model_ctrl.nobs),
        "r2_simple": float(model_simple.rsquared),
        "r2_ctrl": float(model_ctrl.rsquared),
    }
    print("\nKey summary:")
    for k, v in summ.items():
        print(f"{k}: {v}")

    # Also store to a JSON file for downstream mapping if desired
    import json

    with open("analysis_summary.json", "w") as f:
        json.dump(summ, f, indent=2)


if __name__ == "__main__":
    main()
