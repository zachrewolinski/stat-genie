import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Relative group size metrics
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["rel_size"] = df["n_focal"] / df["n_other"]

    # Contest location metrics: positive values favor focal group
    df["loc_diff"] = df["dist_other"] - df["dist_focal"]
    df["focal_closer"] = (df["dist_focal"] < df["dist_other"]).astype(int)

    print("N rows:", len(df))
    print("\nWin rate overall:", df["win"].mean())

    # Descriptive summaries for relative group size
    df["focal_larger"] = (df["size_diff"] > 0).astype(int)
    print("\nWin rate by focal larger vs not:")
    print(df.groupby("focal_larger")["win"].mean())

    # Descriptive summaries for contest location
    print("\nWin rate by focal closer to own center:")
    print(df.groupby("focal_closer")["win"].mean())

    # Logistic regression: probability focal wins as a function of
    # relative group size and relative location
    formulas = {
        "logit_size_loc": "win ~ size_diff + loc_diff",
        "logit_rel_size_loc": "win ~ rel_size + loc_diff",
        "logit_size_only": "win ~ size_diff",
        "logit_loc_only": "win ~ loc_diff",
        "logit_binary_cats": "win ~ focal_larger + focal_closer",
    }

    for name, formula in formulas.items():
        print(f"\n=== {name} ===")
        try:
            model = smf.logit(formula, data=df).fit(disp=False)
            print(model.summary())
            print("\nOdds ratios and 95% CI:")
            params = model.params
            conf = model.conf_int()
            or_df = np.exp(
                pd.DataFrame(
                    {
                        "OR": params,
                        "CI_lower": conf[0],
                        "CI_upper": conf[1],
                    }
                )
            )
            print(or_df)
        except Exception as exc:
            print(f"Model {name} failed to fit: {exc}")


if __name__ == "__main__":
    main()
