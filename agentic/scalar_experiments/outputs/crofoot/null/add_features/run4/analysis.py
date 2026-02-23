import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors
    df["size_diff"] = df["n_focal"] - df["n_other"]
    df["larger_focal"] = (df["size_diff"] > 0).astype(int)

    # Positive dist_diff means focal group is closer to its home range center
    df["dist_diff"] = df["dist_other"] - df["dist_focal"]
    df["home_adv"] = (df["dist_diff"] > 0).astype(int)

    # Basic descriptive summaries
    print("Number of contests:", len(df))
    print()

    print("Win rate overall:", df["win"].mean())
    print()

    print("Win rate by focal larger (larger_focal):")
    print(df.groupby("larger_focal")["win"].mean())
    print()

    print("Win rate by home advantage (home_adv):")
    print(df.groupby("home_adv")["win"].mean())
    print()

    # Logistic regression with binary predictors
    model_bin = smf.logit("win ~ larger_focal + home_adv", data=df).fit(disp=False)
    print("Logistic regression with binary predictors (larger_focal, home_adv):")
    print(model_bin.summary())
    print()

    # Logistic regression with continuous differences
    model_cont = smf.logit("win ~ size_diff + dist_diff", data=df).fit(disp=False)
    print("Logistic regression with continuous predictors (size_diff, dist_diff):")
    print(model_cont.summary())
    print()

    # Report odds ratios and p-values for easier interpretation
    def report_effects(model, label: str) -> None:
        params = model.params
        conf = model.conf_int()
        pvalues = model.pvalues

        print(f"Effect summary for {label}:")
        print("term, coef, odds_ratio, ci_low, ci_high, pvalue")
        for term in params.index:
            or_val = np.exp(params[term])
            ci_low = np.exp(conf.loc[term, 0])
            ci_high = np.exp(conf.loc[term, 1])
            print(
                f"{term}, {params[term]:.3f}, {or_val:.3f}, "
                f"{ci_low:.3f}, {ci_high:.3f}, {pvalues[term]:.3g}"
            )
        print()

    report_effects(model_bin, "binary predictors")
    report_effects(model_cont, "continuous predictors")


if __name__ == "__main__":
    main()

