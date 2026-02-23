import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Construct key predictors
    # Relative group size: focal minus other (positive means focal group larger)
    df["rel_size_diff"] = df["n_focal"] - df["n_other"]
    df["rel_size_ratio"] = df["n_focal"] / df["n_other"]

    # Contest location: difference in distance to each group's home-range center
    # dist_* is distance (m) from each group's own home-range center.
    # Positive loc_diff => focal is closer to its own center than the opponent is to theirs
    # (i.e., contest is more on the focal group's "home turf").
    df["loc_diff"] = df["dist_other"] - df["dist_focal"]

    print("N rows:", len(df))
    print("Win rate (focal):", df["win"].mean())
    print()

    # Simple univariable models
    print("Logit: win ~ rel_size_diff")
    m_size = smf.logit("win ~ rel_size_diff", data=df).fit(disp=False)
    print(m_size.summary())
    print()

    print("Logit: win ~ loc_diff")
    m_loc = smf.logit("win ~ loc_diff", data=df).fit(disp=False)
    print(m_loc.summary())
    print()

    # Multivariable model including both predictors
    print("Logit: win ~ rel_size_diff + loc_diff")
    m_both = smf.logit("win ~ rel_size_diff + loc_diff", data=df).fit(disp=False)
    print(m_both.summary())
    print()

    # Also report odds ratios for the joint model
    params = m_both.params
    conf = m_both.conf_int()
    or_table = pd.DataFrame(
        {
            "OR": params.apply(lambda x: float(sm.tools.tools._ensure_2d(pd.Series(x)).values[0]) if pd.notna(x) else float("nan")).pipe(
                lambda s: (s).apply(lambda v: float(pd.np.exp(v)))
            )
        }
    )
    or_table["2.5%"] = (conf[0]).apply(lambda v: float(pd.np.exp(v)))
    or_table["97.5%"] = (conf[1]).apply(lambda v: float(pd.np.exp(v)))
    print("Odds ratios (joint model):")
    print(or_table)


if __name__ == "__main__":
    main()

