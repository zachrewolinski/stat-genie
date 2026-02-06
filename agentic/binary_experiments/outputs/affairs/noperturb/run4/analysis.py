import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("affairs.csv")

    # Basic indicators
    df["has_affair"] = (df["affairs"] > 0).astype(int)
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)

    # Descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            n=("affairs", "size"),
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            any_affair_rate=("has_affair", "mean"),
        )
        .reset_index()
    )

    # Unadjusted difference in means (affairs count)
    ols_unadj = smf.ols("affairs ~ children_yes", data=df).fit()

    # Unadjusted difference in any-affair rate (linear probability model)
    lpm_unadj = smf.ols("has_affair ~ children_yes", data=df).fit()

    # Adjusted models with controls
    controls = "age + yearsmarried + religiousness + education + occupation + rating + C(gender)"
    ols_adj = smf.ols(f"affairs ~ children_yes + {controls}", data=df).fit()
    lpm_adj = smf.ols(f"has_affair ~ children_yes + {controls}", data=df).fit()

    print("Descriptive statistics by children status:")
    print(desc.to_string(index=False))
    print("\nUnadjusted OLS (affairs ~ children_yes):")
    print(ols_unadj.summary().tables[1])
    print("\nUnadjusted LPM (has_affair ~ children_yes):")
    print(lpm_unadj.summary().tables[1])
    print("\nAdjusted OLS (affairs ~ children_yes + controls):")
    print(ols_adj.summary().tables[1])
    print("\nAdjusted LPM (has_affair ~ children_yes + controls):")
    print(lpm_adj.summary().tables[1])


if __name__ == "__main__":
    main()
