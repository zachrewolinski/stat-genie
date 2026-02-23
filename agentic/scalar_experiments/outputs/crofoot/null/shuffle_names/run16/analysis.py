import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Rename columns to more meaningful aliases based on info.json descriptions
    df = df.rename(
        columns={
            "m_focal": "win_focal",          # 1 if focal won, 0 otherwise
            "m_other": "dist_focal_home",    # distance of focal group from its home-range center
            "n_focal": "dist_other_home",    # distance of other group from its home-range center
            "f_other": "size_focal",         # number of individuals in focal group
            "win": "size_other",             # number of individuals in other group
        }
    )

    # Construct relative predictors
    df["rel_size"] = df["size_focal"] - df["size_other"]
    df["home_advantage"] = df["dist_other_home"] - df["dist_focal_home"]

    print("N rows:", len(df))
    print("Win focal value counts:")
    print(df["win_focal"].value_counts(dropna=False))
    print()

    print("Relative size (focal - other) summary:")
    print(df["rel_size"].describe())
    print()

    print("Home advantage (other_dist - focal_dist) summary:")
    print(df["home_advantage"].describe())
    print()

    # Standardize predictors to help interpret coefficients
    for col in ["rel_size", "home_advantage"]:
        df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

    # Logistic regression: probability focal group wins
    formula = "win_focal ~ z_rel_size + z_home_advantage"
    model = smf.logit(formula=formula, data=df)
    result = model.fit(disp=False)

    print("Logit model summary (win_focal ~ rel_size + home_advantage):")
    print(result.summary())
    print()

    # Also inspect univariate effects
    for col in ["z_rel_size", "z_home_advantage"]:
        print(f"Univariate logit for {col}:")
        m_uni = smf.logit(formula=f"win_focal ~ {col}", data=df).fit(disp=False)
        print(m_uni.summary())
        print()


if __name__ == "__main__":
    main()

