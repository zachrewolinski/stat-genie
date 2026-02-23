import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Outcome encodings
    df["majority_choice"] = (df["majority_first"] == 2).astype(int)
    df["social_choice"] = (df["majority_first"] != 1).astype(int)
    df["site"] = df["y"].astype("category")

    df_mm = df[df["majority_first"].isin([2, 3])].copy()
    df_mm["majority_vs_minority"] = (df_mm["majority_first"] == 2).astype(int)
    df_mm["site"] = df_mm["y"].astype("category")

    print("N total:", len(df))
    print("Proportion majority choice:", df["majority_choice"].mean())
    print("Proportion any social choice:", df["social_choice"].mean())
    print("N majority/minority-only subset:", len(df_mm))
    print("Proportion majority within social users:", df_mm["majority_vs_minority"].mean())
    print()

    # Social reliance: any demonstrated option vs undemonstrated
    mod_social = smf.glm(
        "social_choice ~ age + C(site) + gender + culture",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print("=== Social choice (any demonstration vs undemonstrated) ===")
    print(mod_social.summary())
    print()

    # Majority vs all other
    mod_major = smf.glm(
        "majority_choice ~ age + C(site) + gender + culture",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print("=== Majority vs non-majority (all children) ===")
    print(mod_major.summary())
    print()

    # Majority vs minority among social choosers
    mod_mm = smf.glm(
        "majority_vs_minority ~ age + C(site) + gender + culture",
        data=df_mm,
        family=sm.families.Binomial(),
    ).fit()
    print("=== Majority vs minority (conditional on social choice) ===")
    print(mod_mm.summary())


if __name__ == "__main__":
    main()

