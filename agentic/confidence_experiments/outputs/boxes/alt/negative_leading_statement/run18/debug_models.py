import pandas as pd
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("boxes.csv")
    df["social_choice"] = df["y"].isin([2, 3]).astype(int)
    df["majority_choice"] = (df["y"] == 2).astype(int)
    df["demonstrated_choice"] = df["y"].isin([2, 3])
    df_demo = df[df["demonstrated_choice"]].copy()

    social_full = smf.logit(
        "social_choice ~ age + C(culture) + gender + majority_first", df
    ).fit(disp=0, maxiter=1000)
    social_no_culture = smf.logit(
        "social_choice ~ age + gender + majority_first", df
    ).fit(disp=0, maxiter=1000)
    social_no_age = smf.logit(
        "social_choice ~ C(culture) + gender + majority_first", df
    ).fit(disp=0, maxiter=1000)

    print("Social info: full vs no culture:", social_full.compare_lr_test(social_no_culture))
    print("Social info: full vs no age:", social_full.compare_lr_test(social_no_age))

    maj_full = smf.logit(
        "majority_choice ~ age + C(culture) + gender + majority_first", df_demo
    ).fit(disp=0, maxiter=1000)
    maj_no_culture = smf.logit(
        "majority_choice ~ age + gender + majority_first", df_demo
    ).fit(disp=0, maxiter=1000)
    maj_no_age = smf.logit(
        "majority_choice ~ C(culture) + gender + majority_first", df_demo
    ).fit(disp=0, maxiter=1000)

    print("Majority pref: full vs no culture:", maj_full.compare_lr_test(maj_no_culture))
    print("Majority pref: full vs no age:", maj_full.compare_lr_test(maj_no_age))


if __name__ == "__main__":
    main()

