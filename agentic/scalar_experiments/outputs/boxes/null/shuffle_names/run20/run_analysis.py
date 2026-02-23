import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Recode outcomes
    df["choice_social"] = (df["majority_first"] != 1).astype(int)
    df["choice_majority"] = (df["majority_first"] == 2).astype(int)

    # Treat site as categorical proxy for culture
    df["site"] = df["y"].astype("category")

    print("=== Basic proportions ===")
    overall = df[["choice_social", "choice_majority"]].mean()
    print(overall)

    print("\n=== Proportions by site (culture proxy) ===")
    by_site = (
        df.groupby("site")[["choice_social", "choice_majority"]]
        .mean()
        .assign(n=df.groupby("site")["choice_social"].size())
    )
    print(by_site)

    print("\n=== Proportions by age (integer years) ===")
    by_age = (
        df.groupby("age")[["choice_social", "choice_majority"]]
        .mean()
        .assign(n=df.groupby("age")["choice_social"].size())
    )
    print(by_age)

    # Logistic regression: social information use ~ age + site + gender + demo order
    print("\n=== Logistic regression: choice_social ~ age + C(site) + C(gender) + culture ===")
    model_social = smf.logit(
        "choice_social ~ age + C(site) + C(gender) + culture", data=df
    ).fit(disp=False)
    print(model_social.summary())

    # Logistic regression: majority preference ~ age + site + gender + demo order
    print(
        "\n=== Logistic regression: choice_majority ~ age + C(site) + C(gender) + culture ==="
    )
    model_majority = smf.logit(
        "choice_majority ~ age + C(site) + C(gender) + culture", data=df
    ).fit(disp=False)
    print(model_majority.summary())

    # Simple LR tests for overall site and age effects in majority choice
    print("\n=== Likelihood-ratio tests for majority choice models ===")
    full = model_majority
    # Drop site
    model_no_site = smf.logit(
        "choice_majority ~ age + C(gender) + culture", data=df
    ).fit(disp=False)
    lr_site = 2 * (full.llf - model_no_site.llf)
    df_site = full.df_model - model_no_site.df_model
    print(f"LR test for site effect: LR={lr_site:.2f}, df={df_site}")

    # Drop age
    model_no_age = smf.logit(
        "choice_majority ~ C(site) + C(gender) + culture", data=df
    ).fit(disp=False)
    lr_age = 2 * (full.llf - model_no_age.llf)
    df_age = full.df_model - model_no_age.df_model
    print(f"LR test for age effect: LR={lr_age:.2f}, df={df_age}")


if __name__ == "__main__":
    main()

