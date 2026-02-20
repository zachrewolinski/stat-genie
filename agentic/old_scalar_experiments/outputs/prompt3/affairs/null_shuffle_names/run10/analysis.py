import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # According to info.json, the variable names are shuffled.
    # Mapping from column names to their semantic meaning:
    # - age: frequency of extramarital intercourse in past year (0 = none, >0 = some)
    # - religiousness: "yes"/"no" indicating whether there are children in the marriage
    # Other columns are used as controls but are not central to the research question.

    df = df.copy()
    df["affair_freq"] = df["age"]
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Sanity check: drop any unexpected / missing encodings
    df = df.dropna(subset=["affair_freq", "has_children"])

    # Descriptive statistics by children status
    group_stats = (
        df.groupby("has_children")
        .agg(
            mean_affair_freq=("affair_freq", "mean"),
            prop_any_affair=("any_affair", "mean"),
            count=("any_affair", "size"),
        )
        .reset_index()
    )

    print("Descriptive statistics by children status (has_children=1 means children present):")
    print(group_stats.to_string(index=False))
    print()

    # Logistic regression for having any affair, controlling for key covariates.
    # Map other shuffled columns to more interpretable names using info.json descriptions.
    df["age_years"] = df["occupation"]  # coded age categories
    df["years_married"] = df["children"]  # coded years married
    df["religiousness_score"] = df["rating"]  # 1-5 religiousness
    df["education_years"] = df["yearsmarried"]  # 9-20 education
    df["occupation_code"] = df["rownames"]  # 1-7 occupation
    df["marriage_rating"] = df["affairs"]  # 1-5 self rating of marriage

    formula = (
        "any_affair ~ has_children + age_years + years_married + "
        "religiousness_score + education_years + occupation_code + marriage_rating"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)

    print("Logistic regression for any affair ~ children + controls")
    print(logit_model.summary())
    print()

    # Extract and print the key coefficient for children
    coef = logit_model.params["has_children"]
    se = logit_model.bse["has_children"]
    pval = logit_model.pvalues["has_children"]
    odds_ratio = np.exp(coef)

    print("Effect of having children on odds of any affair:")
    print(f"  Coefficient (log-odds): {coef:.3f}")
    print(f"  Std. error:             {se:.3f}")
    print(f"  p-value:                {pval:.4f}")
    print(f"  Odds ratio:             {odds_ratio:.3f}")


if __name__ == "__main__":
    main()

