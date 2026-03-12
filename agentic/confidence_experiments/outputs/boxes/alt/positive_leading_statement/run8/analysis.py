import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Reliance on social information: 1 if majority/minority (y in {2,3}), 0 if undemonstrated option (y==1)
    df["follow_social"] = (df["y"] != 1).astype(int)

    # Majority preference among those who follow social information
    df_social = df[df["follow_social"] == 1].copy()
    df_social["prefer_majority"] = (df_social["y"] == 2).astype(int)

    print("=== Basic counts ===")
    print("N total:", len(df))
    print("Proportion follow social (overall):", df["follow_social"].mean())
    print("Proportion prefer majority among social followers (overall):", df_social["prefer_majority"].mean())
    print()

    print("=== Follow social by culture ===")
    print(df.groupby("culture")["follow_social"].agg(["mean", "count"]))
    print()

    print("=== Follow social by age group ===")
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ["4-6", "7-9", "10-12", "13-14"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    print(df.groupby("age_group")["follow_social"].agg(["mean", "count"]))
    print()

    print("=== Majority preference by culture (among followers) ===")
    print(df_social.groupby("culture")["prefer_majority"].agg(["mean", "count"]))
    print()

    print("=== Majority preference by age group (among followers) ===")
    df_social["age_group"] = pd.cut(df_social["age"], bins=bins, labels=labels)
    print(df_social.groupby("age_group")["prefer_majority"].agg(["mean", "count"]))
    print()

    print("=== Logistic regression: follow_social ~ age + culture + gender + majority_first ===")
    model_follow = smf.logit(
        "follow_social ~ age + C(culture) + gender + majority_first",
        data=df,
    ).fit(disp=False)
    print(model_follow.summary())
    print("Odds ratios:")
    print(np.exp(model_follow.params))
    print()

    print("=== Logistic regression: prefer_majority ~ age + culture + gender + majority_first (social followers only) ===")
    model_pref = smf.logit(
        "prefer_majority ~ age + C(culture) + gender + majority_first",
        data=df_social,
    ).fit(disp=False)
    print(model_pref.summary())
    print("Odds ratios:")
    print(np.exp(model_pref.params))


if __name__ == "__main__":
    main()

