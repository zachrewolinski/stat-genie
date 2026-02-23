import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Create derived outcomes
    df["social"] = (df["y"] != 1).astype(int)
    df_social = df.copy()

    print("=== Descriptive statistics ===")
    print("Total N:", len(df))
    print("\nOutcome counts (1=undemonstrated, 2=majority, 3=minority):")
    print(df["y"].value_counts().sort_index())

    print("\nProportion choosing any demonstrated option (social reliance) by culture:")
    social_by_culture = df.groupby("culture")["social"].mean()
    print(social_by_culture)

    print("\nProportion choosing any demonstrated option (social reliance) by age:")
    social_by_age = df.groupby("age")["social"].mean()
    print(social_by_age)

    # Majority preference among those who chose a demonstrated option
    df_major = df[df["y"].isin([2, 3])].copy()
    df_major["majority"] = (df_major["y"] == 2).astype(int)

    print("\nProportion choosing majority option among social choosers by culture:")
    maj_by_culture = df_major.groupby("culture")["majority"].mean()
    print(maj_by_culture)

    print("\nProportion choosing majority option among social choosers by age:")
    maj_by_age = df_major.groupby("age")["majority"].mean()
    print(maj_by_age)

    # Logistic regression: social reliance ~ age + culture + gender + majority_first
    print("\n=== Logistic regression: social reliance (full model) ===")
    try:
        model_social = smf.logit(
            "social ~ age + C(culture) + gender + majority_first", data=df_social
        ).fit(disp=False, method="lbfgs", maxiter=1000)
        print(model_social.summary())
        print("\nSocial reliance pseudo R^2:", model_social.prsquared)
    except Exception as exc:  # pragma: no cover - debug fallback
        print("Social reliance model failed to converge:", exc)

    # Logistic regression: majority preference among social choosers
    print(
        "\n=== Logistic regression: majority preference (social choosers only, full model) ==="
    )
    try:
        model_major = smf.logit(
            "majority ~ age + C(culture) + gender + majority_first", data=df_major
        ).fit(disp=False, method="lbfgs", maxiter=1000)
        print(model_major.summary())
        print("\nMajority preference pseudo R^2:", model_major.prsquared)
    except Exception as exc:  # pragma: no cover - debug fallback
        print("Majority preference model failed to converge:", exc)

    # Reduced models focusing on culture and age only
    print(
        "\n=== Logistic regression: social reliance ~ age + C(culture) (reduced model) ==="
    )
    try:
        model_social_reduced = smf.logit(
            "social ~ age + C(culture)", data=df_social
        ).fit(disp=False, method="lbfgs", maxiter=1000)
        print(model_social_reduced.summary())
        print("\nSocial reliance (reduced) pseudo R^2:", model_social_reduced.prsquared)
    except Exception as exc:  # pragma: no cover - debug fallback
        print("Reduced social reliance model failed to converge:", exc)

    print(
        "\n=== Logistic regression: majority preference ~ age + C(culture) (reduced model) ==="
    )
    try:
        model_major_reduced = smf.logit(
            "majority ~ age + C(culture)", data=df_major
        ).fit(disp=False, method="lbfgs", maxiter=1000)
        print(model_major_reduced.summary())
        print(
            "\nMajority preference (reduced) pseudo R^2:",
            model_major_reduced.prsquared,
        )
    except Exception as exc:  # pragma: no cover - debug fallback
        print("Reduced majority preference model failed to converge:", exc)


if __name__ == "__main__":
    main()
