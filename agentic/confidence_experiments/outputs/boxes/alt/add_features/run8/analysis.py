import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


def lr_test(full_model, reduced_model, df_diff: int):
    """Likelihood ratio test comparing a full and reduced model."""
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    p_value = chi2.sf(lr_stat, df=df_diff)
    return float(lr_stat), float(p_value)


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Derived outcomes
    df["social_choice"] = (df["y"] != 1).astype(int)
    df_social = df.copy()
    df_social["age_c"] = df_social["age"] - df_social["age"].mean()

    # Logistic models for reliance on social information (any demonstrated option vs undemonstrated)
    social_full = smf.logit(
        "social_choice ~ age_c + C(culture) + C(gender) + majority_first",
        data=df_social,
    ).fit(disp=False)

    social_no_age = smf.logit(
        "social_choice ~ C(culture) + C(gender) + majority_first",
        data=df_social,
    ).fit(disp=False)

    social_no_culture = smf.logit(
        "social_choice ~ age_c + C(gender) + majority_first",
        data=df_social,
    ).fit(disp=False)

    lr_age_social, p_age_social = lr_test(social_full, social_no_age, df_diff=1)
    df_culture_social = int(social_full.df_model - social_no_culture.df_model)
    lr_culture_social, p_culture_social = lr_test(
        social_full, social_no_culture, df_diff=df_culture_social
    )

    # Restrict to children who chose a demonstrated option and model majority vs minority preference
    df_majority = df[df["y"] != 1].copy()
    df_majority["majority_choice"] = (df_majority["y"] == 2).astype(int)
    df_majority["age_c"] = df_majority["age"] - df_majority["age"].mean()

    majority_full = smf.logit(
        "majority_choice ~ age_c + C(culture) + C(gender) + majority_first",
        data=df_majority,
    ).fit(disp=False)

    majority_no_age = smf.logit(
        "majority_choice ~ C(culture) + C(gender) + majority_first",
        data=df_majority,
    ).fit(disp=False)

    majority_no_culture = smf.logit(
        "majority_choice ~ age_c + C(gender) + majority_first",
        data=df_majority,
    ).fit(disp=False)

    lr_age_majority, p_age_majority = lr_test(majority_full, majority_no_age, df_diff=1)
    df_culture_majority = int(majority_full.df_model - majority_no_culture.df_model)
    lr_culture_majority, p_culture_majority = lr_test(
        majority_full, majority_no_culture, df_diff=df_culture_majority
    )

    # Effect size summaries: predicted probabilities across age and cultures
    def predict_social(age_value: float, culture_value: int) -> float:
        mean_age = df_social["age"].mean()
        new = pd.DataFrame(
            {
                "age_c": [age_value - mean_age],
                "culture": [culture_value],
                "gender": [1],
                "majority_first": [1],
            }
        )
        return float(social_full.predict(new)[0])

    def predict_majority(age_value: float, culture_value: int) -> float:
        mean_age = df_majority["age"].mean()
        new = pd.DataFrame(
            {
                "age_c": [age_value - mean_age],
                "culture": [culture_value],
                "gender": [1],
                "majority_first": [1],
            }
        )
        return float(majority_full.predict(new)[0])

    age_p10 = float(df["age"].quantile(0.1))
    age_p90 = float(df["age"].quantile(0.9))

    social_prob_young = predict_social(age_p10, culture_value=1)
    social_prob_old = predict_social(age_p90, culture_value=1)

    majority_prob_young = predict_majority(age_p10, culture_value=1)
    majority_prob_old = predict_majority(age_p90, culture_value=1)

    cultures = sorted(df["culture"].unique())
    social_probs_by_culture = {
        int(c): predict_social(df["age"].mean(), culture_value=int(c)) for c in cultures
    }
    majority_probs_by_culture = {
        int(c): predict_majority(df["age"].mean(), culture_value=int(c))
        for c in cultures
    }

    print("Logistic regression LR tests")
    print("----------------------------------------")
    print(f"Reliance on social information (any demonstration vs undemonstrated):")
    print(f"  Age effect:     LR = {lr_age_social:.3f}, p = {p_age_social:.5f}")
    print(
        f"  Culture effect: LR = {lr_culture_social:.3f}, df = {df_culture_social}, "
        f"p = {p_culture_social:.5f}"
    )
    print()
    print("Preference for majority vs minority (among social learners):")
    print(f"  Age effect:     LR = {lr_age_majority:.3f}, p = {p_age_majority:.5f}")
    print(
        f"  Culture effect: LR = {lr_culture_majority:.3f}, df = {df_culture_majority}, "
        f"p = {p_culture_majority:.5f}"
    )
    print()
    print("Approximate effect sizes (predicted probabilities)")
    print("----------------------------------------")
    print(
        f"Reliance on social information (culture 1, gender=girl, majority_first=1): "
        f"P(social | young age={age_p10:.1f}) = {social_prob_young:.3f}, "
        f"P(social | old age={age_p90:.1f}) = {social_prob_old:.3f}"
    )
    print(
        f"Majority preference (among social learners, culture 1): "
        f"P(majority | young age={age_p10:.1f}) = {majority_prob_young:.3f}, "
        f"P(majority | old age={age_p90:.1f}) = {majority_prob_old:.3f}"
    )
    print()
    print("Predicted reliance on social information by culture (at mean age):")
    for c, p in social_probs_by_culture.items():
        print(f"  Culture {c}: P(social) = {p:.3f}")
    print()
    print("Predicted majority preference by culture (at mean age, among social learners):")
    for c, p in majority_probs_by_culture.items():
        print(f"  Culture {c}: P(majority) = {p:.3f}")


if __name__ == "__main__":
    main()

