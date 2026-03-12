import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # Basic description
    print("N rows:", len(df))
    print("Columns:", list(df.columns))
    print(df[["y", "age", "culture"]].describe(include="all"))

    # Encode outcomes
    df["used_social"] = (df["y"] != 1).astype(int)
    df_social = df.copy()

    # Logistic regression: reliance on social information (main effects)
    model_social = smf.logit("used_social ~ age + C(culture)", data=df_social).fit(disp=False)
    print("\n=== Reliance on social information (used_social) ===")
    print(model_social.summary())

    # With age × culture interaction
    model_social_int = smf.logit("used_social ~ age * C(culture)", data=df_social).fit(disp=False)
    print("\n--- With age × culture interaction (used_social) ---")
    print(model_social_int.summary2().tables[0])

    # Among those who used social information, examine majority preference
    df_majority = df[df["y"].isin([2, 3])].copy()
    df_majority["majority_choice"] = (df_majority["y"] == 2).astype(int)

    # Logistic regression: majority preference (main effects)
    model_majority = smf.logit("majority_choice ~ age + C(culture)", data=df_majority).fit(disp=False)
    print("\n=== Majority preference among social users (majority_choice) ===")
    print(model_majority.summary())

    # With age × culture interaction
    model_majority_int = smf.logit("majority_choice ~ age * C(culture)", data=df_majority).fit(disp=False)
    print("\n--- With age × culture interaction (majority_choice) ---")
    print(model_majority_int.summary2().tables[0])

    # Simple descriptive statistics by culture and age bands
    df["age_band"] = pd.qcut(df["age"], 4, labels=["Q1_youngest", "Q2", "Q3", "Q4_oldest"])
    print("\n=== Proportion using social information by culture ===")
    print(df.groupby("culture")["used_social"].mean())
    print("\n=== Proportion using social information by age band ===")
    print(df.groupby("age_band")["used_social"].mean())

    print("\n=== Majority preference among social users by culture ===")
    print(df_majority.groupby("culture")["majority_choice"].mean())
    df_majority["age_band"] = pd.qcut(df_majority["age"], 4, labels=["Q1_youngest", "Q2", "Q3", "Q4_oldest"])
    print("\n=== Majority preference among social users by age band ===")
    print(df_majority.groupby("age_band")["majority_choice"].mean())


if __name__ == "__main__":
    main()
