import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic derived rate
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    print("=== Basic description ===")
    print("Rows:", len(df))
    print(df[["num_amtl", "sockets", "age", "prob_male"]].describe())
    print("\nMean AMTL rate by genus:")
    print(df.groupby("genus")["amtl_rate"].mean())

    # Ensure Homo sapiens is the reference genus
    df["genus"] = df["genus"].astype("category")
    if "Homo sapiens" in df["genus"].cat.categories:
        df["genus"] = df["genus"].cat.reorder_categories(
            ["Homo sapiens"]
            + [g for g in df["genus"].cat.categories if g != "Homo sapiens"],
            ordered=False,
        )

    # Binomial regression of AMTL rate on genus, controlling for age, sex, and tooth class
    # Use sockets as frequency weights so the response is a proportion.
    formula = "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print("\n=== Binomial regression summary ===")
    print(model.summary())

    # Predicted AMTL probabilities for each genus at typical covariate values
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    # Use the most common tooth class as a reference level for predictions
    common_tooth_class = df["tooth_class"].mode()[0]

    pred_df = pd.DataFrame(
        {
            "genus": df["genus"].cat.categories,
            "age": mean_age,
            "prob_male": mean_prob_male,
            "tooth_class": common_tooth_class,
        }
    )
    pred_df["pred_amtl_prob"] = model.predict(pred_df)

    print("\n=== Predicted AMTL probabilities by genus ===")
    print(pred_df)


if __name__ == "__main__":
    main()

