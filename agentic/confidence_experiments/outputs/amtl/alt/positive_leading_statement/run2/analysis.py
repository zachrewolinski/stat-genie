import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Create proportion of antemortem tooth loss and binomial weights
    df = df.copy()
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Drop any rows with non-finite values just in case
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["amtl_prop", "sockets", "age", "prob_male", "tooth_class", "is_human"]
    )

    model = smf.glm(
        formula="amtl_prop ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    result = model.fit()

    print(result.summary())

    # Compute predicted AMTL probabilities for a typical non-human and human individual
    typical_age = df["age"].median()
    typical_prob_male = df["prob_male"].mean()
    ref_tooth_class = df["tooth_class"].mode().iat[0]

    base = pd.DataFrame(
        {
            "is_human": [0, 1],
            "age": [typical_age, typical_age],
            "prob_male": [typical_prob_male, typical_prob_male],
            "tooth_class": [ref_tooth_class, ref_tooth_class],
        }
    )

    preds = result.get_prediction(base).summary_frame()
    print("\nPredicted AMTL proportions (typical individual):")
    for label, is_human in zip(["Non-human primate", "Modern human"], [0, 1]):
        row = preds.loc[base["is_human"] == is_human].iloc[0]
        print(
            f"{label}: mean={row['mean']:.3f}, "
            f"2.5%={row['mean_ci_lower']:.3f}, 97.5%={row['mean_ci_upper']:.3f}"
        )


if __name__ == "__main__":
    main()

