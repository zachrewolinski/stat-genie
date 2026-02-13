import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic derived variables
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nOverall AMTL rate by genus (num_amtl / sockets):")
    genus_rates = (
        df.groupby("genus")[["num_amtl", "sockets"]]
        .sum()
        .assign(rate=lambda g: g["num_amtl"] / g["sockets"])
    )
    print(genus_rates)

    # GLM with genus as categorical predictor
    print("\nGLM with C(genus) + age + prob_male + C(tooth_class)")
    glm_genus = smf.glm(
        formula="amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    res_genus_robust = glm_genus.fit(
        cov_type="cluster", cov_kwds={"groups": df["specimen"]}
    )
    print(res_genus_robust.summary())

    # GLM with human indicator vs all non-human primates
    print("\nGLM with is_human + age + prob_male + C(tooth_class)")
    glm_human = smf.glm(
        formula="amtl_rate ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    res_human_robust = glm_human.fit(
        cov_type="cluster", cov_kwds={"groups": df["specimen"]}
    )
    print(res_human_robust.summary())

    # Predicted AMTL probabilities by genus at typical covariate values
    age_median = df["age"].median()
    prob_male_mean = df["prob_male"].mean()
    tooth_mode = df["tooth_class"].mode().iat[0]

    print(
        "\nPredicted AMTL probability by genus at "
        f"age={age_median:.2f}, prob_male={prob_male_mean:.2f}, "
        f"tooth_class={tooth_mode}"
    )
    for genus in sorted(df["genus"].unique()):
        new = pd.DataFrame(
            {
                "genus": [genus],
                "age": [age_median],
                "prob_male": [prob_male_mean],
                "tooth_class": [tooth_mode],
                # placeholder column for formula
                "amtl_rate": [0.0],
            }
        )
        pred = res_genus_robust.predict(new)[0]
        print(f"{genus:12s}: {pred:.4f}")


if __name__ == "__main__":
    main()
