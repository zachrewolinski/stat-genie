import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    assert (df["sockets"] > 0).all()

    # Proportion of antemortem tooth loss per row
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Quick descriptive statistics by genus
    desc = (
        df.groupby("genus")["prop_amtl"]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_index()
    )
    print("Descriptive AMTL proportions by genus:")
    print(desc.to_string(float_format=lambda x: f"{x:0.3f}"))
    print()

    # Treat Homo sapiens as the baseline category (if present)
    df["genus"] = df["genus"].astype("category")
    if "Homo sapiens" in df["genus"].cat.categories:
        df["genus"] = df["genus"].cat.reorder_categories(
            ["Homo sapiens"] + [g for g in df["genus"].cat.categories if g != "Homo sapiens"],
            ordered=False,
        )

    df["tooth_class"] = df["tooth_class"].astype("category")

    # Binomial regression: proportion missing with sockets as trial weights
    formula = "prop_amtl ~ genus + age + prob_male + tooth_class"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("Binomial regression results:")
    print(result.summary())
    print()

    # Predicted AMTL probabilities for each genus at reference covariates
    ref_age = df["age"].mean()
    ref_prob_male = df["prob_male"].mean()
    ref_tooth_class = df["tooth_class"].mode().iloc[0]

    pred_rows = []
    for genus in df["genus"].cat.categories:
        pred_rows.append(
            {
                "genus": genus,
                "age": ref_age,
                "prob_male": ref_prob_male,
                "tooth_class": ref_tooth_class,
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    pred_probs = result.predict(pred_df)

    print("Predicted AMTL probabilities at reference covariates:")
    for genus, p in zip(pred_df["genus"], pred_probs):
        print(f"{genus}: {p:0.3f}")

    print("\n---\n")
    print("Collapsing genera into human vs non-human:")

    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    formula2 = "prop_amtl ~ is_human + age + prob_male + tooth_class"
    model2 = smf.glm(
        formula=formula2,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result2 = model2.fit()

    print("Binomial regression (human vs non-human) results:")
    print(result2.summary())
    print()

    ref_rows2 = []
    for is_human in [0, 1]:
        ref_rows2.append(
            {
                "is_human": is_human,
                "age": ref_age,
                "prob_male": ref_prob_male,
                "tooth_class": ref_tooth_class,
            }
        )

    pred_df2 = pd.DataFrame(ref_rows2)
    pred_probs2 = result2.predict(pred_df2)

    labels = {0: "Non-human primates", 1: "Homo sapiens"}
    print("Predicted AMTL probabilities at reference covariates (human vs non-human):")
    for is_human, p in zip(pred_df2["is_human"], pred_probs2):
        print(f"{labels[is_human]}: {p:0.3f}")


if __name__ == "__main__":
    main()
