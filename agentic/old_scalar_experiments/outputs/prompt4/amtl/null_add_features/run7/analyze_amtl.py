import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")
    df["amtl_any"] = (df["num_amtl"] > 0).astype(int)
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    model_cols = ["amtl_any", "genus", "age", "prob_male", "tooth_class", "is_human"]
    df_model = df[model_cols].dropna()

    print(f"Number of rows used in model: {len(df_model)}")

    # Logistic regression with genus as categorical (baseline: Homo sapiens)
    logit_genus = smf.logit(
        "amtl_any ~ C(genus) + age + prob_male + C(tooth_class)", data=df_model
    ).fit(disp=False)

    print("\nLogistic regression with full genus categories")
    print(logit_genus.summary())

    # Logistic regression comparing humans vs all non-human primates
    logit_human = smf.logit(
        "amtl_any ~ is_human + age + prob_male + C(tooth_class)", data=df_model
    ).fit(disp=False)

    print("\nLogistic regression with human indicator")
    print(logit_human.summary())

    # Predicted probabilities at representative values
    rep_age = df_model["age"].median()
    rep_prob_male = df_model["prob_male"].mean()
    rep_tooth_class = df_model["tooth_class"].mode()[0]

    print(
        f"\nRepresentative values for predictions: age={rep_age:.2f}, "
        f"prob_male={rep_prob_male:.2f}, tooth_class={rep_tooth_class}"
    )

    genera = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    new_data_genus = pd.DataFrame(
        {
            "genus": genera,
            "age": rep_age,
            "prob_male": rep_prob_male,
            "tooth_class": rep_tooth_class,
        }
    )
    new_data_genus["amtl_any"] = 0  # placeholder, not used in prediction

    preds_genus = logit_genus.predict(new_data_genus)
    print("\nPredicted AMTL prevalence by genus (adjusted):")
    for g, p in zip(genera, preds_genus):
        print(f"  {g}: {p:.3f}")

    new_data_human = pd.DataFrame(
        {
            "is_human": [1, 0],
            "age": rep_age,
            "prob_male": rep_prob_male,
            "tooth_class": rep_tooth_class,
        }
    )
    new_data_human["amtl_any"] = 0  # placeholder

    preds_human = logit_human.predict(new_data_human)
    print(
        "\nPredicted AMTL prevalence at representative values "
        "(human vs non-human, adjusted):"
    )
    print(f"  Human:     {preds_human.iloc[0]:.3f}")
    print(f"  Non-human: {preds_human.iloc[1]:.3f}")

    coef = logit_human.params["is_human"]
    pval = logit_human.pvalues["is_human"]
    print(f"\nHuman indicator coefficient: {coef:.3f}, p-value={pval:.4g}")


if __name__ == "__main__":
    main()

