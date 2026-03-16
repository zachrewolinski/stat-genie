import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    print("=== Dataset overview ===")
    print(df.head())
    print("\nGenus counts:")
    print(df["genus"].value_counts())

    # Create proportion of antemortem tooth loss
    df = df[df["sockets"] > 0].copy()
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Ensure categorical types
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    print("\nGenus categories:", df["genus"].cat.categories.tolist())
    print("Tooth_class categories:", df["tooth_class"].cat.categories.tolist())

    # Binomial regression (logit link) with genus, age, sex proxy, and tooth class
    formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print("\n=== Binomial GLM with genus ===")
    print(result.summary())

    # Extract genus effects relative to Homo sapiens baseline (expected)
    params = result.params
    pvalues = result.pvalues

    print("\n=== Genus effects (log-odds relative to Homo sapiens baseline) ===")
    for name, coef in params.items():
        if name.startswith("C(genus)[T."):
            print(f"{name}: coef={coef:.3f}, p-value={pvalues[name]:.4g}")

    # Predicted AMTL probabilities by genus at mean age/sex and Posterior tooth class
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()

    print("\n=== Predicted AMTL probability by genus (Posterior teeth, mean age/sex) ===")
    genera = df["genus"].cat.categories
    pred_rows = []
    for g in genera:
        row = {
            "prop_amtl": 0.0,  # placeholder, not used directly
            "genus": g,
            "age": mean_age,
            "prob_male": mean_prob_male,
            "tooth_class": "Posterior",
            "sockets": 1.0,
        }
        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_probs = result.predict(pred_df)
    for g, p in zip(genera, pred_probs):
        print(f"{g}: predicted AMTL probability = {p:.3f}")

    # Direct human vs non-human contrast
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    formula_human = "prop_amtl ~ is_human + age + prob_male + C(tooth_class)"
    model_human = smf.glm(
        formula=formula_human,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result_human = model_human.fit()

    print("\n=== Binomial GLM with is_human indicator ===")
    print(result_human.summary())

    coef_human = result_human.params["is_human"]
    p_human = result_human.pvalues["is_human"]
    print(
        f"\nis_human coefficient (log-odds, human vs non-human): {coef_human:.3f}, p-value={p_human:.4g}"
    )



if __name__ == "__main__":
    main()
