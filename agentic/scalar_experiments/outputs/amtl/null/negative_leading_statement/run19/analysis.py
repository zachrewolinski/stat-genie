import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Basic summaries of AMTL proportion by genus
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]
    genus_summary = (
        df.groupby("genus")["amtl_prop"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print("Raw AMTL proportion by genus:")
    print(genus_summary.to_string(index=False))
    print()

    # Binomial regression with Homo sapiens as reference genus,
    # adjusting for age, sex (prob_male), and tooth_class.
    formula = (
        "num_amtl ~ C(genus, Treatment(reference='Homo sapiens'))"
        " + age + prob_male + C(tooth_class)"
    )
    # Statsmodels expects the response as a proportion with the number
    # of trials passed in via freq_weights for binomial GLM.
    model = smf.glm(
        formula=formula.replace("num_amtl", "amtl_prop"),
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print("Binomial GLM results (logit link):")
    print(model.summary())
    print()

    # Marginal predicted AMTL probabilities by genus,
    # averaging over the observed distribution of covariates.
    def marginal_prob_for_genus(target_genus: str) -> float:
        tmp = df.copy()
        tmp["genus"] = target_genus
        return float(model.predict(tmp).mean())

    genera = sorted(df["genus"].unique())
    print("Marginal predicted AMTL probabilities by genus:")
    for g in genera:
        prob = marginal_prob_for_genus(g)
        print(f"  {g:12s}: {prob:.4f}")

    # Additional model: humans vs all non-human primates combined.
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    formula_human = "amtl_prop ~ is_human + age + prob_male + C(tooth_class)"
    model_human = smf.glm(
        formula=formula_human,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    ).fit()

    print()
    print("Binomial GLM results (humans vs non-humans):")
    print(model_human.summary())

    def marginal_prob_is_human(flag: int) -> float:
        tmp = df.copy()
        tmp["is_human"] = flag
        return float(model_human.predict(tmp).mean())

    print("Marginal predicted AMTL probabilities by human status:")
    for flag, label in [(1, "Homo sapiens"), (0, "Non-human primates")]:
        prob = marginal_prob_is_human(flag)
        print(f"  {label:18s}: {prob:.4f}")


if __name__ == "__main__":
    main()
