import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    df = df.rename(
        columns={
            "feature1": "tooth_class",
            "feature2": "specimen_id",
            "feature3": "missing_teeth",
            "feature4": "observable_sockets",
            "feature5": "age",
            "feature6": "age_uncertainty",
            "feature7": "sex_estimate",
            "feature8": "genus",
            "feature9": "region",
        }
    )

    # Proportion of missing teeth in this tooth class
    df["prop_missing"] = df["missing_teeth"] / df["observable_sockets"]

    # Fit a binomial GLM with logit link:
    # response: proportion missing with observable_sockets as binomial trials
    # predictors: genus (categorical), age, sex_estimate, tooth_class (categorical)
    model = smf.glm(
        formula="prop_missing ~ C(genus) + age + sex_estimate + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["observable_sockets"],
    ).fit()

    print("=== Genus coefficients (reference: Homo sapiens) ===")
    for name, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
        if name.startswith("C(genus)"):
            print(f"{name}: coef={coef:.3f}, p={pval:.4g}")

    # Compute average predicted AMTL probability for each genus, holding age/sex/tooth_class
    # at their observed values (marginalizing over the sample).
    genera = sorted(df["genus"].unique())
    print("\n=== Average predicted AMTL probability by genus (covariate-adjusted) ===")
    for g in genera:
        df_g = df.copy()
        df_g["genus"] = g
        preds = model.predict(df_g)
        mean_pred = preds.mean()
        print(f"{g}: {mean_pred:.4f}")

    # Direct test: humans vs all non-human primates combined
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    human_model = smf.glm(
        formula="prop_missing ~ is_human + age + sex_estimate + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["observable_sockets"],
    ).fit()

    coef_human = human_model.params["is_human"]
    pval_human = human_model.pvalues["is_human"]
    print(
        "\n=== Humans vs non-human primates (combined) ===\n"
        f"is_human coefficient (log-odds): {coef_human:.3f}, p={pval_human:.4g}"
    )

    df_human = df.copy()
    df_human["is_human"] = 1
    df_nonhuman = df.copy()
    df_nonhuman["is_human"] = 0
    human_pred = human_model.predict(df_human).mean()
    nonhuman_pred = human_model.predict(df_nonhuman).mean()
    print(
        f"Average predicted AMTL probability - humans: {human_pred:.4f}, "
        f"non-human primates: {nonhuman_pred:.4f}"
    )


if __name__ == "__main__":
    main()
