import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Treat genus and tooth class as categorical and set Homo sapiens as reference.
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    if "Homo sapiens" in list(df["genus"].cat.categories):
        # Reorder so Homo sapiens is the baseline category.
        other = [g for g in df["genus"].cat.categories if g != "Homo sapiens"]
        df["genus"] = df["genus"].cat.reorder_categories(
            ["Homo sapiens", *other], ordered=False
        )

    # AMTL proportion and binomial GLM with frequency weights = sockets.
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    formula = "amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    print(result.summary())

    # Predicted AMTL probability by genus at average covariate values.
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    mode_tooth_class = df["tooth_class"].mode().iat[0]

    print("\nPredicted AMTL rates by genus at mean age/sex and modal tooth class:")
    rows = []
    for genus in df["genus"].cat.categories:
        new = pd.DataFrame(
            {
                "genus": [genus],
                "age": [mean_age],
                "prob_male": [mean_prob_male],
                "tooth_class": [mode_tooth_class],
            }
        )
        pred = result.get_prediction(new)
        sf = pred.summary_frame()
        mean = sf["mean"].iat[0]
        lo = sf["mean_ci_lower"].iat[0]
        hi = sf["mean_ci_upper"].iat[0]
        rows.append((genus, mean, lo, hi))

    for genus, mean, lo, hi in rows:
        print(f"{genus:12s}  mean={mean:.3f}, 95% CI=({lo:.3f}, {hi:.3f})")

    print("\nGenus coefficients (log-odds relative to Homo sapiens):")
    for name, coef in result.params.items():
        if name.startswith("C(genus)[T."):
            se = result.bse[name]
            pval = result.pvalues[name]
            print(f"{name:20s} coef={coef: .3f}, se={se: .3f}, p={pval: .3g}")


if __name__ == "__main__":
    main()

