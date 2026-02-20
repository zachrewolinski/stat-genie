import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data = pd.read_csv("amtl.csv")

    # Construct binomial response: proportion of missing teeth with number of trials = sockets
    data = data.copy()
    data["amtl_rate"] = data["num_amtl"] / data["sockets"]

    # Fit binomial logistic regression with Homo sapiens as baseline genus.
    # Adjust for age, sex (prob_male), and tooth class. Cluster by specimen to
    # account for non-independence of rows within specimens.
    model = smf.glm(
        formula="amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)",
        data=data,
        family=sm.families.Binomial(),
        freq_weights=data["sockets"],
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": data["specimen"]})

    print(result.summary())

    # Predicted AMTL probabilities by genus at mean age, mean prob_male,
    # averaging equally over tooth classes.
    age_mean = data["age"].mean()
    prob_male_mean = data["prob_male"].mean()
    tooth_classes = sorted(data["tooth_class"].unique())
    genera = sorted(data["genus"].unique())

    rows = []
    for g in genera:
        for t in tooth_classes:
            rows.append(
                {
                    "genus": g,
                    "tooth_class": t,
                    "age": age_mean,
                    "prob_male": prob_male_mean,
                }
            )

    pred_df = pd.DataFrame(rows)
    pred_df["pred_prob"] = result.predict(pred_df)

    genus_means = pred_df.groupby("genus")["pred_prob"].mean().sort_values(ascending=False)

    print("\nPredicted AMTL probability by genus (mean over tooth classes):")
    for genus, prob in genus_means.items():
        print(f"{genus}: {prob:.4f}")

    # Also print genus coefficients relative to Homo sapiens baseline.
    print("\nGenus coefficients relative to Homo sapiens:")
    for name, coef in result.params.items():
        if name.startswith("C(genus)"):
            pval = result.pvalues[name]
            print(f"{name}: coef={coef:.3f}, p={pval:.3g}")


if __name__ == "__main__":
    main()

