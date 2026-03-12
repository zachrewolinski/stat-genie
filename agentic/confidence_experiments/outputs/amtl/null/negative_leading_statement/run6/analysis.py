import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Basic sanity checks
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "genus", "tooth_class"])
    df = df[df["num_amtl"] >= 0]
    df = df[df["sockets"] > 0]
    # Drop any rows where the count of missing teeth exceeds the number of observable sockets
    df = df[df["num_amtl"] <= df["sockets"]]

    # Ensure categorical types and reference levels
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    if "Homo sapiens" in list(df["genus"].cat.categories):
        # Reorder so Homo sapiens is the reference category
        other = [g for g in df["genus"].cat.categories if g != "Homo sapiens"]
        df["genus"] = df["genus"].cat.reorder_categories(["Homo sapiens"] + other)

    # Proportion of antemortem tooth loss
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    # Binomial regression with genus, age, sex proxy (prob_male), and tooth class
    formula = "prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    # Cluster-robust SEs by specimen to account for repeated rows per individual
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    print(result.summary())
    print("\nModel uses binomial family with logit link, weighted by sockets and clustered by specimen.\n")

    # Predicted probabilities by genus and tooth class at typical covariate values
    base = {
        "age": df["age"].median(),
        "prob_male": 0.5,
    }

    rows = []
    for genus in df["genus"].cat.categories:
        for tooth_class in df["tooth_class"].cat.categories:
            rows.append(
                {
                    "genus": genus,
                    "tooth_class": tooth_class,
                    **base,
                }
            )

    new_df = pd.DataFrame(rows)
    pred = result.get_prediction(new_df)
    sf = pred.summary_frame(alpha=0.05)

    new_df["pred_prob"] = sf["mean"]
    new_df["pred_prob_ci_low"] = sf["mean_ci_lower"]
    new_df["pred_prob_ci_high"] = sf["mean_ci_upper"]

    print("\nPredicted AMTL probability by genus and tooth class\n"
          "(age fixed at median, prob_male=0.5):")
    print(new_df)

    avg = new_df.groupby("genus")[["pred_prob", "pred_prob_ci_low", "pred_prob_ci_high"]].mean()
    print("\nAverage predicted AMTL probability by genus (mean over tooth classes):")
    print(avg)

    coef = result.params
    ci = result.conf_int()
    coef_table = pd.concat([coef, ci], axis=1)
    coef_table.columns = ["coef", "ci_low", "ci_high"]
    print("\nCoefficients with 95% confidence intervals (cluster-robust):")
    print(coef_table)


if __name__ == "__main__":
    main()
