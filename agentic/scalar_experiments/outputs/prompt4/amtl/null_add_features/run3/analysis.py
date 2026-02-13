import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_model():
    df = pd.read_csv("amtl.csv")

    # Indicator for modern humans vs. non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Proportion of missing teeth and binomial weights
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    model = smf.glm(
        formula="prop_amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    return df, result


def summarize(result, df):
    # Simple genus-level summary for context
    genus_summary = (
        df.assign(prop_amtl=df["num_amtl"] / df["sockets"])
        .groupby("genus")
        .agg(
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
        )
    )
    genus_summary["overall_prop_amtl"] = (
        genus_summary["total_missing"] / genus_summary["total_sockets"]
    )

    human_coef = result.params["is_human"]
    human_pvalue = result.pvalues["is_human"]
    human_ci_low, human_ci_high = result.conf_int().loc["is_human"]

    # Predicted probabilities for a typical tooth socket in humans vs. non-humans
    mean_age = df["age"].mean()
    mean_prob_male = df["prob_male"].mean()
    common_tooth_class = df["tooth_class"].mode()[0]

    base_row = {
        "age": mean_age,
        "prob_male": mean_prob_male,
        "tooth_class": common_tooth_class,
    }

    human_row = base_row.copy()
    human_row["is_human"] = 1

    nonhuman_row = base_row.copy()
    nonhuman_row["is_human"] = 0

    pred_df = pd.DataFrame([human_row, nonhuman_row])
    preds = result.get_prediction(pred_df).summary_frame()

    human_prob = preds["mean"].iloc[0]
    nonhuman_prob = preds["mean"].iloc[1]

    print("Genus-level AMTL summary (overall):")
    print(genus_summary)
    print()
    print("Human coefficient (log-odds):", human_coef)
    print("Human p-value:", human_pvalue)
    print("Human 95% CI (log-odds):", human_ci_low, human_ci_high)
    print()
    print("Predicted AMTL probability per socket (typical case):")
    print("  Humans:     {:.4f}".format(human_prob))
    print("  Non-humans: {:.4f}".format(nonhuman_prob))
    print("  Difference: {:.4f}".format(human_prob - nonhuman_prob))


def main():
    df, result = fit_model()
    summarize(result, df)


if __name__ == "__main__":
    main()
