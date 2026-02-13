import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("amtl.csv")

    # Rename columns to more descriptive internal names
    df = df.rename(
        columns={
            "genus": "num_missing",
            "age": "num_sockets",
            "pop": "age_at_death",
            "stdev_age": "prob_male",  # continuous sex estimate
            "prob_male": "specimen_id",  # original ID string
            "tooth_class": "genus",
            "sockets": "tooth_class",
        }
    )

    # Filter out any rows where sockets are zero or missing just in case
    df = df[df["num_sockets"] > 0].copy()

    # Compute proportion of teeth missing in that tooth class for each specimen
    df["prop_missing"] = df["num_missing"] / df["num_sockets"]

    # Create an indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Inspect dtypes to ensure predictors are 1D
    print("Dtypes:\n", df.dtypes)
    print("\nSample genus values:", df["genus"].head().tolist())
    print("Sample tooth_class values:", df["tooth_class"].head().tolist())

    # Fit a binomial GLM on proportions with frequency weights (num_sockets)
    # Genus and tooth_class are treated as categorical automatically.
    formula = "prop_missing ~ genus + age_at_death + prob_male + tooth_class"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["num_sockets"],
    )
    result = model.fit()

    # Extract genus-related coefficients
    params = result.params.to_dict()
    conf_int = result.conf_int().rename(columns={0: "ci_lower", 1: "ci_upper"})

    genus_terms = {}
    for name, coef in params.items():
        if name.startswith("genus["):
            stats = {
                "coef": coef,
                "ci_lower": float(conf_int.loc[name, "ci_lower"]),
                "ci_upper": float(conf_int.loc[name, "ci_upper"]),
                "p_value": float(result.pvalues.get(name, float("nan"))),
            }
            genus_terms[name] = stats

    # For interpretability, compute predicted probabilities at representative values
    # Choose mid-range age, prob_male=0.5, and each genus/tooth_class combination.
    rep_age = float(df["age_at_death"].median())
    rep_sex = 0.5

    pred_rows = []
    genuses = sorted(df["genus"].unique())
    tooth_classes = sorted(df["tooth_class"].unique())
    for genus_val in genuses:
        for tc in tooth_classes:
            pred_rows.append(
                {
                    "genus": genus_val,
                    "tooth_class": tc,
                    "age_at_death": rep_age,
                    "prob_male": rep_sex,
                }
            )

    pred_df = pd.DataFrame(pred_rows)

    pred_probs = result.predict(pred_df)
    pred_df["pred_prob_missing"] = pred_probs

    # Summarize predicted probabilities by genus (averaged across tooth classes)
    avg_pred_by_genus = (
        pred_df.groupby("genus")["pred_prob_missing"].mean().to_dict()
    )

    # Save a compact JSON with the key quantities to support interpretation.
    out = {
        "genus_coefficients": genus_terms,
        "avg_pred_prob_missing_by_genus": avg_pred_by_genus,
        "representative_age": rep_age,
        "representative_prob_male": rep_sex,
    }

    Path("analysis_results.json").write_text(json.dumps(out, indent=2))

    # Also print a brief textual summary for interactive inspection.
    print("Genus coefficients (relative to Pan):")
    for term, stats in genus_terms.items():
        print(term, stats)

    print(
        "\nAverage predicted proportion of missing teeth by genus "
        "(across tooth classes):"
    )
    for genus, prob in avg_pred_by_genus.items():
        print(f"{genus}: {prob:.4f}")


if __name__ == "__main__":
    main()
