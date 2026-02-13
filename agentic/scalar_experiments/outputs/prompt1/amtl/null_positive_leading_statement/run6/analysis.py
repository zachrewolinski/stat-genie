import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    df = pd.read_csv(data_path)

    # Basic cleaning and sanity checks
    required_cols = [
        "num_amtl",
        "sockets",
        "genus",
        "tooth_class",
        "age",
        "prob_male",
    ]
    df = df.dropna(subset=required_cols).copy()

    # Ensure sensible values
    df = df[df["sockets"] > 0]
    df = df[df["num_amtl"] >= 0]
    df = df[df["num_amtl"] <= df["sockets"]]

    if df.empty:
        raise ValueError("No valid rows remain after filtering.")

    # Aggregated binomial response: proportion of missing teeth with socket counts as weights
    df["prop_missing"] = df["num_amtl"] / df["sockets"]

    # Categorical encodings
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    if "Homo sapiens" not in list(df["genus"].cat.categories):
        raise ValueError("Expected 'Homo sapiens' in genus categories.")

    # Reorder so that modern humans are the reference category
    genus_cats = list(df["genus"].cat.categories)
    genus_cats.remove("Homo sapiens")
    df["genus"] = df["genus"].cat.reorder_categories(
        ["Homo sapiens"] + genus_cats, ordered=False
    )

    # Fit binomial GLM with logit link, using sockets as binomial denominators
    model = sm.GLM.from_formula(
        "prop_missing ~ C(genus) + C(tooth_class) + age + prob_male",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Predict adjusted AMTL probabilities for each genus at typical covariate values
    mean_age = float(df["age"].mean())
    mean_prob_male = float(df["prob_male"].mean())
    ref_tooth_class = df["tooth_class"].mode()[0]

    genus_levels = list(df["genus"].cat.categories)
    pred_df = pd.DataFrame(
        {
            "genus": genus_levels,
            "tooth_class": ref_tooth_class,
            "age": mean_age,
            "prob_male": mean_prob_male,
        }
    )
    pred_df["genus"] = pd.Categorical(
        pred_df["genus"], categories=df["genus"].cat.categories
    )
    pred_df["tooth_class"] = pd.Categorical(
        pred_df["tooth_class"], categories=df["tooth_class"].cat.categories
    )

    pred_df["pred_amtl_prob"] = result.predict(pred_df)

    # Extract genus effects (relative to Homo sapiens as reference)
    params = result.params
    pvalues = result.pvalues

    nonhuman_genera = [g for g in genus_levels if g != "Homo sapiens"]

    genus_results = []
    homo_higher_significant = True

    homo_pred = float(
        pred_df.loc[pred_df["genus"] == "Homo sapiens", "pred_amtl_prob"].iloc[0]
    )

    for g in nonhuman_genera:
        term = f"C(genus)[T.{g}]"
        if term not in params.index:
            # If a term is missing, treat as no clear evidence for a difference
            homo_higher_significant = False
            effect = np.nan
            pval = np.nan
            g_pred = float(
                pred_df.loc[pred_df["genus"] == g, "pred_amtl_prob"].iloc[0]
            )
        else:
            effect = float(params[term])
            pval = float(pvalues[term])
            g_pred = float(
                pred_df.loc[pred_df["genus"] == g, "pred_amtl_prob"].iloc[0]
            )
            # For humans to have higher AMTL, genus coefficients (relative to Homo) should be negative
            # and statistically significant.
            if not (effect < 0 and pval < 0.05):
                homo_higher_significant = False

        genus_results.append(
            {
                "genus": g,
                "coef_vs_homo": effect,
                "pvalue": pval,
                "pred_prob": g_pred,
            }
        )

    response = "Yes" if homo_higher_significant else "No"

    # Build human-readable explanation
    lines = []
    lines.append(
        "I fit a binomial logistic regression model for the proportion of missing teeth "
        "(num_amtl / sockets) with sockets as binomial denominators and predictors "
        "for genus, tooth class, age at death, and sex (prob_male)."
    )
    lines.append(
        f"Modern humans (Homo sapiens) were set as the reference genus. "
        f"Predicted AMTL probability for humans at the average age ({mean_age:.1f} years), "
        f"average sex estimate (prob_male={mean_prob_male:.2f}), and the most common "
        f"tooth class ({ref_tooth_class}) was {homo_pred:.3f}."
    )

    for gr in genus_results:
        lines.append(
            f"For {gr['genus']}, the genus coefficient relative to humans was "
            f"{gr['coef_vs_homo']:.3f} (p = {gr['pvalue']:.3g}), with an adjusted "
            f"predicted AMTL probability of {gr['pred_prob']:.3f} under the same covariate values."
        )

    if response == "Yes":
        lines.append(
            "All non-human genera show significantly lower AMTL log-odds than humans "
            "(negative coefficients with p < 0.05), indicating that, after adjusting for "
            "age, sex, and tooth class, modern humans have higher frequencies of antemortem "
            "tooth loss than each of the non-human primate genera (Pan, Pongo, Papio)."
        )
    else:
        lines.append(
            "At least one non-human genus does not differ significantly from humans, or "
            "has an estimated AMTL rate that is not clearly lower than that of humans "
            "once age, sex, and tooth class are controlled. Thus the data do not provide "
            "consistent evidence that modern humans have higher AMTL frequencies than all "
            "non-human primate genera."
        )

    explanation = " ".join(lines)

    conclusion = {
        "response": response,
        "explanation": explanation,
    }

    # Write the required JSON object to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

