import json

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("amtl.csv")

    # Indicator for modern humans
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Binomial response: AMTL rate with sockets as trials
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    # Basic descriptive rates by genus
    genus_summary = (
        df.assign(amtl_rate=df["num_amtl"] / df["sockets"])
        .groupby("genus")
        .agg(
            mean_rate=("amtl_rate", "mean"),
            n_rows=("amtl_rate", "size"),
            total_missing=("num_amtl", "sum"),
            total_sockets=("sockets", "sum"),
        )
    )

    # Overall human vs non‑human difference (raw, unadjusted)
    humans = df[df["is_human"] == 1]
    non_humans = df[df["is_human"] == 0]
    human_rate_raw = humans["num_amtl"].sum() / humans["sockets"].sum()
    nonhuman_rate_raw = non_humans["num_amtl"].sum() / non_humans["sockets"].sum()

    # Binomial regression controlling for age, sex, and tooth class
    formula = "amtl_rate ~ is_human + age + prob_male + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )

    # Use cluster-robust SEs at the specimen level to account for repeated rows
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["specimen"]})

    coef_human = float(result.params["is_human"])
    pval_human = float(result.pvalues["is_human"])

    # Compute predicted AMTL probability for a typical tooth, human vs non‑human
    median_age = float(df["age"].median())
    mean_prob_male = float(df["prob_male"].mean())
    ref_tooth_class = df["tooth_class"].mode().iloc[0]

    def predict_prob(is_human_value: int) -> float:
        new = pd.DataFrame(
            {
                "amtl_rate": [0.0],  # placeholder, not used in prediction
                "is_human": [is_human_value],
                "age": [median_age],
                "prob_male": [mean_prob_male],
                "tooth_class": [ref_tooth_class],
            }
        )
        return float(result.predict(new)[0])

    pred_human = predict_prob(1)
    pred_nonhuman = predict_prob(0)
    diff_pred = pred_human - pred_nonhuman

    # Map model evidence to a 0–100 Likert scale where 0 = strong "No"
    # and 100 = strong "Yes" to the question:
    # "Do modern humans have higher AMTL frequencies than non‑human primates,
    #  after accounting for age, sex, and tooth class?"
    #
    # Heuristic mapping based on sign and strength of the human effect:
    # - Strong evidence humans have LOWER or equal AMTL (coef <= 0, p < 0.05): near 0
    # - Strong evidence humans have HIGHER AMTL (coef > 0, p < 0.05): near 100
    # - Weak/ambiguous evidence: around the middle, shifted toward the sign.
    if pval_human < 0.05:
        if coef_human > 0:
            response = 90
            qualitative = (
                "There is strong evidence that modern humans have higher AMTL "
                "frequencies than non-human primates after adjustment."
            )
        else:
            response = 10
            qualitative = (
                "There is strong evidence that modern humans do not have higher AMTL "
                "frequencies than non-human primates after adjustment; if anything, "
                "their rates are lower."
            )
    else:
        # No statistically clear difference; center around 50 and tilt toward the sign
        if coef_human > 0:
            response = 60
            qualitative = (
                "The adjusted model suggests slightly higher AMTL in modern humans, "
                "but the evidence is weak and not statistically significant."
            )
        elif coef_human < 0:
            response = 40
            qualitative = (
                "The adjusted model suggests slightly lower AMTL in modern humans, "
                "but the evidence is weak and not statistically significant."
            )
        else:
            response = 50
            qualitative = (
                "The adjusted model shows essentially no difference in AMTL "
                "frequencies between modern humans and non-human primates."
            )

    # Build a concise explanation summarizing the key evidence
    explanation_lines = [
        "Research question: Do modern humans (Homo sapiens) have higher frequencies "
        "of antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio), "
        "after accounting for age, sex, and tooth class?",
        "",
        "Data and model:",
        f"- Dataset contains {len(df)} rows with AMTL counts and sockets by specimen and tooth class.",
        "- Response modeled as binomial AMTL rate (missing teeth / sockets) with sockets as the number of trials.",
        "- Fitted a binomial GLM with logit link: AMTL rate ~ is_human + age + prob_male + tooth_class,",
        "  using cluster-robust standard errors at the specimen level.",
        "",
        "Descriptive patterns:",
        f"- Raw overall AMTL rate in humans: {human_rate_raw:.3f}.",
        f"- Raw overall AMTL rate in non-human primates: {nonhuman_rate_raw:.3f}.",
        "- Genus-specific summaries (mean AMTL rate, rows, total missing / total sockets):",
    ]

    for genus, row in genus_summary.iterrows():
        rate = row["total_missing"] / row["total_sockets"]
        explanation_lines.append(
            f"  * {genus}: mean row-wise rate={row['mean_rate']:.3f}, "
            f"n={int(row['n_rows'])}, total rate={rate:.3f}"
        )

    explanation_lines.extend(
        [
            "",
            "Adjusted model results:",
            f"- Coefficient for human indicator (is_human) on the log-odds scale: {coef_human:.3f}.",
            f"- p-value for is_human: {pval_human:.3g}.",
            f"- Predicted AMTL probability for a typical tooth (median age, mean sex, {ref_tooth_class}):",
            f"  * Human: {pred_human:.3f}",
            f"  * Non-human primate: {pred_nonhuman:.3f}",
            f"  * Difference (human - non-human): {diff_pred:.3f}",
            "",
            "Interpretation:",
            qualitative,
            "",
            "Conclusion mapped to Likert scale (0 = strong 'No', 100 = strong 'Yes'):",
            f"- Response score: {response}",
        ]
    )

    explanation = "\n".join(explanation_lines)

    conclusion = {
        "response": int(response),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

