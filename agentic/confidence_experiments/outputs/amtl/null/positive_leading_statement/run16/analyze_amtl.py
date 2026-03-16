import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    if not data_path.exists():
        raise FileNotFoundError("amtl.csv not found in current directory")

    df = pd.read_csv(data_path)

    # Basic sanity checks
    df = df.dropna(subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "genus"])
    df = df[df["sockets"] > 0].copy()

    # Create AMTL proportion for descriptive purposes
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Make genus a categorical variable with Homo sapiens as the reference
    df["genus"] = df["genus"].astype("category")
    if "Homo sapiens" not in df["genus"].cat.categories:
        raise ValueError("Expected 'Homo sapiens' in genus column")
    df["genus"] = df["genus"].cat.reorder_categories(
        sorted(df["genus"].cat.categories, key=lambda x: (x != "Homo sapiens", x))
    )

    # Tooth class as categorical
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Binomial regression: num_amtl successes out of sockets trials
    # Model: logit(p) ~ genus + age + prob_male + tooth_class
    model = smf.glm(
        formula="num_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract coefficients comparing each non-human genus to Homo sapiens
    summary_frame = result.summary2().tables[1]

    # Coefficient names depend on pandas/statsmodels coding, standard form:
    # C(genus)[T.Pan], C(genus)[T.Papio], C(genus)[T.Pongo]
    genus_effects = {}
    nonhuman_genera = ["Pan", "Papio", "Pongo"]
    for g in nonhuman_genera:
        term = f"C(genus)[T.{g}]"
        if term in summary_frame.index:
            coef = float(summary_frame.loc[term, "Coef."])
            pvalue = float(summary_frame.loc[term, "P>|z|"])
            genus_effects[g] = {"coef": coef, "pvalue": pvalue}

    # Compute predicted probabilities at representative covariate values
    # Use median age, prob_male=0.5, and Posterior tooth class as a typical class if available.
    median_age = float(df["age"].median())
    rep_prob_male = 0.5
    rep_tooth_class = "Posterior"
    if rep_tooth_class not in df["tooth_class"].cat.categories:
        rep_tooth_class = df["tooth_class"].cat.categories[0]

    genera_for_pred = ["Homo sapiens"] + [g for g in nonhuman_genera if g in df["genus"].cat.categories]
    pred_rows = []
    for g in genera_for_pred:
        pred_rows.append(
            {
                "genus": g,
                "age": median_age,
                "prob_male": rep_prob_male,
                "tooth_class": rep_tooth_class,
                "sockets": 1,
                "num_amtl": 0,
            }
        )
    pred_df = pd.DataFrame(pred_rows)
    pred_df["genus"] = pred_df["genus"].astype("category")
    pred_df["genus"] = pred_df["genus"].cat.set_categories(df["genus"].cat.categories)
    pred_df["tooth_class"] = pred_df["tooth_class"].astype("category")
    pred_df["tooth_class"] = pred_df["tooth_class"].cat.set_categories(df["tooth_class"].cat.categories)

    predicted = result.predict(pred_df)
    genus_predicted = dict(zip(pred_df["genus"], predicted))

    # Determine evidence that humans have higher AMTL than non-human genera
    # We expect negative coefficients for non-human genera relative to Homo sapiens
    evidence_flags = []
    for g, eff in genus_effects.items():
        coef = eff["coef"]
        pvalue = eff["pvalue"]
        higher_for_humans = coef < 0  # non-human genus has lower log-odds than Homo
        significant = pvalue < 0.05
        evidence_flags.append((g, higher_for_humans, significant))

    # Assess overall strength of evidence
    num_genera_with_sig_higher_humans = sum(1 for g, higher, sig in evidence_flags if higher and sig)
    num_genera_modeled = len(genus_effects)

    # Likert scale mapping:
    # - Strong consistent evidence humans > all non-human genera: 85–100
    # - Mixed but generally higher and at least some significance: 60–80
    # - Little or no evidence: 0–40
    if num_genera_with_sig_higher_humans == num_genera_modeled and num_genera_modeled > 0:
        response_score = 92
        qualitative = "strong"
    elif num_genera_with_sig_higher_humans > 0:
        response_score = 72
        qualitative = "moderate"
    else:
        # Check if humans still tend to have higher predicted probabilities
        humans_prob = genus_predicted.get("Homo sapiens", np.nan)
        nonhuman_probs = [
            genus_predicted[g] for g in nonhuman_genera if g in genus_predicted and not np.isnan(genus_predicted[g])
        ]
        if nonhuman_probs and humans_prob > max(nonhuman_probs):
            response_score = 55
            qualitative = "weak"
        else:
            response_score = 20
            qualitative = "no"

    # Build textual explanation
    lines = []
    lines.append(
        "I fit a binomial logistic regression model for the number of missing teeth "
        "(num_amtl) out of the total observable sockets, with predictors genus, age, "
        "sex (prob_male), and tooth class."
    )
    if genus_effects:
        lines.append(
            "In this model, 'Homo sapiens' was treated as the reference genus so that "
            "coefficients for Pan, Papio, and Pongo represent differences in AMTL log-odds relative to humans."
        )
        for g, eff in genus_effects.items():
            direction = "lower" if eff["coef"] < 0 else "higher"
            lines.append(
                f"For {g}, the coefficient relative to Homo sapiens was {eff['coef']:.3f} "
                f"with p-value {eff['pvalue']:.3g}, indicating that this genus has "
                f"{direction} AMTL log-odds than humans after controlling for age, sex, and tooth class."
            )

    humans_prob = genus_predicted.get("Homo sapiens", np.nan)
    if not np.isnan(humans_prob):
        nonhuman_probs_desc = []
        for g in nonhuman_genera:
            if g in genus_predicted:
                nonhuman_probs_desc.append(f"{g}: {genus_predicted[g]:.3f}")
        if nonhuman_probs_desc:
            lines.append(
                "Using representative covariate values (median age, prob_male=0.5, and a common tooth class), "
                f"the predicted AMTL probability for Homo sapiens was {humans_prob:.3f}, "
                "compared to " + ", ".join(nonhuman_probs_desc) + " for the non-human genera."
            )

    if qualitative == "strong":
        conclusion_sentence = (
            "These results provide strong and consistent evidence that modern humans "
            "have higher frequencies of antemortem tooth loss than the non-human primate genera "
            "after accounting for age, sex, and tooth class."
        )
    elif qualitative == "moderate":
        conclusion_sentence = (
            "Overall, the regression results suggest that humans tend to have higher "
            "frequencies of antemortem tooth loss than at least some of the non-human primate genera, "
            "though the evidence is not uniformly strong across all comparisons."
        )
    elif qualitative == "weak":
        conclusion_sentence = (
            "The evidence that humans have higher frequencies of antemortem tooth loss than "
            "the non-human primate genera is weak: effect sizes generally favor humans but "
            "statistical support is limited."
        )
    else:
        conclusion_sentence = (
            "There is little statistical evidence that modern humans have higher frequencies "
            "of antemortem tooth loss than the non-human primate genera once age, sex, and "
            "tooth class are taken into account."
        )

    lines.append(conclusion_sentence)
    lines.append(
        f"On a 0–100 Likert scale where 0 is a strong 'No' and 100 is a strong 'Yes', "
        f"I assign a value of {response_score} to the claim that humans have higher AMTL "
        "frequencies than the non-human primates after adjusting for the covariates."
    )

    explanation = " ".join(lines)

    output = {"response": int(response_score), "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
