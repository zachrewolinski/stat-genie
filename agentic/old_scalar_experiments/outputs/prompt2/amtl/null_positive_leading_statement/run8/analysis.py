import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Ensure genus uses the human category label exactly as in the CSV
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Binomial GLM: AMTL counts (num_amtl) out of sockets, with covariates
    # Use proportion as response with sockets as frequency weights.
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    model = smf.glm(
        formula="amtl_prop ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # In statsmodels, with C(genus), the first category (alphabetical) is baseline.
    # With categories 'Homo sapiens', 'Pan', 'Papio', 'Pongo', Homo sapiens is baseline.
    # Coefficients for non-human genera show difference in log-odds vs humans.
    params = result.params
    conf_int = result.conf_int()

    genus_terms = {
        "Pan": "C(genus)[T.Pan]",
        "Papio": "C(genus)[T.Papio]",
        "Pongo": "C(genus)[T.Pongo]",
    }

    evidence_messages = []
    all_lower_than_humans = True

    for genus, term in genus_terms.items():
        if term not in params.index:
            evidence_messages.append(
                f"No coefficient estimated for genus {genus}; treating evidence as inconclusive for this genus."
            )
            all_lower_than_humans = False
            continue

        coef = params[term]
        ci_low, ci_high = conf_int.loc[term]
        pval = result.pvalues[term]

        direction = "lower" if coef < 0 else "higher"
        signif = pval < 0.05

        msg = (
            f"For genus {genus}, the adjusted log-odds of AMTL are "
            f"{direction} than humans (coef={coef:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], p={pval:.3g})."
        )
        evidence_messages.append(msg)

        # To conclude that humans have higher AMTL, we require all non-human genera
        # to have significantly lower log-odds than humans.
        if not (coef < 0 and signif):
            all_lower_than_humans = False

    if all_lower_than_humans:
        response = "Yes"
        confidence = 85
        conclusion = (
            "A binomial regression model of AMTL counts (out of sockets) shows that, "
            "after controlling for age, sex (prob_male), and tooth class, all non-human "
            "genera (Pan, Papio, Pongo) have significantly lower AMTL log-odds than Homo sapiens. "
            "This supports the hypothesis that modern humans have higher frequencies of AMTL than "
            "these non-human primates."
        )
    else:
        response = "No"
        confidence = 65
        conclusion = (
            "A binomial regression model of AMTL counts (out of sockets) controlling for age, "
            "sex (prob_male), and tooth class does not show that all non-human genera (Pan, "
            "Papio, Pongo) have significantly lower AMTL log-odds than Homo sapiens. "
            "At least one non-human genus has AMTL rates that are not significantly lower than "
            "those of humans, so the data do not strongly support the claim that modern humans "
            "have higher AMTL frequencies than all three non-human primate genera once covariates "
            "are accounted for."
        )

    explanation = (
        "Modeling details: I fit a binomial GLM using statsmodels with AMTL proportion "
        "(num_amtl / sockets) as the response, sockets as frequency weights, and predictors "
        "for genus (categorical with Homo sapiens as the baseline), age, sex estimate (prob_male), "
        "and tooth_class. I then examined the coefficients and confidence intervals for the genus "
        "terms comparing each non-human genus to humans. Evidence summary:\n- "
        + "\n- ".join(evidence_messages)
        + "\nThese results form the basis for the Yes/No conclusion above."
    )

    conclusion_obj = {
        "response": response,
        "confidence": confidence,
        "explanation": conclusion,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion_obj, ensure_ascii=False))


if __name__ == "__main__":
    main()

