import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Response: proportion of missing teeth within class, with binomial trials = observable sockets
    df = df.copy()
    df["prop_missing"] = df["feature3"] / df["feature4"]

    # Indicator for modern humans (Homo sapiens) vs all non-human primates
    df["is_human"] = (df["feature8"] == "Homo sapiens").astype(int)

    # Fit binomial regression: logit(p_missing) ~ human + age + sex + tooth class
    # Use number of observable sockets as frequency weights to respect binomial trials
    model = smf.glm(
        formula="prop_missing ~ is_human + feature5 + feature7 + C(feature1)",
        data=df,
        family=Binomial(),
        freq_weights=df["feature4"],
    )

    # Cluster-robust SEs by specimen to account for repeated tooth classes per individual
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["feature2"]})

    coef = float(result.params["is_human"])
    pvalue = float(result.pvalues["is_human"])
    odds_ratio = float(np.exp(coef))

    # Map evidence strength to a 0-100 Likert scale where higher = stronger "Yes"
    score = 50

    if pvalue < 0.05:
        if coef > 0:
            # Statistically significant higher AMTL in humans
            score = 70
            if odds_ratio >= 1.5:
                score = 80
            if odds_ratio >= 2.0:
                score = 90
        else:
            # Statistically significant lower AMTL in humans
            score = 30
            if odds_ratio <= 1 / 1.5:
                score = 20
            if odds_ratio <= 1 / 2.0:
                score = 10
    else:
        # Not statistically significant: lean slightly based on direction only
        if coef > 0:
            score = 60
        elif coef < 0:
            score = 40
        else:
            score = 50

    score_int = int(round(score))

    answer = "Yes" if score_int > 50 else "No" if score_int < 50 else "Inconclusive"

    explanation = (
        "Research question: Do modern humans (Homo sapiens) have higher frequencies "
        "of antemortem tooth loss (AMTL) than non-human primates (Pan, Pongo, Papio) "
        "after accounting for age, sex, and tooth class?\n\n"
        f"Using the provided dataset, I modeled the proportion of missing teeth within each "
        "tooth class as a binomial outcome (missing teeth out of observable sockets), with "
        "a logistic regression of AMTL frequency on an indicator for modern humans versus "
        "non-human primates, controlling for estimated age at death, a quantitative sex "
        "estimate, and tooth class (anterior, posterior, premolar). I used the number of "
        "observable sockets as binomial weights and clustered standard errors by specimen "
        "to account for repeated tooth-class observations per individual.\n\n"
        f"The coefficient for the human indicator is {coef:.3f}, corresponding to an odds "
        f"ratio of {odds_ratio:.2f} for AMTL in modern humans relative to non-human primates, "
        f"with a p-value of {pvalue:.3g}. This indicates that the data "
        f"{'provide' if pvalue < 0.05 else 'do not provide strong'} statistical evidence that "
        "modern humans have "
        f"{'higher' if coef > 0 else 'lower' if coef < 0 else 'similar'} AMTL frequencies than "
        "non-human primates once age, sex, and tooth class are controlled for.\n\n"
        f"Based on this model, my answer to the research question is: {answer}. "
        f"The Likert-scale response value of {score_int} reflects the direction and magnitude "
        "of the human effect (via the odds ratio) and the strength of statistical evidence "
        "(via the p-value), with values farther from 50 indicating stronger support for either "
        "a 'Yes' (higher AMTL in humans) or 'No' (no evidence or lower AMTL in humans) conclusion."
    )

    output = {"response": score_int, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(output, ensure_ascii=False))

    # Also print a concise summary for interactive inspection
    print(json.dumps({"response": score_int, "answer": answer, "coef": coef, "pvalue": pvalue, "odds_ratio": odds_ratio}))


if __name__ == "__main__":
    main()

