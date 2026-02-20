import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_model(df: pd.DataFrame):
    df = df.copy()
    df = df.dropna(
        subset=[
            "num_amtl",
            "sockets",
            "age",
            "prob_male",
            "genus",
            "tooth_class",
        ]
    )
    df = df[df["sockets"] > 0]

    df["genus"] = pd.Categorical(
        df["genus"], categories=["Homo sapiens", "Pan", "Papio", "Pongo"]
    )
    df["tooth_class"] = pd.Categorical(df["tooth_class"])
    df["prop_amtl"] = df["num_amtl"] / df["sockets"]

    model = smf.glm(
        formula="prop_amtl ~ C(genus) + age + prob_male + C(tooth_class)",
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    return model.fit()


def summarize_results(result) -> dict:
    params = result.params
    pvalues = result.pvalues

    comparisons = {}
    for genus in ["Pan", "Papio", "Pongo"]:
        term = f"C(genus)[T.{genus}]"
        coef = float(params.get(term, float("nan")))
        pval = float(pvalues.get(term, float("nan")))
        comparisons[genus] = {"coef_vs_human": coef, "pvalue": pval}

    homo_higher_flags = [
        comp["coef_vs_human"] < 0 for comp in comparisons.values()
    ]

    if all(homo_higher_flags):
        response = "Yes"
    else:
        response = "No"

    valid_pvals = [
        comp["pvalue"] for comp in comparisons.values() if not pd.isna(comp["pvalue"])
    ]
    if valid_pvals:
        max_p = max(valid_pvals)
    else:
        max_p = 1.0

    confidence = max(0.0, min(100.0, (1.0 - max_p) * 100.0))

    genuses = ["Homo sapiens", "Pan", "Papio", "Pongo"]
    categories = ["Homo sapiens", "Pan", "Papio", "Pongo"]

    tooth_mode = result.model.data.frame["tooth_class"].mode().iat[0]
    age_mean = float(result.model.data.frame["age"].mean())
    prob_male_mean = float(result.model.data.frame["prob_male"].mean())

    preds = {}
    for g in genuses:
        pred_df = pd.DataFrame(
            {
                "genus": pd.Categorical([g], categories=categories),
                "age": [age_mean],
                "prob_male": [prob_male_mean],
                "tooth_class": pd.Categorical(
                    [tooth_mode],
                    categories=result.model.data.frame["tooth_class"].cat.categories,
                ),
            }
        )
        preds[g] = float(result.predict(pred_df)[0])

    explanation = (
        "Using a binomial regression model on the proportion of missing teeth "
        "(num_amtl / sockets) with a logit link and weights equal to the number "
        "of observable sockets, I modeled AMTL as a function of genus, age, sex "
        "(prob_male), and tooth class. Homo sapiens was set as the reference "
        "genus, so coefficients for Pan, Papio, and Pongo represent the change "
        "in log-odds of AMTL relative to modern humans after adjusting for age, "
        "sex, and tooth class. For all non-human genera, the estimated "
        "coefficients relative to Homo sapiens were "
        f"{comparisons['Pan']['coef_vs_human']:.3f} (Pan), "
        f"{comparisons['Papio']['coef_vs_human']:.3f} (Papio), and "
        f"{comparisons['Pongo']['coef_vs_human']:.3f} (Pongo), with "
        f"p-values {comparisons['Pan']['pvalue']:.3g}, "
        f"{comparisons['Papio']['pvalue']:.3g}, and "
        f"{comparisons['Pongo']['pvalue']:.3g}, respectively. At typical values "
        "of age, sex, and tooth class, the model-predicted AMTL probabilities "
        f"were approximately {preds['Homo sapiens']:.3f} for Homo sapiens, "
        f"{preds['Pan']:.3f} for Pan, {preds['Papio']:.3f} for Papio, and "
        f"{preds['Pongo']:.3f} for Pongo. "
    )

    if response == "Yes":
        explanation += (
            "Because all non-human genera have lower adjusted log-odds of AMTL "
            "than Homo sapiens (negative coefficients relative to humans), the "
            "model indicates that modern humans do have higher frequencies of "
            "antemortem tooth loss than the non-human primates after accounting "
            "for age, sex, and tooth class."
        )
    else:
        explanation += (
            "Because at least one non-human genus does not show lower adjusted "
            "log-odds of AMTL than Homo sapiens, the model does not support the "
            "claim that modern humans have uniformly higher AMTL frequencies "
            "than all non-human primate genera after accounting for age, sex, "
            "and tooth class."
        )

    return {
        "response": response,
        "confidence": confidence,
        "explanation": explanation,
    }


def main() -> None:
    df = pd.read_csv("amtl.csv")
    result = fit_model(df)
    summary = summarize_results(result)

    conclusion_path = Path("conclusion.txt")
    with conclusion_path.open("w") as f:
        f.write(json.dumps(summary))


if __name__ == "__main__":
    main()
