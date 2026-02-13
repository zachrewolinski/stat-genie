import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm


def main() -> None:
    data_path = Path("amtl.csv")
    df = pd.read_csv(data_path)

    # Basic cleaning: ensure expected columns are present
    expected_cols = {
        "tooth_class",
        "specimen",
        "num_amtl",
        "sockets",
        "age",
        "stdev_age",
        "prob_male",
        "genus",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Filter to genera of interest
    target_genera = ["Homo sapiens", "Pan", "Pongo", "Papio"]
    df = df[df["genus"].isin(target_genera)].copy()

    # Create outcome as proportion of teeth lost, with sockets as trials
    df["amtl_prop"] = df["num_amtl"] / df["sockets"]

    # Drop rows with zero sockets just in case
    df = df[df["sockets"] > 0].copy()

    # Treat categorical predictors appropriately
    df["genus"] = df["genus"].astype("category")
    df["tooth_class"] = df["tooth_class"].astype("category")

    # Center/scale continuous predictors modestly for stability
    df["age_c"] = df["age"] - df["age"].mean()
    df["prob_male_c"] = df["prob_male"] - df["prob_male"].mean()

    # Binomial regression on aggregated data using proportions and weights
    formula = "amtl_prop ~ C(genus) + age_c + prob_male_c + C(tooth_class)"
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df["sockets"],
    )
    result = model.fit()

    # Extract genus effects relative to reference (alphabetical baseline)
    # We will focus on whether Homo sapiens has a significantly higher AMTL rate
    # than each non-human genus after adjustment.
    summary_frame = result.summary2().tables[1]

    # Determine reference level for genus used by statsmodels
    # Statsmodels uses treatment coding with the smallest level as reference.
    genus_levels = list(df["genus"].cat.categories)
    reference_genus = genus_levels[0]

    # Build pairwise comparisons on the logit scale between Homo sapiens and each non-human genus
    # using linear hypothesis tests via the covariance of parameters.
    params = result.params
    cov = result.cov_params()

    def contrast_logit(genus_a: str, genus_b: str) -> dict:
        """
        Compare genus_a - genus_b on the logit scale.
        Positive estimate means higher AMTL for genus_a.
        """
        # Build contrast vector over parameters
        param_names = params.index.tolist()
        import numpy as np

        c = np.zeros(len(param_names))

        def coef_name_for(genus_name: str) -> str | None:
            if genus_name == reference_genus:
                return None
            return f"C(genus)[T.{genus_name}]"

        name_a = coef_name_for(genus_a)
        name_b = coef_name_for(genus_b)

        if name_a is not None and name_a in param_names:
            c[param_names.index(name_a)] += 1.0
        if name_b is not None and name_b in param_names:
            c[param_names.index(name_b)] -= 1.0

        est = float(c @ params.values)
        se = float((c @ cov.values @ c) ** 0.5)
        z = est / se if se > 0 else float("nan")
        p_two_sided = 2 * (1 - norm.cdf(abs(z)))
        return {
            "estimate_logit": est,
            "se": se,
            "z": z,
            "p_value": p_two_sided,
        }

    comparisons = {}
    human = "Homo sapiens"
    for other in ["Pan", "Pongo", "Papio"]:
        if other in df["genus"].unique():
            comparisons[f"{human} vs {other}"] = contrast_logit(human, other)

    # Decide on binary response: Yes if humans have significantly higher AMTL
    # than all non-human genera at alpha = 0.05.
    response = "Yes"
    for label, stats in comparisons.items():
        if not (stats["estimate_logit"] > 0 and stats["p_value"] < 0.05):
            response = "No"
            break

    # Build explanation string summarizing model and key results
    lines = []
    lines.append(
        "I fit a binomial regression model for the proportion of antemortem tooth loss "
        "(num_amtl / sockets) with genus, age, sex (prob_male), and tooth class as predictors, "
        "using sockets as binomial trial weights."
    )
    lines.append(
        f"The model included {len(df)} genus–tooth_class observations across the four genera "
        f"{sorted(df['genus'].unique().tolist())}."
    )
    for label, stats in comparisons.items():
        direction = "higher" if stats["estimate_logit"] > 0 else "lower"
        lines.append(
            f"For {label}, the estimated log-odds difference in AMTL is {stats['estimate_logit']:.3f} "
            f"(SE = {stats['se']:.3f}, z = {stats['z']:.2f}, p = {stats['p_value']:.3f}), "
            f"indicating {direction} AMTL in {label.split(' vs ')[0]}."
        )
    if response == "Yes":
        lines.append(
            "Because Homo sapiens shows significantly higher AMTL than each non-human genus after "
            "adjusting for age, sex, and tooth class (all p < 0.05), I conclude that modern humans "
            "do have higher frequencies of AMTL than non-human primates under this model."
        )
    else:
        lines.append(
            "At least one comparison between Homo sapiens and a non-human genus does not show a "
            "significant increase in AMTL for humans at the 0.05 level, so the data do not support "
            "a consistent pattern of higher human AMTL frequencies once age, sex, and tooth class "
            "are taken into account."
        )

    explanation = " ".join(lines)

    conclusion = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()
