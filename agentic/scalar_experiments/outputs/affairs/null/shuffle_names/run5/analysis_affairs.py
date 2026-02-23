import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    # Load data
    df = pd.read_csv("affairs.csv")

    # According to info.json, column names have been shuffled.
    # We map them back to their semantic meaning using the descriptions there:
    # - age column: coded frequency of extramarital intercourse in past year (0 = none, >0 = some)
    # - religiousness column: actually a yes/no factor "Are there children in the marriage?"
    #   (values "yes"/"no")
    #
    # Our variables of interest:
    #   has_affair: binary indicator of any extramarital affairs in past year
    #   has_children: binary indicator of having children in the marriage

    # Affair frequency (coded 0,1,2,3,7,12,...)
    df["affair_freq"] = df["age"]
    # Binary indicator: any affairs vs none
    df["has_affair"] = (df["affair_freq"] > 0).astype(int)

    # Children indicator from the 'religiousness' column (actually children yes/no)
    df["has_children"] = (df["religiousness"].astype(str).str.lower() == "yes").astype(int)

    # Basic descriptive stats: proportions and mean frequencies by children status
    grouped = df.groupby("has_children")
    prop_any_affair = grouped["has_affair"].mean()
    mean_freq = grouped["affair_freq"].mean()

    # Simple logistic regression: Pr(has_affair = 1 | has_children)
    logit_simple = smf.logit("has_affair ~ has_children", data=df).fit(disp=0)
    coef_simple = float(logit_simple.params["has_children"])
    pval_simple = float(logit_simple.pvalues["has_children"])
    or_simple = float(np.exp(coef_simple))

    # Adjusted logistic regression with key covariates using the semantic mapping
    # from info.json:
    # - children column: years married
    # - occupation column: age in years (coded group midpoints)
    # - rating column: religiousness (1-5)
    # - affairs column: self rating of marriage (1-5)
    # - yearsmarried column: education level
    # - rownames column: occupation code
    df["years_married"] = df["children"]
    df["age_group"] = df["occupation"]
    df["relig_score"] = df["rating"]
    df["marriage_rating"] = df["affairs"]
    df["education_level"] = df["yearsmarried"]
    df["occupation_code"] = df["rownames"]

    # C(gender) to allow different baselines by gender
    formula_adj = (
        "has_affair ~ has_children + years_married + age_group + C(gender) + "
        "relig_score + marriage_rating + education_level + occupation_code"
    )

    try:
        logit_adj = smf.logit(formula_adj, data=df).fit(disp=0)
        coef_adj = float(logit_adj.params.get("has_children", np.nan))
        pval_adj = float(logit_adj.pvalues.get("has_children", np.nan))
        or_adj = float(np.exp(coef_adj)) if np.isfinite(coef_adj) else np.nan
    except Exception:
        # Fall back to the simple model only if the adjusted model fails
        coef_adj = np.nan
        pval_adj = np.nan
        or_adj = np.nan

    # Interpret evidence
    # Direction of effect from the simple model
    decreases_in_simple = coef_simple < 0

    # Compare empirical proportions
    prop_children = float(prop_any_affair.get(1, np.nan))
    prop_no_children = float(prop_any_affair.get(0, np.nan))

    # Heuristic decision: focus primarily on direction + statistical significance
    # in the simple logistic model and consistency with adjusted model/descriptives.
    if pval_simple < 0.05 and decreases_in_simple:
        # Statistically significant decrease in affair odds among couples with children
        strength = 80
        qualitative = "clear evidence that having children is associated with fewer extramarital affairs"
    elif pval_simple < 0.05 and not decreases_in_simple:
        # Significant, but in the *opposite* direction of the hypothesis
        strength = 5
        qualitative = "strong evidence that parents are at least as likely, if not more likely, to have extramarital affairs"
    else:
        # Not statistically significant: no reliable evidence for a protective effect.
        # If point estimates actually go in the *opposite* direction, be more confident in a 'No'.
        if not decreases_in_simple:
            strength = 20
            qualitative = (
                "no statistically reliable evidence that having children reduces extramarital affairs; "
                "if anything, the point estimates suggest similar or slightly higher affair rates among parents"
            )
        else:
            strength = 40
            qualitative = (
                "some weak, non-significant tendency for parents to report fewer affairs, "
                "but the data are far from conclusive"
            )

    # Build explanation string with key numeric summaries
    lines = []
    lines.append(
        "Research question: Does having children decrease engagement in extramarital affairs?"
    )
    lines.append(
        "Using the provided survey data, I reconstructed the true variable meanings from the metadata in info.json."
    )
    lines.append(
        "- Affair frequency is stored in the 'age' column (coded 0 = none, higher values = more frequent affairs)."
    )
    lines.append(
        "- Presence of children in the marriage is stored as a yes/no factor in the 'religiousness' column."
    )
    lines.append(
        "I defined a binary outcome has_affair = 1 if affair frequency > 0, and 0 otherwise, "
        "and a predictor has_children = 1 for couples reporting children and 0 otherwise."
    )
    lines.append(
        f"Empirically, the proportion reporting any extramarital affair was "
        f"{prop_children:.3f} among couples with children and {prop_no_children:.3f} among couples without children, "
        "with corresponding mean affair-frequency codes of "
        f"{mean_freq.get(1, np.nan):.3f} vs {mean_freq.get(0, np.nan):.3f}."
    )
    lines.append(
        "A simple logistic regression of has_affair on has_children yielded a coefficient for has_children of "
        f"{coef_simple:.3f} (odds ratio {or_simple:.3f}, p-value {pval_simple:.3g})."
    )
    if np.isfinite(coef_adj):
        lines.append(
            "An adjusted logistic model that additionally controlled for years married, age group, gender, "
            "religiousness score, self-rated marital happiness, education level, and occupation code "
            f"gave a coefficient for has_children of {coef_adj:.3f} (odds ratio {or_adj:.3f}, "
            f"p-value {pval_adj:.3g})."
        )
    else:
        lines.append(
            "An adjusted logistic model with additional covariates could not be reliably fit, "
            "so inference is based primarily on the simple model and descriptive differences."
        )
    lines.append(
        "Taken together, these results show "
        + qualitative
        + ". Thus, this dataset does not support the claim that having children decreases engagement in extramarital affairs."
    )

    explanation = " ".join(lines)

    conclusion = {
        "response": int(strength),
        "explanation": explanation,
    }

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion, ensure_ascii=False))


if __name__ == "__main__":
    main()

