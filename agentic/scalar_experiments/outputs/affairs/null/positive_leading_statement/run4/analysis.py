import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Basic derived variables
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Descriptive statistics by children status
    desc = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            median_affairs=("affairs", "median"),
            prop_any_affair=("any_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    print("Descriptive stats by children:")
    print(desc.to_string(index=False))
    print()

    # Simple comparison of means (Welch t-test)
    with_children = df[df["children"] == "yes"]["affairs"]
    without_children = df[df["children"] == "no"]["affairs"]

    from scipy import stats

    t_stat, p_val = stats.ttest_ind(
        with_children, without_children, equal_var=False
    )
    print("Welch t-test on mean affairs (children yes vs no):")
    print(f" t = {t_stat:.3f}, p = {p_val:.4f}")
    print()

    # Logistic regression on any affair with children only
    logit_simple = smf.logit("any_affair ~ C(children)", data=df).fit(disp=False)
    print("Logistic regression (any_affair ~ C(children)):")
    print(logit_simple.summary())
    print()

    # Logistic regression controlling for covariates
    logit_full = smf.logit(
        "any_affair ~ C(children) + age + yearsmarried + C(gender) + "
        "religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=False)
    print(
        "Logistic regression with controls "
        "(any_affair ~ children + demographics):"
    )
    print(logit_full.summary())
    print()

    # Poisson regression for counts as a robustness check
    poisson_full = smf.poisson(
        "affairs ~ C(children) + age + yearsmarried + C(gender) + "
        "religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=False)
    print(
        "Poisson regression for counts "
        "(affairs ~ children + demographics):"
    )
    print(poisson_full.summary())
    print()

    # Extract key effects for children from models
    # By construction, statsmodels uses the first sorted category as baseline.
    # Determine which is baseline so we interpret correctly.
    children_categories = sorted(df["children"].unique())
    # In practice, categories should be ['no', 'yes'] but we do not assume.
    print(f"Children categories (sorted): {children_categories}")

    params_logit = logit_full.params
    conf_logit = logit_full.conf_int()
    odds_ratios = np.exp(params_logit)

    # Find any parameter corresponding to children
    children_effects = []
    for name, coef in params_logit.items():
        if "C(children)" in name:
            ci_low, ci_high = conf_logit.loc[name]
            or_point = odds_ratios[name]
            or_low, or_high = np.exp(ci_low), np.exp(ci_high)
            p_value = logit_full.pvalues[name]
            children_effects.append(
                {
                    "term": name,
                    "coef": float(coef),
                    "p_value": float(p_value),
                    "or_point": float(or_point),
                    "or_low": float(or_low),
                    "or_high": float(or_high),
                }
            )

    print("Children-related coefficients in logistic model with controls:")
    for eff in children_effects:
        print(eff)
    print()

    # Decide overall answer based on evidence:
    # - Direction: whether odds ratios are below or above 1.
    # - Significance: p-values for children term(s).
    # - Descriptive differences in mean and proportion any affair.

    mean_affairs_with = float(desc.loc[desc["children"] == "yes", "mean_affairs"])
    mean_affairs_without = float(
        desc.loc[desc["children"] == "no", "mean_affairs"]
    )
    prop_any_with = float(desc.loc[desc["children"] == "yes", "prop_any_affair"])
    prop_any_without = float(
        desc.loc[desc["children"] == "no", "prop_any_affair"]
    )

    explanation_lines = []
    explanation_lines.append(
        "Research question: Does having children decrease engagement in "
        "extramarital affairs?"
    )
    explanation_lines.append(
        "I analyzed 601 married individuals from the Fair (1978) affairs dataset."
    )
    explanation_lines.append(
        f"Descriptively, the mean number of affairs is "
        f"{mean_affairs_with:.3f} for those with children and "
        f"{mean_affairs_without:.3f} for those without children."
    )
    explanation_lines.append(
        f"The proportion reporting any affair is "
        f"{prop_any_with:.3f} with children vs "
        f"{prop_any_without:.3f} without children."
    )
    explanation_lines.append(
        f"A Welch t-test comparing mean affair counts between those with and "
        f"without children yields p = {p_val:.4f}."
    )

    # Summarize logistic regression results for children term(s)
    if children_effects:
        for eff in children_effects:
            direction = "lower" if eff["or_point"] < 1 else "higher"
            explanation_lines.append(
                "In a logistic regression of having any affair on children, "
                "age, years married, gender, religiousness, education, "
                "occupation, and marital rating, the children term "
                f"{eff['term']} has an odds ratio of {eff['or_point']:.3f} "
                f"(95% CI {eff['or_low']:.3f}–{eff['or_high']:.3f}, "
                f"p = {eff['p_value']:.4f}), indicating {direction} odds of "
                "affairs relative to the reference group."
            )
    else:
        explanation_lines.append(
            "The logistic regression did not estimate a distinct coefficient "
            "for children (unexpected given the coding), so inference relies "
            "on descriptive statistics and other model terms."
        )

    # Interpret Poisson coefficient for children if present
    params_pois = poisson_full.params
    conf_pois = poisson_full.conf_int()
    irrs = np.exp(params_pois)
    children_pois_effects = []
    for name, coef in params_pois.items():
        if "C(children)" in name:
            ci_low, ci_high = conf_pois.loc[name]
            irr_point = irrs[name]
            irr_low, irr_high = np.exp(ci_low), np.exp(ci_high)
            p_value = poisson_full.pvalues[name]
            children_pois_effects.append(
                {
                    "term": name,
                    "coef": float(coef),
                    "p_value": float(p_value),
                    "irr_point": float(irr_point),
                    "irr_low": float(irr_low),
                    "irr_high": float(irr_high),
                }
            )

    if children_pois_effects:
        for eff in children_pois_effects:
            direction = "lower" if eff["irr_point"] < 1 else "higher"
            explanation_lines.append(
                "In a Poisson regression for the count of affairs with the "
                "same controls, the children term "
                f"{eff['term']} has an incidence rate ratio of "
                f"{eff['irr_point']:.3f} (95% CI {eff['irr_low']:.3f}–"
                f"{eff['irr_high']:.3f}, p = {eff['p_value']:.4f}), "
                f"indicating {direction} expected counts relative to the "
                "reference group."
            )

    # Determine qualitative conclusion and Likert scale value.
    # Default stance: base largely on sign and significance of children terms.
    response_value: int
    qualitative_conclusion: str

    # Look at the strongest (smallest p) children effect across models
    combined_effects = []
    for eff in children_effects:
        combined_effects.append(
            ("logit", eff["p_value"], eff["or_point"])
        )
    for eff in children_pois_effects:
        combined_effects.append(
            ("poisson", eff["p_value"], eff["irr_point"])
        )

    if combined_effects:
        # sort by p-value ascending
        combined_effects.sort(key=lambda x: x[1])
        best_model, best_p, best_ratio = combined_effects[0]
        if best_p < 0.05:
            if best_ratio < 1:
                qualitative_conclusion = (
                    "There is statistically significant evidence that having "
                    "children is associated with fewer extramarital affairs."
                )
                # Strength scaled by how far the ratio is from 1
                distance = abs(np.log(best_ratio))
                # Cap distance for mapping to 0–100
                # distance ~ 0.0 -> weak effect; >0.7 -> strong
                if distance < 0.1:
                    response_value = 60
                elif distance < 0.3:
                    response_value = 75
                elif distance < 0.7:
                    response_value = 85
                else:
                    response_value = 95
            else:
                qualitative_conclusion = (
                    "There is statistically significant evidence that having "
                    "children is associated with more extramarital affairs, "
                    "contrary to the hypothesis that children reduce affairs."
                )
                # For a significant increase, answer is a strong 'No'
                distance = abs(np.log(best_ratio))
                if distance < 0.1:
                    response_value = 40
                elif distance < 0.3:
                    response_value = 25
                elif distance < 0.7:
                    response_value = 10
                else:
                    response_value = 5
        else:
            qualitative_conclusion = (
                "There is no statistically significant evidence that having "
                "children changes the likelihood or frequency of extramarital "
                "affairs once other factors are controlled."
            )
            # No significant effect: this is effectively a 'No' answer
            # with moderate certainty that we do not detect an effect.
            response_value = 35
    else:
        qualitative_conclusion = (
            "Model results for the children variable were inconclusive, so "
            "I rely on descriptive comparisons, which do not show a clear "
            "reduction in affairs among those with children."
        )
        response_value = 30

    explanation_lines.append(qualitative_conclusion)

    explanation = " ".join(explanation_lines)

    conclusion = {
        "response": int(response_value),
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)

    print("Final response value:", response_value)
    print("Conclusion written to conclusion.txt")


if __name__ == "__main__":
    main()

