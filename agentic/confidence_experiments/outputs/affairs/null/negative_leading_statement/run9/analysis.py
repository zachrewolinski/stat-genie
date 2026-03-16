import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE_DIR = Path(__file__).resolve().parent

def load_data():
    df = pd.read_csv(BASE_DIR / "affairs.csv")
    return df


def describe_children_affairs(df: pd.DataFrame):
    # Binary indicator of any affairs
    df = df.copy()
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    summary = {}
    for has_children, group in df.groupby("children"):
        n = len(group)
        mean_affairs = group["affairs"].mean()
        prop_any = group["any_affair"].mean()
        summary[has_children] = {
            "n": int(n),
            "mean_affairs": float(mean_affairs),
            "prop_any_affair": float(prop_any),
        }

    return summary, df


def compare_means_and_proportions(df: pd.DataFrame):
    from statsmodels.stats.weightstats import ttest_ind
    from statsmodels.stats.proportion import proportions_ztest

    df_yes = df[df["children"] == "yes"]
    df_no = df[df["children"] == "no"]

    # T-test on number of affairs
    t_stat, p_val, _ = ttest_ind(df_yes["affairs"], df_no["affairs"], usevar="unequal")

    # Proportion test on any affair
    count = np.array([df_yes["any_affair"].sum(), df_no["any_affair"].sum()])
    nobs = np.array([len(df_yes), len(df_no)])
    z_stat, p_prop = proportions_ztest(count, nobs)

    return {
        "t_stat_affairs": float(t_stat),
        "p_val_affairs": float(p_val),
        "z_stat_any_affair": float(z_stat),
        "p_val_any_affair": float(p_prop),
        "mean_affairs_yes_children": float(df_yes["affairs"].mean()),
        "mean_affairs_no_children": float(df_no["affairs"].mean()),
        "prop_any_yes_children": float(df_yes["any_affair"].mean()),
        "prop_any_no_children": float(df_no["any_affair"].mean()),
    }


def regression_analysis(df: pd.DataFrame):
    # Use a Poisson regression for count of affairs as a function of children status and controls
    df = df.copy()
    df["children_binary"] = (df["children"] == "yes").astype(int)

    formula = "affairs ~ children_binary + age + yearsmarried + religiousness + education + C(occupation) + rating + C(gender)"

    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
    result = model.fit()

    coef = result.params["children_binary"]
    se = result.bse["children_binary"]
    p_val = result.pvalues["children_binary"]
    irr = float(np.exp(coef))

    return {
        "children_coef": float(coef),
        "children_se": float(se),
        "children_p_val": float(p_val),
        "children_incidence_rate_ratio": irr,
    }


def main():
    df = load_data()
    summary, df_any = describe_children_affairs(df)
    comp = compare_means_and_proportions(df_any)
    reg = regression_analysis(df_any)

    # Determine Likert-style response
    # Null belief: having children decreases engagement in extramarital affairs.
    # We assess direction and significance of children effect.
    p = reg["children_p_val"]
    irr = reg["children_incidence_rate_ratio"]

    # If IRR is clearly <1 and significant, this would support "Yes, children decrease affairs".
    # If IRR is >=1 or not significant, we move towards "No".
    if p < 0.01 and irr < 0.9:
        response = 80
    elif p < 0.05 and irr < 1.0:
        response = 65
    elif p < 0.1 and irr < 1.0:
        response = 55
    elif p >= 0.1 and irr < 1.0:
        response = 45
    elif p < 0.05 and irr >= 1.0:
        response = 25
    elif p < 0.1 and irr >= 1.0:
        response = 35
    else:
        response = 20

    explanation_lines = []
    explanation_lines.append(
        "We examined whether having children is associated with lower engagement in extramarital affairs."
    )
    explanation_lines.append(
        f"Descriptively, the mean number of affairs in the last year was {comp['mean_affairs_yes_children']:.2f} for people with children and {comp['mean_affairs_no_children']:.2f} for people without children."
    )
    explanation_lines.append(
        f"The proportion having any affair was {comp['prop_any_yes_children']:.2%} with children vs {comp['prop_any_no_children']:.2%} without children."
    )
    explanation_lines.append(
        f"A Welch t-test comparing average affair counts between groups yielded p = {comp['p_val_affairs']:.3f}, and a z-test for proportions yielded p = {comp['p_val_any_affair']:.3f}."
    )
    explanation_lines.append(
        "We also estimated a Poisson regression of affair counts on having children controlling for age, years married, religiousness, education, occupation, marital rating, and gender."
    )
    explanation_lines.append(
        f"In this model, the incidence rate ratio for having children was {reg['children_incidence_rate_ratio']:.2f} with p = {reg['children_p_val']:.3f}."
    )

    if response >= 50:
        explanation_lines.append(
            "Overall, the direction and statistical evidence are at most weakly consistent with a modest decrease in affairs for couples with children, and we do not find strong evidence for a substantial effect."
        )
    else:
        explanation_lines.append(
            "Overall, the direction and statistical evidence do not support the claim that having children meaningfully decreases engagement in extramarital affairs."
        )

    explanation_lines.append(
        f"On a 0–100 scale where higher values indicate stronger evidence that having children decreases affairs, we assign a score of {response}."
    )

    explanation = " " .join(explanation_lines)

    conclusion = {"response": int(response), "explanation": explanation}

    out_path = BASE_DIR / "conclusion.txt"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
