import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def build_conclusion(results: Dict[str, Any]) -> Tuple[int, str]:
    """
    Build the scalar response (0–100) and textual explanation
    from the computed descriptive statistics and regression outputs.
    """
    mean_affairs = results["mean_affairs_by_children"]
    prop_any = results["prop_any_affair_by_children"]
    t_res = results["t_test_affairs_children"]
    logit = results.get("logit_children_effect", {})
    pois = results.get("poisson_children_effect", {})
    nb = results.get("negbin_children_effect", {})

    mean_no = mean_affairs.get("no")
    mean_yes = mean_affairs.get("yes")
    prop_no = prop_any.get("no")
    prop_yes = prop_any.get("yes")

    t_p = t_res.get("p_value")
    logit_or = logit.get("odds_ratio")
    logit_p = logit.get("p_value")
    pois_rr = pois.get("rate_ratio")
    pois_p = pois.get("p_value")
    nb_rr = nb.get("rate_ratio")
    nb_p = nb.get("p_value")

    # Interpret evidence:
    # - Simple comparisons and logistic regression show little to no
    #   effect of children on *whether* any affair occurs.
    # - Count models (Poisson and negative binomial) indicate that,
    #   conditional on covariates, having children is associated with
    #   a modest but statistically significant reduction in the *number*
    #   of affairs (nb_p ~ 0.03 with rate ratio ~0.77).
    #
    # Overall we treat this as moderate, not strong, evidence that
    # having children is associated with lower engagement in affairs.
    response = 60

    explanation = (
        "Using the 601 married respondents in the dataset, I examined whether having children "
        "is associated with lower engagement in extramarital affairs. Descriptively, respondents "
        f"without children reported an average of about {mean_no:.2f} affairs in the past year, "
        f"versus {mean_yes:.2f} among those with children, and the share reporting at least one affair "
        f"was roughly {prop_no:.1%} for those without children and {prop_yes:.1%} for those with children. "
        f"A simple two-sample t-test on the mean number of affairs by children status was not statistically "
        f"significant (p ≈ {t_p:.3f}), and a logistic regression for having any affair showed essentially no "
        f"effect of children status (odds ratio for having children ≈ {logit_or:.2f}, p ≈ {logit_p:.3f}). "
        "To better capture engagement (frequency of affairs), I then modeled the affair counts. "
        "A Poisson regression including controls for age, years married, religiousness, education, occupation, "
        f"marital satisfaction rating, and gender estimated that respondents with children have about "
        f"{(1 - pois_rr) * 100:.0f}% fewer affairs on average than comparable respondents without children "
        f"(rate ratio ≈ {pois_rr:.2f}, p ≈ {pois_p:.3f}). A negative binomial regression, which is more appropriate "
        f"for overdispersed count data, yielded a similar estimated reduction of roughly {(1 - nb_rr) * 100:.0f}% "
        f"(rate ratio ≈ {nb_rr:.2f}, p ≈ {nb_p:.3f}). "
        "These results indicate that having children does not materially change the probability of having any affair at all, "
        "but conditional on observed covariates it is associated with a modest, statistically significant reduction in the "
        "number of extramarital affairs. Because the reduction is moderate in size and the evidence depends somewhat on the "
        "modeling choice, I regard this as moderate rather than strong evidence that having children decreases engagement in "
        "extramarital affairs, and I therefore answer 'Yes' to the research question with a score of 60 on a 0–100 scale, "
        "where 0 represents a strong 'No' and 100 represents a strong 'Yes'."
    )

    return response, explanation


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator for having any extramarital affair in the past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    results = {}

    # Descriptive statistics by presence of children.
    group = df.groupby("children")
    mean_affairs = group["affairs"].mean()
    prop_any = group["any_affair"].mean()

    results["mean_affairs_by_children"] = {
        k: float(v) for k, v in mean_affairs.to_dict().items()
    }
    results["prop_any_affair_by_children"] = {
        k: float(v) for k, v in prop_any.to_dict().items()
    }

    # Two-sample t-test for difference in average number of affairs.
    yes_affairs = df.loc[df["children"] == "yes", "affairs"]
    no_affairs = df.loc[df["children"] == "no", "affairs"]
    t_stat, p_t = stats.ttest_ind(yes_affairs, no_affairs, equal_var=False)
    results["t_test_affairs_children"] = {
        "t_stat": float(t_stat),
        "p_value": float(p_t),
        "n_yes": int(yes_affairs.shape[0]),
        "n_no": int(no_affairs.shape[0]),
    }

    # Logistic regression: probability of any affair ~ children + controls.
    logit_formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    logit_model = smf.logit(logit_formula, data=df).fit(disp=False)
    logit_params = logit_model.params.to_dict()
    logit_pvalues = logit_model.pvalues.to_dict()

    results["logit_params"] = {k: float(v) for k, v in logit_params.items()}
    results["logit_pvalues"] = {k: float(v) for k, v in logit_pvalues.items()}

    if "C(children)[T.yes]" in logit_params:
        coef_child = float(logit_params["C(children)[T.yes]"])
        or_child = float(np.exp(coef_child))
        p_child = float(logit_pvalues["C(children)[T.yes]"])
        results["logit_children_effect"] = {
            "coef": coef_child,
            "odds_ratio": or_child,
            "p_value": p_child,
        }

    # Poisson regression for affair counts (as a robustness check).
    pois_formula = (
        "affairs ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating + C(gender)"
    )
    pois_model = smf.poisson(pois_formula, data=df).fit(disp=False)
    pois_params = pois_model.params.to_dict()
    pois_pvalues = pois_model.pvalues.to_dict()

    results["poisson_params"] = {k: float(v) for k, v in pois_params.items()}
    results["poisson_pvalues"] = {k: float(v) for k, v in pois_pvalues.items()}

    if "C(children)[T.yes]" in pois_params:
        coef_child_pois = float(pois_params["C(children)[T.yes]"])
        rr_child = float(np.exp(coef_child_pois))
        p_child_pois = float(pois_pvalues["C(children)[T.yes]"])
        results["poisson_children_effect"] = {
            "coef": coef_child_pois,
            "rate_ratio": rr_child,
            "p_value": p_child_pois,
        }

    # Negative binomial regression as a robustness check for overdispersion.
    nb_model = smf.glm(
        pois_formula,
        data=df,
        family=sm.families.NegativeBinomial(),
    ).fit()
    nb_params = nb_model.params.to_dict()
    nb_pvalues = nb_model.pvalues.to_dict()
    results["negbin_params"] = {k: float(v) for k, v in nb_params.items()}
    results["negbin_pvalues"] = {k: float(v) for k, v in nb_pvalues.items()}

    if "C(children)[T.yes]" in nb_params:
        coef_child_nb = float(nb_params["C(children)[T.yes]"])
        rr_child_nb = float(np.exp(coef_child_nb))
        p_child_nb = float(nb_pvalues["C(children)[T.yes]"])
        results["negbin_children_effect"] = {
            "coef": coef_child_nb,
            "rate_ratio": rr_child_nb,
            "p_value": p_child_nb,
        }

    # Print JSON with detailed statistical results for inspection.
    print(json.dumps(results, indent=2))

    # Build scalar response and explanation, then write conclusion.txt.
    response, explanation = build_conclusion(results)
    conclusion = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
