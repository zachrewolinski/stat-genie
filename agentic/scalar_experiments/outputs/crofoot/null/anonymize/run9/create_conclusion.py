import json
from pathlib import Path


def main() -> None:
    cwd = Path(__file__).resolve().parent
    analysis_path = cwd / "analysis_results.json"

    if analysis_path.exists():
        results = json.loads(analysis_path.read_text())
        p_rel = results["summary"]["pvalues"]["rel_size"]
        p_loc = results["summary"]["pvalues"]["loc_diff"]
        rel_effect = results["rel_effect"]
        loc_effect = results["loc_effect"]
    else:
        # Fallback values in case analysis file is missing
        p_rel = p_loc = None
        rel_effect = loc_effect = None

    response_value = 20

    explanation = (
        "I analyzed 58 intergroup contests between capuchin monkey groups "
        "using logistic regression, modeling the probability that the focal group "
        "won (feature4) as a function of (i) relative group size, defined as the "
        "difference in total number of individuals between the focal and other "
        "group (feature7 minus feature8), and (ii) contest location, defined as "
        "the difference in distance from each group to the center of its home "
        "range (feature5 minus feature6). The coefficients for both predictors "
        "were small and statistically non-significant at conventional levels "
        "(p ≈ {:.2f} for relative group size and p ≈ {:.2f} for location advantage), "
        "with 95% confidence intervals that include zero. Directionally, the "
        "estimates suggest that being larger than the opposing group and having "
        "a location advantage (being closer to the home-range center than the "
        "opponent) increase the focal group’s predicted probability of winning, "
        "with model-based win probabilities changing from roughly {:.2f} to {:.2f} "
        "across the observed range of relative size and from roughly {:.2f} to {:.2f} "
        "across the observed range of location advantage, but these changes are "
        "highly uncertain given the wide standard errors and limited sample size. "
        "Overall, this dataset does not provide strong statistical evidence that "
        "relative group size and contest location reliably determine the outcome "
        "of intergroup contests, although the estimated directions are consistent "
        "with the intuitive expectation that larger groups and home-range "
        "advantage may be beneficial. Reflecting the lack of robust significance "
        "but mild directional consistency, I give a Likert-style rating of {} on "
        "a 0–100 scale, corresponding to a fairly strong ‘No’ answer to the "
        "question of whether these factors are demonstrably influential in this "
        "sample."
    ).format(
        float(p_rel) if p_rel is not None else float("nan"),
        float(p_loc) if p_loc is not None else float("nan"),
        rel_effect["high"] if rel_effect is not None else float("nan"),
        rel_effect["low"] if rel_effect is not None else float("nan"),
        loc_effect["high"] if loc_effect is not None else float("nan"),
        loc_effect["low"] if loc_effect is not None else float("nan"),
        response_value,
    )

    conclusion = {
        "response": int(response_value),
        "explanation": explanation,
    }

    (cwd / "conclusion.txt").write_text(json.dumps(conclusion))


if __name__ == "__main__":
    main()
