def extract_final_answer(model_output):
    """
    Extracts age-related effects (main effect and Age x Culture interactions)
    from a fitted statsmodels GLMResults-like object (possibly with clustered
    covariances already applied).

    Returns:
      {
        "object": {
           "reference": { "coef_logodds": ..., "se": ..., "z": ..., "p": ...,
                          "95ci_logodds": (low, high),
                          "OR": ..., "95ci_OR": (low_or, high_or) },
           "cultures": {
              "<CultureLabel>": { same fields as reference, but represents the
                                  age slope (log-odds) for that culture (i.e.
                                  reference slope + interaction) },
              ...
           }
        },
        "description": "Plain-language summary of what the numbers mean..."
      }
    """
    import re
    import math
    import numpy as np
    from scipy.stats import norm

    # Helper: safe access to params and cov
    try:
        params = model_output.params  # pandas Series
        cov = model_output.cov_params()  # DataFrame
    except Exception as e:
        raise ValueError(f"Model output does not have expected attributes: {e}")

    # Check that AgeCentered main effect exists
    base_term = "AgeCentered"
    if base_term not in params.index:
        raise ValueError(f"Expected main effect parameter '{base_term}' not found in model parameters. "
                         f"Found parameters: {list(params.index)}")

    # Find interaction terms of the form AgeCentered:C(Culture)[T.<label>]
    interaction_pattern = re.compile(r"^AgeCentered:C\(Culture\)\[T\.(.+)\]$")
    interactions = {}
    for name in params.index:
        m = interaction_pattern.match(name)
        if m:
            label = m.group(1)
            interactions[label] = name

    # Utility to compute linear combination estimate, se, z, p, CI, OR
    def linear_combination_stats(term_coeffs):
        # term_coeffs: dict mapping parameter name -> multiplier (usually 1 or sum)
        # Build estimate
        est = 0.0
        for t, mult in term_coeffs.items():
            if t not in params.index:
                raise KeyError(f"Parameter '{t}' not found in model parameters.")
            est += float(params.loc[t]) * mult
        # variance: v = c' * Cov * c
        terms = list(term_coeffs.keys())
        coefs = np.array([term_coeffs[t] for t in terms], dtype=float)
        cov_subset = cov.loc[terms, terms].values
        var = float(coefs.dot(cov_subset).dot(coefs))
        se = math.sqrt(var) if var >= 0 else float("nan")
        z = est / se if se and not math.isnan(se) else float("nan")
        p = 2.0 * norm.sf(abs(z)) if not math.isnan(z) else float("nan")
        ci_low = est - 1.96 * se
        ci_high = est + 1.96 * se
        # Odds ratio interpretation per one-unit (year) increase
        OR = math.exp(est)
        OR_ci = (math.exp(ci_low), math.exp(ci_high))
        return {
            "coef_logodds": est,
            "se": se,
            "z": z,
            "p": p,
            "95ci_logodds": (ci_low, ci_high),
            "OR": OR,
            "95ci_OR": OR_ci
        }

    results = {"reference": None, "cultures": {}}

    # Reference (baseline culture) slope is just AgeCentered
    try:
        results["reference"] = linear_combination_stats({base_term: 1.0})
    except Exception as e:
        raise RuntimeError(f"Failed computing stats for reference AgeCentered term: {e}")

    # For each identified culture interaction, compute slope = AgeCentered + interaction
    for label, interaction_param in interactions.items():
        try:
            res = linear_combination_stats({base_term: 1.0, interaction_param: 1.0})
            results["cultures"][label] = res
        except Exception as e:
            # if something fails for a given culture, record the error message instead of raising
            results["cultures"][label] = {"error": str(e)}

    # Build a concise description
    desc_lines = []
    desc_lines.append("This extracts the estimated effect of age (per year) on choosing the majority option,")
    desc_lines.append("reported both on the log-odds scale and as an odds ratio (OR) with 95% CIs.")
    desc_lines.append("Interpretation: a positive log-odds (OR>1) means reliance on the majority increases with age;")
    desc_lines.append("a negative log-odds (OR<1) means reliance decreases with age. p-values are two-sided.")
    desc_lines.append("")  # blank line

    # Summarize reference
    ref = results["reference"]
    desc_lines.append(
        f"Reference (baseline culture) age slope: log-odds={ref['coef_logodds']:.3f}, se={ref['se']:.3f}, "
        f"p={ref['p']:.3f}. OR per year = {ref['OR']:.3f} "
        f"(95% CI {ref['95ci_OR'][0]:.3f}–{ref['95ci_OR'][1]:.3f})."
    )

    # Summarize each culture
    if results["cultures"]:
        for label, entry in results["cultures"].items():
            if "error" in entry:
                desc_lines.append(f"Culture '{label}': error computing stats: {entry['error']}")
            else:
                desc_lines.append(
                    f"Culture '{label}' age slope: log-odds={entry['coef_logodds']:.3f}, se={entry['se']:.3f}, "
                    f"p={entry['p']:.3f}. OR = {entry['OR']:.3f} "
                    f"(95% CI {entry['95ci_OR'][0]:.3f}–{entry['95ci_OR'][1]:.3f})."
                )

    description = " ".join(desc_lines)

    return {"object": results, "description": description}