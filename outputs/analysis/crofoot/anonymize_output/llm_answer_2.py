def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, 95% CIs, and odds ratios (with CIs)
    for the key predictors in the fitted GLM:
      - RelativeSize (main effect)
      - LocationAdvBinary (main effect)
      - RelativeSize:LocationAdvBinary (interaction)

    Returns:
      {
        "object": {
           "RelativeSize": {coef, se, z, p, ci_lower, ci_upper, odds_ratio, or_ci_lower, or_ci_upper},
           "LocationAdvBinary": {...},
           "Interaction": {...},
           "model_converged": bool,
           "params_all": pandas.Series (all coefficients),
        },
        "description": str (brief interpretation guidance)
      }
    The function tries to be robust to whether the results object is a raw GLMResultsWrapper
    or a robustified results object returned by get_robustcov_results.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic availability checks
    if not hasattr(res, "params"):
        raise AttributeError("model_output does not have .params attribute; not a statsmodels results object")

    params = res.params
    # Some results objects have .bse; others have cov_params() from which we can get bse
    if hasattr(res, "bse") and res.bse is not None:
        bse = res.bse
    else:
        # try to compute bse from covariance matrix
        try:
            cov = res.cov_params()
            bse = np.sqrt(np.diag(cov))
            bse = pd.Series(bse, index=params.index)
        except Exception:
            raise AttributeError("Could not obtain standard errors from model_output (no .bse or .cov_params())")

    # p-values: some robust results supply .pvalues; if not, compute using normal approx
    if hasattr(res, "pvalues") and res.pvalues is not None:
        pvals = res.pvalues
    else:
        zvals = params / bse
        from scipy import stats
        pvals = 2 * (1 - stats.norm.cdf(np.abs(zvals)))
        pvals = pd.Series(pvals, index=params.index)

    # Confidence intervals: try res.conf_int(), otherwise use normal approx
    try:
        ci = res.conf_int()
        # conf_int() may return a DataFrame or ndarray; convert to DataFrame with proper index and columns
        if isinstance(ci, (list, tuple, np.ndarray)):
            ci = pd.DataFrame(ci, index=params.index, columns=["2.5%", "97.5%"])
        else:
            ci = pd.DataFrame(ci, index=params.index, columns=["2.5%", "97.5%"])
    except Exception:
        z_crit = 1.96
        ci_lower = params - z_crit * bse
        ci_upper = params + z_crit * bse
        ci = pd.DataFrame({"2.5%": ci_lower, "97.5%": ci_upper}, index=params.index)

    # Helper to find term names robustly
    idx = list(params.index)

    def find_term(names):
        # names: list of candidate substrings that should uniquely identify the term
        for nm in idx:
            ok = all(part in nm for part in names)
            if ok:
                return nm
        return None

    # Identify the names for the three quantities
    name_rel = find_term(["RelativeSize"])
    name_loc = find_term(["LocationAdvBinary"])  # main effect
    # Interaction usually appears as 'RelativeSize:LocationAdvBinary' or 'RelativeSize:LocationAdvBinary'
    name_int = None
    # search for any index containing both tokens and a separator (':')
    for nm in idx:
        if ":" in nm and "RelativeSize" in nm and "LocationAdvBinary" in nm:
            name_int = nm
            break
    # fallback: any name that contains both substrings even without ':'
    if name_int is None:
        name_int = find_term(["RelativeSize", "LocationAdvBinary"])

    results_summary = {}

    # function to build a summary dict for a term name
    def summarize_term(term_name):
        if term_name is None or term_name not in params.index:
            return None
        coef = float(params[term_name])
        se = float(bse[term_name])
        z = coef / se if se != 0 else np.nan
        p = float(pvals[term_name]) if term_name in pvals.index else np.nan
        ci_lower = float(ci.loc[term_name, "2.5%"])
        ci_upper = float(ci.loc[term_name, "97.5%"])
        or_coef = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower))
        or_ci_upper = float(np.exp(ci_upper))
        return {
            "term_name": term_name,
            "coef": coef,
            "se": se,
            "z": z,
            "p_value": p,
            "ci_2.5%": ci_lower,
            "ci_97.5%": ci_upper,
            "odds_ratio": or_coef,
            "or_ci_2.5%": or_ci_lower,
            "or_ci_97.5%": or_ci_upper,
        }

    results_summary["RelativeSize"] = summarize_term(name_rel)
    results_summary["LocationAdvBinary"] = summarize_term(name_loc)
    results_summary["Interaction"] = summarize_term(name_int)

    # add everything else for transparency
    results_summary["model_converged"] = bool(getattr(res, "converged", True))
    # include full params as pandas Series for inspection
    results_summary["params_all"] = params

    # Build a concise description about what these stats mean in context
    description_lines = [
        "Extracted coefficients, standard errors, z-statistics, two-sided p-values, 95% confidence intervals,",
        "and odds ratios (with 95% CIs) for the predictors relevant to the question:",
        "- RelativeSize: tests whether larger focal group size (vs. opponent) changes log-odds of winning.",
        "- LocationAdvBinary: tests whether being closer to the focal group's home-range center (home-field) changes log-odds of winning.",
        "- Interaction (RelativeSize x LocationAdvBinary): tests whether the effect of relative size differs when the contest is on the focal group's home range.",
        "",
        "Interpretation guidance:",
        "- A positive coef for RelativeSize means that as the focal group is larger than the opponent, the log-odds (and odds) of winning increase. The odds_ratio > 1 indicates the multiplicative change in odds per unit increase in RelativeSize.",
        "- A positive coef for LocationAdvBinary means contests nearer the focal group's center increase its chance of winning (binary home-field advantage).",
        "- A significant interaction (p < 0.05) indicates that the size advantage effect depends on contest location; inspect the interaction coef and/or plot predicted probabilities to interpret direction and magnitude.",
        "",
        "The 'object' field contains numeric values you can use to make a yes/no decision (e.g., check p-values and confidence intervals)."
    ]
    description = " ".join(description_lines)

    return {"object": results_summary, "description": description}