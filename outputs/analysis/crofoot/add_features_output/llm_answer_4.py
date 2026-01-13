def extract_final_answer(model_output):
    """
    Extract key statistics for the predictors of interest from a fitted statsmodels GLM/GLMResultsWrapper
    (possibly with cluster-robust covariance applied).

    Returns a dict with:
      - "object": dict keyed by predictor names with entries:
            {"coef", "se", "z", "p", "ci_lower", "ci_upper", "odds_ratio", "or_ci_lower", "or_ci_upper"}
        for the three terms: LogSizeRatio_z, HomeAdvantage_z, and their interaction.
      - "description": short plain-language interpretation of what each statistic means
        in the context of the question.

    The function is defensive about missing attributes and tries reasonable fallbacks.
    """
    import numpy as np

    # Terms we care about (statsmodels names interaction with ':')
    terms = ['LogSizeRatio_z', 'HomeAdvantage_z', 'LogSizeRatio_z:HomeAdvantage_z']

    # Prepare containers
    results = {}

    # Helper to safely pull arrays from model object
    try:
        params = model_output.params
    except Exception:
        raise ValueError("model_output has no .params attribute")

    # standard errors: robust result wrappers expose .bse
    if hasattr(model_output, 'bse'):
        bse = model_output.bse
    else:
        # try to compute from covariance matrix if available
        if hasattr(model_output, 'cov_params'):
            cov = model_output.cov_params()
            bse = np.sqrt(np.diag(cov))
            # convert to a pandas Series-like mapping if params is a Series
            try:
                import pandas as pd
                bse = pd.Series(bse, index=params.index)
            except Exception:
                pass
        else:
            raise ValueError("Cannot find standard errors (no .bse and no .cov_params())")

    # p-values: available as .pvalues for most statsmodels results
    pvalues = getattr(model_output, 'pvalues', None)

    # confidence intervals: try model_output.conf_int(), else compute from bse (normal approx)
    try:
        ci = model_output.conf_int()
        # ensure it's a DataFrame-like with same index as params
    except Exception:
        # use normal approximation (z=1.96)
        z = 1.96
        ci = {}
        for name, val in params.items():
            se_val = bse[name] if name in bse else np.nan
            ci[name] = [val - z * se_val, val + z * se_val]
        # convert to pandas-like structure if params is a Series
        try:
            import pandas as pd
            ci = pd.DataFrame.from_dict(ci, orient='index', columns=[0, 1])
        except Exception:
            pass

    # Build summaries for each term
    for term in terms:
        if term in params.index:
            coef = float(params[term])
            se = float(bse[term]) if term in bse.index else float(bse[term])
            z_stat = coef / se if se != 0 else np.nan
            p = float(pvalues[term]) if (pvalues is not None and term in pvalues.index) else None
            ci_lower = float(ci.loc[term, 0]) if hasattr(ci, 'loc') and term in ci.index else (coef - 1.96 * se)
            ci_upper = float(ci.loc[term, 1]) if hasattr(ci, 'loc') and term in ci.index else (coef + 1.96 * se)

            # odds ratio and CI on odds ratio scale
            or_point = float(np.exp(coef))
            or_ci_lower = float(np.exp(ci_lower))
            or_ci_upper = float(np.exp(ci_upper))

            results[term] = {
                "coef": coef,
                "se": se,
                "z": z_stat,
                "p": p,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "odds_ratio": or_point,
                "or_ci_lower": or_ci_lower,
                "or_ci_upper": or_ci_upper
            }
        else:
            results[term] = None  # term not present in model

    # Compose a short interpretation
    def interpret_entry(name, entry):
        if entry is None:
            return f"{name}: term not present in model."
        sig = ""
        if entry["p"] is not None:
            sig = "statistically significant (p < 0.05)" if entry["p"] < 0.05 else "not statistically significant (p >= 0.05)"
        return (
            f"{name}: coef={entry['coef']:.3f}, SE={entry['se']:.3f}, z={entry['z']:.2f}, p={entry['p']:.3f} -> "
            f"odds ratio={entry['odds_ratio']:.3f} (95% CI [{entry['or_ci_lower']:.3f}, {entry['or_ci_upper']:.3f}]); "
            f"interpretation: positive coef means higher {name.replace(':',' x ')} increases the log-odds of the focal group winning; {sig}."
        )

    descriptions = []
    for t in terms:
        descriptions.append(interpret_entry(t, results[t]))

    overall_description = (
        "Extracted coefficients, standard errors, p-values and 95% CIs (on both log-odds and odds-ratio scales) "
        "for the main predictors and their interaction. In logistic regression: a positive coefficient means an increase "
        "in that (standardized) predictor is associated with higher probability of the focal group winning. "
        "The interaction term (LogSizeRatio_z:HomeAdvantage_z) indicates whether the effect of relative group size "
        "on winning depends on contest location (home advantage). "
        "Below are per-term summaries:\n- " + "\n- ".join(descriptions)
    )

    return {"object": results, "description": overall_description}