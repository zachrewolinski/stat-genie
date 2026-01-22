def extract_final_answer(model_output):
    """
    Extracts statistics about the 'is_human' effect from the model_output dict
    returned by the modeling function.

    Returns a dictionary:
      - "object": dict with numeric results (coef, se, z, p_value, odds_ratio, ci)
      - "description": brief interpretation stating whether modern humans have
                       higher AMTL after accounting for covariates
    """
    import math
    import numpy as np

    result = {
        "object": None,
        "description": ""
    }

    # Helper to compute two-tailed p-value from z using normal distribution (no scipy)
    def z_to_p(z):
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))

    # Try to get the fitted results object
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('glm_clustered', None)
    if res is None:
        # Maybe the user passed the raw results object directly
        res = model_output

    # Prepare containers for extracted values
    coef = se = pval = None
    odds_ratio = ci_low = ci_high = None
    z = None

    # Try extraction from statsmodels-like results object
    try:
        params = getattr(res, 'params', None)
        bse = getattr(res, 'bse', None)
        pvalues = getattr(res, 'pvalues', None)

        if params is not None and 'is_human' in params.index:
            coef = float(params['is_human'])
            if bse is not None and 'is_human' in bse.index:
                se = float(bse['is_human'])
            # p-value: prefer provided pvalues, otherwise compute from z
            if pvalues is not None and 'is_human' in pvalues.index:
                pval = float(pvalues['is_human'])
            if se is None and coef is not None:
                # cannot compute z/p without se
                pass
            else:
                if pval is None and se is not None:
                    z = coef / se if se != 0 else float('nan')
                    pval = z_to_p(z)
                elif se is not None:
                    z = coef / se if se != 0 else float('nan')
        else:
            # Fall back to looking for precomputed entries in the dict
            raise AttributeError
    except Exception:
        # Fallback: check for precomputed odds ratio and CI in the dict
        try:
            odds_ratio = float(model_output.get('odds_ratio_is_human', np.nan))
            ci = model_output.get('ci_is_human', (np.nan, np.nan))
            ci_low, ci_high = float(ci[0]), float(ci[1])
            # If we have OR and CI but not coef/se/pval, we still can form a message
        except Exception:
            pass

    # If we successfully got coef and se, compute odds ratio and CI if not already set
    if coef is not None:
        try:
            odds_ratio = float(np.exp(coef))
            if se is not None:
                ci_low = float(np.exp(coef - 1.96 * se))
                ci_high = float(np.exp(coef + 1.96 * se))
        except Exception:
            pass

    # If still missing odds_ratio but model_output provided one, use it
    if odds_ratio is None and isinstance(model_output, dict):
        odds_ratio = model_output.get('odds_ratio_is_human', None)
        ci_tuple = model_output.get('ci_is_human', None)
        if ci_tuple is not None and (ci_low is None or ci_high is None):
            try:
                ci_low, ci_high = float(ci_tuple[0]), float(ci_tuple[1])
            except Exception:
                pass

    # Build the object to return
    obj = {
        "coef_is_human": coef,
        "se_is_human": se,
        "z_is_human": z,
        "p_value_is_human": pval,
        "odds_ratio_is_human": odds_ratio,
        "ci_95_is_human": (ci_low, ci_high)
    }

    # Formulate a concise conclusion: test for OR>1 and statistical significance
    conclusion = "Inconclusive"
    if odds_ratio is not None and pval is not None:
        if (odds_ratio > 1.0) and (pval < 0.05):
            conclusion = (
                "Yes — modern humans (Homo sapiens) have a statistically significantly "
                "higher frequency of AMTL compared to the non-human primates in the sample, "
                "after controlling for age, sex, and tooth class."
            )
        elif (odds_ratio <= 1.0) and (pval < 0.05):
            conclusion = (
                "No — modern humans have a statistically significantly lower frequency of AMTL "
                "after controls."
            )
        else:
            conclusion = (
                "No statistically significant difference in AMTL frequency between modern humans "
                "and the non-human primates after controlling for covariates (p >= 0.05)."
            )
    else:
        # If we lack a p-value but have OR and CI, use CI to infer significance (CI excluding 1)
        if odds_ratio is not None and ci_low is not None and ci_high is not None:
            if ci_low > 1.0:
                conclusion = (
                    "Yes — odds ratio CI excludes 1 and indicates higher AMTL in modern humans "
                    "after controls."
                )
            elif ci_high < 1.0:
                conclusion = (
                    "No — odds ratio CI excludes 1 in the direction of lower AMTL in modern humans."
                )
            else:
                conclusion = "Inconclusive — CI includes 1."

    # Prepare description summarizing the extracted stats and conclusion
    desc_parts = []
    if odds_ratio is not None:
        desc_parts.append(f"Odds ratio (is_human) = {odds_ratio:.3f}")
    if ci_low is not None and ci_high is not None:
        desc_parts.append(f"95% CI = ({ci_low:.3f}, {ci_high:.3f})")
    if pval is not None:
        desc_parts.append(f"p-value = {pval:.3g}")
    if coef is not None and se is not None:
        desc_parts.append(f"log-OR coef = {coef:.3f} (SE = {se:.3f})")

    description = "; ".join(desc_parts)
    if description:
        description = description + ". " + conclusion
    else:
        description = conclusion

    result["object"] = obj
    result["description"] = description

    return result