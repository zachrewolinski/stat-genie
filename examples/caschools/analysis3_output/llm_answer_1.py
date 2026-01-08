def extract_final_answer(model_output):
    """
    Extracts the estimate, uncertainty, sample size and an interpretation for the
    StudentTeacherRatio coefficient from a fitted model output.

    Returns a dict with keys:
      - "object": dict or None. If dict, contains keys:
            "coef", "p_value", "ci_lower", "ci_upper", "nobs",
            "significant_at_0.05", "supports_lower_ratio_higher_perf"
      - "description": human-readable interpretation of the extracted results
    """
    import numpy as np

    try:
        # Helpers to safely access attributes / dict-like items
        def has_key_like(container, key):
            try:
                if container is None:
                    return False
                if hasattr(container, 'index'):
                    return key in container.index
                if isinstance(container, dict):
                    return key in container
                return False
            except Exception:
                return False

        key = 'StudentTeacherRatio'

        # Attempt to read params, pvalues, conf_int, nobs
        params = getattr(model_output, 'params', None)
        pvalues = getattr(model_output, 'pvalues', None)
        nobs = getattr(model_output, 'nobs', None)

        # Try to extract coefficient
        coef = None
        if has_key_like(params, key):
            coef = params[key]
            try:
                coef = float(coef)
            except Exception:
                coef = None

        # Try to extract p-value
        pval = None
        if has_key_like(pvalues, key):
            try:
                pval = float(pvalues[key])
            except Exception:
                pval = None

        # Try to extract confidence interval via conf_int() if available
        ci_lower = ci_upper = None
        if hasattr(model_output, 'conf_int'):
            try:
                conf = model_output.conf_int()
                # conf may be a DataFrame or ndarray; prefer indexing by row label
                if hasattr(conf, 'loc') and key in conf.index:
                    row = conf.loc[key]
                    ci_lower, ci_upper = float(row[0]), float(row[1])
                else:
                    # If conf is array-like with same order as params, try to locate index position
                    if hasattr(params, 'index') and key in params.index:
                        pos = list(params.index).index(key)
                        ci_lower, ci_upper = float(conf[pos, 0]), float(conf[pos, 1])
            except Exception:
                ci_lower = ci_upper = None

        # Normalize nobs if possible
        try:
            if nobs is not None:
                # statsmodels sometimes stores nobs as numpy scalar
                nobs = int(nobs)
        except Exception:
            nobs = None

        # If coefficient missing or NaN or no observations, return informative message
        if coef is None or (isinstance(coef, float) and np.isnan(coef)) or nobs == 0:
            description = (
                "The model output does not contain a valid estimated coefficient for "
                "'StudentTeacherRatio' (parameters are missing, NaN, or there are zero observations). "
                "Cannot determine whether a lower student–teacher ratio is associated with higher academic performance."
            )
            return {"object": None, "description": description}

        # Determine statistical significance if p-value present
        significant = None
        if pval is not None and not np.isnan(pval):
            significant = (pval < 0.05)

        # Interpretation: recall StudentTeacherRatio is larger when there are more students per teacher.
        # A negative coefficient => higher ratio (more students per teacher) lowers scores => therefore lower ratio (fewer students per teacher) is associated with higher scores.
        supports_hypothesis = None
        try:
            supports_hypothesis = (float(coef) < 0)
        except Exception:
            supports_hypothesis = None

        # Build object to return
        obj = {
            "coef": float(coef),
            "p_value": float(pval) if pval is not None else None,
            "ci_lower": float(ci_lower) if ci_lower is not None else None,
            "ci_upper": float(ci_upper) if ci_upper is not None else None,
            "nobs": nobs,
            "significant_at_0.05": bool(significant) if significant is not None else None,
            "supports_lower_ratio_higher_perf": bool(supports_hypothesis) if supports_hypothesis is not None else None
        }

        # Build human-readable description
        desc_parts = []
        desc_parts.append(f"Estimated coefficient for StudentTeacherRatio = {obj['coef']:.4f}.")
        if obj['p_value'] is not None:
            desc_parts.append(f"p-value = {obj['p_value']:.3g}.")
        if obj['ci_lower'] is not None and obj['ci_upper'] is not None:
            desc_parts.append(f"95% CI = [{obj['ci_lower']:.4f}, {obj['ci_upper']:.4f}].")
        if obj['nobs'] is not None:
            desc_parts.append(f"n = {obj['nobs']}.")

        # Interpret direction and significance
        if obj['significant_at_0.05'] is True:
            if obj['supports_lower_ratio_higher_perf']:
                desc_parts.append(
                    "The (negative) effect is statistically significant at the 0.05 level, "
                    "which supports the hypothesis that a lower student–teacher ratio (fewer students per teacher) "
                    "is associated with higher average test scores, controlling for the included covariates."
                )
            else:
                desc_parts.append(
                    "The (positive) effect is statistically significant at the 0.05 level, "
                    "which indicates the opposite of the hypothesis: lower ratios would be associated with lower scores."
                )
        elif obj['significant_at_0.05'] is False:
            desc_parts.append(
                "The coefficient is not statistically significant at the 0.05 level; "
                "we cannot conclude there is a relationship between student–teacher ratio and test scores."
            )
        else:
            # p-value unavailable
            if obj['supports_lower_ratio_higher_perf']:
                desc_parts.append(
                    "The coefficient is negative (suggesting lower ratios may be associated with higher scores), "
                    "but a p-value was not available to assess statistical significance."
                )
            else:
                desc_parts.append(
                    "The coefficient is positive or zero (suggesting lower ratios may be associated with lower or no change in scores), "
                    "but a p-value was not available to assess statistical significance."
                )

        description = " ".join(desc_parts)
        return {"object": obj, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting results: {e}"
        }