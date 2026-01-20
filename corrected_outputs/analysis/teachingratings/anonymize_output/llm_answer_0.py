def extract_final_answer(model_output):
    """
    Extracts statistics for the predictor 'Beauty_z' from a fitted statsmodels
    MixedLMResults (or wrapper) object.

    Returns a dict with:
      - "object": dict with numeric results (coef, se, z, p, 95% CI)
      - "description": human-readable interpretation of the effect in context

    Notes:
      - Coefficient is on the TeachingEval scale (1-5) per 1 standard-deviation
        increase in Beauty (because Beauty_z is standardized).
      - The model includes a random intercept for InstructorID (so this effect
        is the within/between-instructor modeled association controlling for
        the specified covariates).
    """
    import math

    # Basic accessors
    params = getattr(model_output, "params", None)
    bse = getattr(model_output, "bse", None)
    pvalues = getattr(model_output, "pvalues", None)

    if params is None:
        raise ValueError("model_output has no 'params' attribute; not a fitted statsmodels result?")
    if 'Beauty_z' not in params.index:
        raise ValueError("Parameter 'Beauty_z' not found in model output parameters.")

    coef = float(params['Beauty_z'])

    # Standard error
    se = None
    if bse is not None and 'Beauty_z' in bse.index:
        se = float(bse['Beauty_z'])

    # z-statistic and p-value (compute p from z if p-value not provided)
    z = None
    p = None
    if se is not None:
        z = coef / se
        if pvalues is not None and 'Beauty_z' in pvalues.index:
            p = float(pvalues['Beauty_z'])
        else:
            # two-sided p-value from standard normal
            p = float(math.erfc(abs(z) / math.sqrt(2)))  # erfc gives 2*(1 - Phi(|z|))

    # 95% confidence interval
    ci_lower = ci_upper = None
    try:
        conf = model_output.conf_int()
        # conf_int returns a DataFrame indexed by parameter names
        if 'Beauty_z' in conf.index:
            ci_lower = float(conf.loc['Beauty_z', 0])
            ci_upper = float(conf.loc['Beauty_z', 1])
    except Exception:
        # If conf_int unavailable or different shape, ignore CI
        ci_lower = ci_upper = None

    result_object = {
        'coef': coef,
        'se': se,
        'z': z,
        'p': p,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }

    # Human-readable description / interpretation
    # Describe meaning of coefficient given standardized predictor and teaching eval scale (1-5).
    descr_parts = []
    descr_parts.append(
        f"Estimated effect of Beauty_z on TeachingEval: coef = {coef:.4f}"
    )
    if se is not None:
        descr_parts.append(f"(SE = {se:.4f}, z = {z:.3f}, p = {p:.4f})" if p is not None else f"(SE = {se:.4f})")
    if ci_lower is not None and ci_upper is not None:
        descr_parts.append(f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    descr = " ".join(descr_parts)

    # Interpretation sentence
    interp = (
        f"Interpretation: Because Beauty_z is standardized, the coefficient {coef:.4f} "
        f"represents the change in the course teaching evaluation (on the 1–5 scale) "
        f"associated with a one standard-deviation increase in judged instructor attractiveness, "
        f"holding the included covariates constant and accounting for instructor-level random intercepts."
    )
    if p is not None:
        if p < 0.05:
            interp += f" The effect is statistically significant at conventional levels (p = {p:.4f})."
        else:
            interp += f" The effect is not statistically significant at conventional levels (p = {p:.4f})."

    description = descr + " " + interp

    return {"object": result_object, "description": description}