def extract_final_answer(model_output):
    """
    Extracts age-related statistics from a fitted statsmodels logistic regression
    model (with cluster-robust covariance) that predicts MajorityChoice.

    Returns a dictionary with:
      - "object": a dict containing:
          * 'age2' stats (coefficient for the quadratic age term)
          * 'slopes_by_culture': list of dicts, one per culture level (including the
            reference/base culture), giving the instantaneous slope of the
            log-odds of choosing the majority option with respect to age
            evaluated at the centered age = 0 (i.e., at mean age_c).
            For each culture we return slope (log-odds per unit age), SE,
            z, two-sided p-value, and 95% CI.
      - "description": brief explanation of what these numbers mean.
    """
    import re
    import math

    res = model_output  # expected to be a statsmodels results object (robustified)
    params = res.params
    cov = res.cov_params()

    # Helper: normal CDF using math.erf
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # 1) Extract age^2 (quadratic) term stats if present
    age2_name = None
    for name in params.index:
        if name == 'age2' or name.endswith('.age2') or ('age2' in name and ':' not in name):
            age2_name = name
            break
    age2_stats = None
    if age2_name is not None:
        coef = float(params[age2_name])
        se = float(math.sqrt(float(cov.loc[age2_name, age2_name])))
        z = coef / se if se > 0 else float('nan')
        p = 2.0 * (1.0 - norm_cdf(abs(z))) if not math.isnan(z) else float('nan')
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        age2_stats = {
            'term': age2_name,
            'coef': coef,
            'se': se,
            'z': z,
            'p': p,
            'ci_95': [ci_lower, ci_upper],
            'interpretation': (
                "Age^2 coefficient: non-zero indicates curvature in the age-"
                "trajectory (positive => accelerating increase with age; "
                "negative => decelerating or inverted-U)."
            )
        }

    # 2) Find interaction terms between age_c and culture and base slope
    # Identify any parameter names that include both 'age_c' and 'C(culture)'
    interaction_param_names = [
        name for name in params.index
        if ('age_c' in name) and ('C(culture)' in name)
    ]

    # Extract culture levels from interaction names using regex
    culture_levels = []
    interaction_map = {}  # maps culture_level -> interaction parameter name
    for iname in interaction_param_names:
        m = re.search(r'C\(culture\)\[T\.?(.*?)\]', iname)
        if m:
            lvl = m.group(1)
        else:
            # fallback: attempt to extract between brackets
            m2 = re.search(r'\[T\.(.*?)\]', iname)
            lvl = m2.group(1) if m2 else iname
        culture_levels.append(lvl)
        interaction_map[lvl] = iname

    # Add a label for the reference (baseline) culture which has no explicit interaction param
    # We cannot know the baseline level's name from the model object directly, so denote as 'reference'.
    all_culture_labels = ['reference'] + sorted(culture_levels)

    slopes_by_culture = []
    base_age_name = None
    # find the exact parameter name for age_c (could be 'age_c' typically)
    for name in params.index:
        # choose the parameter that is just age_c (not part of interaction)
        if name == 'age_c':
            base_age_name = name
            break
    if base_age_name is None:
        # fallback: find a parameter that equals 'age_c' substring but not containing 'C(culture)'
        for name in params.index:
            if ('age_c' in name) and ('C(culture)' not in name):
                base_age_name = name
                break

    if base_age_name is None:
        raise ValueError("Could not find the 'age_c' main effect parameter in model params.")

    # For each culture, compute slope at age_c = 0 (mean-centered age).
    # At age_c = 0 the derivative of log-odds w.r.t age is: beta_age_c + beta_age_c:C(culture)
    for lvl in all_culture_labels:
        if lvl == 'reference':
            coef_age = float(params[base_age_name])
            # variance is var(age_c)
            var = float(cov.loc[base_age_name, base_age_name])
            se = math.sqrt(var) if var >= 0 else float('nan')
            inter_name = None
        else:
            inter_name = interaction_map.get(lvl)
            inter_coef = float(params[inter_name]) if inter_name in params.index else 0.0
            coef_age = float(params[base_age_name]) + inter_coef
            # variance of sum: var(a)+var(b)+2cov(a,b)
            var_a = float(cov.loc[base_age_name, base_age_name])
            var_b = float(cov.loc[inter_name, inter_name]) if inter_name in cov.index else 0.0
            cov_ab = float(cov.loc[base_age_name, inter_name]) if (inter_name in cov.index and base_age_name in cov.index) else 0.0
            var = var_a + var_b + 2.0 * cov_ab
            se = math.sqrt(var) if var >= 0 else float('nan')

        z = coef_age / se if se > 0 else float('nan')
        p = 2.0 * (1.0 - norm_cdf(abs(z))) if not math.isnan(z) else float('nan')
        ci_lower = coef_age - 1.96 * se
        ci_upper = coef_age + 1.96 * se

        slopes_by_culture.append({
            'culture': lvl,
            'slope_logodds_at_mean_age_c': coef_age,
            'se': se,
            'z': z,
            'p': p,
            'ci_95': [ci_lower, ci_upper],
            'note': (
                "This is the instantaneous change in log-odds of choosing the majority "
                "option per 1 unit increase in centered age (age_c) evaluated at age_c=0. "
                "Positive => higher reliance on majority with increasing age; negative => decrease."
            )
        })

    result_object = {
        'age2': age2_stats,
        'slopes_by_culture': slopes_by_culture,
        'model_params_snapshot': {k: float(v) for k, v in params.items() if k in [base_age_name] + ([age2_name] if age2_name else [])}
    }

    description_lines = [
        "Extracted statistics relevant to developmental change in reliance on majority preference:",
        "- For each culture we report the instantaneous slope of the log-odds (beta_age_c + beta_age_c:C(culture)) evaluated at centered age = 0 (mean age).",
        "- A positive slope means that, at the mean age, older children are more likely to choose the majority option (higher reliance); a negative slope means the opposite.",
        "- The age^2 term (if present) indicates curvature: a significant age^2 implies the age effect changes nonlinearly with age.",
        "- Reported p-values are two-sided tests for whether the slope (or age^2 coefficient) differs from zero.",
        "- Confidence intervals are 95% Wald-style intervals based on the reported (cluster-robust) covariance matrix."
    ]

    return {
        "object": result_object,
        "description": "\n".join(description_lines)
    }