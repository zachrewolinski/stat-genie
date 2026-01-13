def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, z-scores, p-values, odds ratios (OR) and 95% CIs
    for the effect of rel_size_ratio on the probability that the focal group wins,
    summarized separately for each contest location (FocalNear, OtherNear, Neutral).
    
    Returns:
      {
        "object": {
          "reference_level": <which contest_loc_* dummy was omitted (reference)>,
          "per_location": {
            "FocalNear": {
              "slope_logit": ...,
              "se": ...,
              "z": ...,
              "p": ...,
              "OR": ...,
              "OR_CI": [lower, upper]
            },
            "OtherNear": { ... },
            "Neutral": { ... }
          },
          "raw_params": { <model params as dict> }
        },
        "description": "..."
      }
    """
    import math

    res = model_output  # GLMResultsWrapper or LogitResults-like object

    # Extract parameter vector and covariance matrix
    params = res.params  # pandas Series
    cov = res.cov_params()  # pandas DataFrame

    # Names used when creating dummies in the model code
    loc_levels = ['FocalNear', 'OtherNear', 'Neutral']
    dummy_names = [f'contest_loc_{lev}' for lev in loc_levels]
    interaction_names = [f'rel_size_ratio:{dn}' for dn in dummy_names]

    # Determine which location dummy was dropped (reference) by seeing which dummy is missing
    present_dummies = [dn for dn in dummy_names if dn in params.index]
    missing = [dn for dn in dummy_names if dn not in params.index]
    reference_level = None
    if len(missing) == 1:
        # missing dummy indicates reference
        reference_level = missing[0].replace('contest_loc_', '')
    else:
        # fallback: if none or multiple missing, mark unknown
        reference_level = 'unknown_or_all_present'

    # Base coefficient for rel_size_ratio
    if 'rel_size_ratio' not in params.index:
        raise KeyError("Model does not contain 'rel_size_ratio' parameter.")
    beta_base = float(params['rel_size_ratio'])
    var_base = float(cov.loc['rel_size_ratio', 'rel_size_ratio']) if 'rel_size_ratio' in cov.index else None

    # Helper: normal cdf using math.erf to avoid scipy dependency
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    results_by_location = {}
    for lev in loc_levels:
        dn = f'contest_loc_{lev}'
        inter = f'rel_size_ratio:{dn}'

        # If interaction term exists in model, combine base + interaction
        if inter in params.index:
            beta_inter = float(params[inter])
            # variance of sum = var(base) + var(inter) + 2*cov(base, inter)
            var_inter = float(cov.loc[inter, inter]) if inter in cov.index else None
            cov_bi = float(cov.loc['rel_size_ratio', inter]) if (('rel_size_ratio' in cov.index) and (inter in cov.index)) else 0.0

            slope = beta_base + beta_inter
            if (var_base is None) or (var_inter is None):
                se = None
            else:
                se = math.sqrt(max(0.0, var_base + var_inter + 2.0 * cov_bi))
        else:
            # No interaction term -> effect is just the base coefficient
            slope = beta_base
            se = math.sqrt(max(0.0, var_base)) if var_base is not None else None

        # z, p-value (two-sided) and odds ratio + CI
        if (se is None) or (se == 0):
            z = None
            p = None
            ci_lower = None
            ci_upper = None
        else:
            z = slope / se
            p = 2.0 * (1.0 - norm_cdf(abs(z)))
            # 95% CI on log-odds scale
            z_crit = 1.959963984540054  # approximate 97.5% quantile
            ci_lower = slope - z_crit * se
            ci_upper = slope + z_crit * se

        # Odds ratio and CI (exponentiated)
        OR = math.exp(slope) if slope is not None else None
        OR_ci = [math.exp(ci_lower) if ci_lower is not None else None,
                 math.exp(ci_upper) if ci_upper is not None else None]

        results_by_location[lev] = {
            "slope_logit": slope,
            "se": se,
            "z": z,
            "p": p,
            "OR": OR,
            "OR_CI": OR_ci
        }

    # Package raw params for reference (convert to plain dict)
    try:
        raw_params = {k: float(v) for k, v in params.items()}
    except Exception:
        raw_params = params.to_dict()

    output = {
        "object": {
            "reference_level": reference_level,
            "per_location": results_by_location,
            "raw_params": raw_params
        },
        "description": (
            "For each contest location (FocalNear, OtherNear, Neutral) the function reports:\n"
            "- slope_logit: the effect (coefficient) of rel_size_ratio on the log-odds that the focal group wins\n"
            "- se: standard error of that linear combination (base rel_size_ratio +/- interaction if present)\n"
            "- z and p: z-statistic and two-sided p-value testing slope != 0\n"
            "- OR: odds ratio = exp(slope) (multiplicative change in odds of focal win per unit increase in rel_size_ratio)\n"
            "- OR_CI: 95% CI for the odds ratio (exponentiated 95% CI on the log-odds scale)\n\n"
            "Interpretation: a slope_logit > 0 (OR > 1) means that increasing rel_size_ratio "
            "(i.e., the focal group being larger relative to the other group) increases the probability "
            "that the focal group wins. The p-value indicates whether that effect is statistically significant. "
            "Because the model included interactions rel_size_ratio:contest_loc_*, the reported slope for each "
            "location reflects the combined base effect plus the location-specific interaction term (if present)."
        )
    }

    return output