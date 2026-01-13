def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of Age (Age_centered) on MajorityChoice
    and how that effect varies by culture from a fitted statsmodels GLMResultsWrapper.

    Returns a dictionary with:
      - "object": a dict containing detailed extracted statistics:
           - age_main_effect: {beta, se, z, p, ci_lower, ci_upper, OR, OR_CI}
           - interactions: dict mapping culture -> {slope, se, z, p, ci_lower, ci_upper, OR, OR_CI}
           - interaction_terms: dict of raw interaction parameter p-values (if present)
      - "description": a plain-English interpretation of results (whether majority reliance
           increases with age overall and whether developmental trajectories differ across cultures).
    """
    import re
    import numpy as np
    from math import exp, sqrt
    try:
        # Access parameter estimates and covariance
        params = model_output.params  # pandas Series
        cov = model_output.cov_params()  # DataFrame
        pvalues = model_output.pvalues
        conf_int = model_output.conf_int()  # DataFrame with columns [0,1]
    except Exception as e:
        raise ValueError("model_output does not look like a fitted statsmodels results object.") from e

    # Helper to get var/cov safely
    def var_of(name):
        return float(cov.loc[name, name])

    def cov_of(name1, name2):
        # if either name not in cov, return 0
        if name1 not in cov.index or name2 not in cov.columns:
            return 0.0
        return float(cov.loc[name1, name2])

    # Identify culture levels from the original data frame if available
    cultures = None
    try:
        df = model_output.model.data.frame
        if 'culture' in df.columns:
            cultures = list(np.unique(df['culture']))
    except Exception:
        cultures = None

    # If we couldn't get cultures from the data, infer from parameter names
    # Look for parameter names like C(culture)[T.X]
    param_names = list(params.index)
    t_levels = []
    pattern = re.compile(r"C\(culture\)\[T\.(.*?)\]")
    for name in param_names:
        m = pattern.search(name)
        if m:
            t_levels.append(m.group(1))
    t_levels = list(dict.fromkeys(t_levels))  # unique, preserve order

    if cultures is None:
        # If we have T.* levels, we can create a provisional list:
        if t_levels:
            # We don't know the reference level's name, so include a placeholder if necessary
            # Attempt to find reference by looking for culture variable in model.data.orig_exog if present
            try:
                # Try to infer from model.data.orig_exog if present
                orig = model_output.model.data.orig_exog
            except Exception:
                orig = None
            if orig is not None and 'culture' in orig.columns:
                cultures = list(np.unique(orig['culture']))
            else:
                # Create a list: reference level unknown, call it 'REFERENCE' if needed
                # We'll compose cultures as [REFERENCE] + t_levels (since t_levels are the non-reference)
                cultures = ['REFERENCE'] + t_levels
        else:
            # As a fallback, assume single culture present (no C(culture) terms)
            cultures = ['ALL']

    # Find the main Age_centered parameter name
    age_name = None
    for name in param_names:
        if name == 'Age_centered':
            age_name = name
            break
    if age_name is None:
        # Try variants (some formula interfaces might name it differently)
        for name in param_names:
            if 'Age_centered' in name:
                age_name = name
                break
    if age_name is None:
        raise KeyError("Could not find 'Age_centered' parameter in model parameters.")

    # Extract main effect for Age
    beta_age = float(params[age_name])
    se_age = sqrt(var_of(age_name))
    z_age = beta_age / se_age if se_age > 0 else np.nan
    # two-sided p-value (we'll prefer reported pvalue if available)
    p_age = float(pvalues.get(age_name, 2 * (1 - 0.5 * (1 + np.math.erf(abs(z_age) / sqrt(2))))))  # fallback
    ci_lower_age, ci_upper_age = float(conf_int.loc[age_name, 0]), float(conf_int.loc[age_name, 1])
    OR_age = exp(beta_age)
    OR_CI_age = (exp(ci_lower_age), exp(ci_upper_age))

    age_main_effect = {
        'beta': beta_age,
        'se': se_age,
        'z': z_age,
        'p': p_age,
        'ci_lower': ci_lower_age,
        'ci_upper': ci_upper_age,
        'OR': OR_age,
        'OR_ci': OR_CI_age
    }

    # Prepare interactions: For each culture, compute the Age slope (beta_age + interaction if present)
    interactions = {}
    interaction_term_pvalues = {}
    for level in cultures:
        # Determine name of the interaction parameter for this culture, if it exists.
        # Interaction names may be 'Age_centered:C(culture)[T.level]' or 'C(culture)[T.level]:Age_centered'
        inter_name = None
        candidate_patterns = [
            f"Age_centered:C(culture)[T.{level}]",
            f"C(culture)[T.{level}]:Age_centered",
            f"Age_centered:C(culture)[T.{level}]",  # redundancy
        ]
        for cand in candidate_patterns:
            if cand in params.index:
                inter_name = cand
                break
        # Also handle cases where level strings include special characters by searching for level substring
        if inter_name is None:
            for name in params.index:
                if 'Age_centered' in name and f"T.{level}" in name:
                    inter_name = name
                    break

        if inter_name is None:
            # No interaction term for this level -> slope is same as main effect (reference or no interaction)
            slope = beta_age
            se = se_age
            z = slope / se if se > 0 else np.nan
            p = float(pvalues.get(age_name, np.nan))
            ci_lower = slope - 1.96 * se
            ci_upper = slope + 1.96 * se
        else:
            # slope = beta_age + beta_inter
            beta_inter = float(params[inter_name])
            slope = beta_age + beta_inter
            # variance = var(beta_age) + var(beta_inter) + 2*cov(beta_age, beta_inter)
            var_slope = var_of(age_name)
            # If inter_name not present in cov, var(...) will KeyError earlier; handled by var_of returning error
            try:
                var_inter = float(cov.loc[inter_name, inter_name])
            except Exception:
                var_inter = 0.0
            try:
                cov_ai = float(cov.loc[age_name, inter_name])
            except Exception:
                cov_ai = 0.0
            se = sqrt(max(var_slope + var_inter + 2.0 * cov_ai, 0.0))
            z = slope / se if se > 0 else np.nan
            # compute p using normal approximation
            from scipy.stats import norm
            p = 2.0 * (1.0 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            ci_lower = slope - 1.96 * se
            ci_upper = slope + 1.96 * se
            # also capture the raw interaction term p-value
            interaction_term_pvalues[level] = float(pvalues.get(inter_name, np.nan))

        OR = exp(slope)
        OR_ci = (exp(ci_lower), exp(ci_upper))

        interactions[level] = {
            'slope_log_odds_per_year': slope,
            'se': se,
            'z': z,
            'p': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'OR_per_year': OR,
            'OR_ci': OR_ci,
            'interaction_param_name': inter_name
        }

    # Decide whether there's evidence that developmental trajectories differ across cultures:
    # Look for any interaction parameter with p < .05
    any_significant_interaction = False
    significant_levels = []
    for level, pname in interaction_term_pvalues.items():
        if pname is not None and not np.isnan(pname) and pname < 0.05:
            any_significant_interaction = True
            significant_levels.append(level)

    # Summarize results in plain language
    desc_lines = []
    # Overall age effect interpretation
    age_dir = "increase" if age_main_effect['beta'] > 0 else ("decrease" if age_main_effect['beta'] < 0 else "no change")
    desc_lines.append(
        f"Overall (main effect): Age_centered beta = {age_main_effect['beta']:.3f}, "
        f"SE = {age_main_effect['se']:.3f}, z = {age_main_effect['z']:.2f}, p = {age_main_effect['p']:.3f}. "
        f"This corresponds to an odds ratio per year of {age_main_effect['OR']:.3f} "
        f"(95% CI [{age_main_effect['OR_ci'][0]:.3f}, {age_main_effect['OR_ci'][1]:.3f}]). "
        f"Direction: older children tend to {age_dir} the odds of choosing the majority."
    )

    # Culture-specific slopes
    for level, stats_dict in interactions.items():
        pval = stats_dict['p']
        sig = "significant" if (not np.isnan(pval) and pval < 0.05) else "not significant"
        dir_text = "increase" if stats_dict['slope_log_odds_per_year'] > 0 else ("decrease" if stats_dict['slope_log_odds_per_year'] < 0 else "no change")
        desc_lines.append(
            f"Culture {level}: slope (log-odds per year) = {stats_dict['slope_log_odds_per_year']:.3f}, "
            f"SE = {stats_dict['se']:.3f}, p = {pval:.3f} -> {sig}; direction: {dir_text}. "
            f"OR per year = {stats_dict['OR_per_year']:.3f} (95% CI [{stats_dict['OR_ci'][0]:.3f}, {stats_dict['OR_ci'][1]:.3f}])."
        )

    # Interaction overall conclusion
    if any_significant_interaction:
        desc_lines.append(
            "There is evidence that developmental trajectories differ across cultures: "
            f"significant Age x Culture interactions found for cultures: {', '.join(significant_levels)}."
        )
    else:
        desc_lines.append(
            "No strong evidence that developmental trajectories differ across cultures (no Age x Culture interaction terms with p < 0.05)."
        )

    description = " ".join(desc_lines)

    result_object = {
        'age_main_effect': age_main_effect,
        'age_by_culture': interactions,
        'interaction_term_pvalues': interaction_term_pvalues
    }

    return {
        "object": result_object,
        "description": description
    }