def extract_final_answer(model_output):
    """
    Extracts statistics relevant to the effect of gender on mortgage acceptance
    from a fitted statsmodels binary outcome model (Logit or GLM with Binomial).

    Returns a dictionary with keys:
      - "object": a dictionary of numeric results (coefficients, SEs, z-stats,
                  p-values, 95% CIs, odds ratios and CI, and average marginal
                  effects computed from the model's exog).
      - "description": A short human-readable interpretation that highlights
                       whether being female affects approval overall and whether
                       that effect differs for Black applicants (based on the
                       female x black interaction).
    """
    import numpy as np
    import math

    # Try to import a normal cdf for p-value calculation; provide fallback if scipy not available
    try:
        from scipy import stats
        norm_cdf = lambda x: stats.norm.cdf(x)
    except Exception:
        norm_cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    res = model_output

    # Names and parameter vector
    try:
        exog_names = list(res.model.exog_names)
    except Exception:
        # If not available, fail gracefully
        raise ValueError("Model object does not expose model.exog_names; cannot proceed.")

    params = res.params  # pandas Series or ndarray
    # Ensure params accessible by name
    if not hasattr(params, "__getitem__") or (isinstance(params, (list, tuple, np.ndarray)) and not getattr(params, "index", None)):
        raise ValueError("Model params not in a name-indexed structure. Expected a pandas Series-like object.")

    # Covariance of parameters
    cov = res.cov_params()
    # Helper to fetch cov element by name (works whether cov is DataFrame or ndarray)
    def cov_element(a, b):
        try:
            return cov.loc[a, b]
        except Exception:
            # fallback to positional indexing
            i = exog_names.index(a)
            j = exog_names.index(b)
            return cov[i, j]

    # Helper to get param and se by name
    def get_param_and_se(name):
        if name not in exog_names:
            return None
        try:
            coef = float(params[name])
        except Exception:
            return None
        # try to get se from cov diag
        try:
            var = cov_element(name, name)
            se = float(math.sqrt(var))
        except Exception:
            se = float("nan")
        z = coef / se if se != 0 and not math.isnan(se) else float("nan")
        p_value = 2.0 * (1.0 - norm_cdf(abs(z))) if not math.isnan(z) else None
        ci_lower = coef - 1.96 * se if not math.isnan(se) else None
        ci_upper = coef + 1.96 * se if not math.isnan(se) else None
        return dict(coef=coef, se=se, z=z, p_value=p_value, ci=(ci_lower, ci_upper))

    # Names we expect
    female_name = 'female'
    interaction_name = 'female_black_interaction'
    black_name = 'black'

    female_stats = get_param_and_se(female_name)
    interaction_stats = get_param_and_se(interaction_name)
    black_stats = get_param_and_se(black_name)

    # Combined effect for female when black = 1: coef_female + coef_interaction
    combined_stats = None
    if (female_name in exog_names) and (interaction_name in exog_names):
        coef_f = float(params[female_name])
        coef_int = float(params[interaction_name])
        coef_sum = coef_f + coef_int
        # variance of sum = var(f) + var(int) + 2*cov(f,int)
        var_sum = cov_element(female_name, female_name) + cov_element(interaction_name, interaction_name) + 2.0 * cov_element(female_name, interaction_name)
        se_sum = float(math.sqrt(var_sum)) if var_sum is not None else float("nan")
        z_sum = coef_sum / se_sum if se_sum != 0 else float("nan")
        p_sum = 2.0 * (1.0 - norm_cdf(abs(z_sum))) if not math.isnan(z_sum) else None
        ci_lower = coef_sum - 1.96 * se_sum if not math.isnan(se_sum) else None
        ci_upper = coef_sum + 1.96 * se_sum if not math.isnan(se_sum) else None
        combined_stats = dict(coef=coef_sum, se=se_sum, z=z_sum, p_value=p_sum, ci=(ci_lower, ci_upper))
    elif (female_name in exog_names) and (female_stats is not None):
        # No interaction column present; effect for black=1 equals female coef if interaction missing
        combined_stats = female_stats.copy()

    # Odds ratios and CI helpers
    def or_from_coef(coef, ci):
        try:
            if coef is None:
                return None, (None, None)
            or_val = math.exp(coef)
            or_ci = (math.exp(ci[0]) if ci[0] is not None else None, math.exp(ci[1]) if ci[1] is not None else None)
            return or_val, or_ci
        except Exception:
            return None, (None, None)

    female_or, female_or_ci = (None, (None, None))
    combined_or, combined_or_ci = (None, (None, None))
    if female_stats and female_stats.get('coef') is not None:
        female_or, female_or_ci = or_from_coef(female_stats['coef'], female_stats['ci'])
    if combined_stats and combined_stats.get('coef') is not None:
        combined_or, combined_or_ci = or_from_coef(combined_stats['coef'], combined_stats['ci'])

    # Compute average marginal effect (ATE) of being female by using the model's exog matrix.
    # We'll compute:
    #  - ATE overall: mean( P(female=1) - P(female=0) ) across all observations (keeping other covariates same)
    #  - ATE among non-Black (black==0) and among Black (black==1)
    ate_overall = None
    ate_black0 = None
    ate_black1 = None
    try:
        exog = np.asarray(res.model.exog).astype(float)  # (n_obs, n_vars)
        # Find column indices
        idx_f = exog_names.index(female_name) if female_name in exog_names else None
        idx_int = exog_names.index(interaction_name) if interaction_name in exog_names else None
        idx_black = exog_names.index(black_name) if black_name in exog_names else None

        # Prepare modified exog matrices
        exog_f1 = exog.copy()
        exog_f0 = exog.copy()
        if idx_f is not None:
            exog_f1[:, idx_f] = 1.0
            exog_f0[:, idx_f] = 0.0
        if idx_int is not None and idx_black is not None:
            # interaction = female * black
            exog_f1[:, idx_int] = exog_f1[:, idx_black] * 1.0
            exog_f0[:, idx_int] = exog_f0[:, idx_black] * 0.0

        # Predicted probabilities
        pred_f1 = res.predict(exog_f1)
        pred_f0 = res.predict(exog_f0)
        diff = np.asarray(pred_f1) - np.asarray(pred_f0)
        ate_overall = float(np.nanmean(diff))

        if idx_black is not None:
            black_vals = exog[:, idx_black]
            mask0 = (black_vals == 0)
            mask1 = (black_vals == 1)
            if mask0.any():
                ate_black0 = float(np.nanmean(diff[mask0]))
            if mask1.any():
                ate_black1 = float(np.nanmean(diff[mask1]))
    except Exception:
        # If something fails (e.g., predict signature different), leave ATEs as None
        ate_overall = ate_black0 = ate_black1 = None

    # Build the object to return
    result_object = {
        'female': {
            'name': female_name,
            'coef_logodds': None if female_stats is None else female_stats.get('coef'),
            'se': None if female_stats is None else female_stats.get('se'),
            'z': None if female_stats is None else female_stats.get('z'),
            'p_value': None if female_stats is None else female_stats.get('p_value'),
            '95ci_logodds': None if female_stats is None else female_stats.get('ci'),
            'odds_ratio': female_or,
            'odds_ratio_95ci': female_or_ci
        },
        'female_when_black1': {
            'expression': f"{female_name} + {interaction_name}",
            'coef_logodds': None if combined_stats is None else combined_stats.get('coef'),
            'se': None if combined_stats is None else combined_stats.get('se'),
            'z': None if combined_stats is None else combined_stats.get('z'),
            'p_value': None if combined_stats is None else combined_stats.get('p_value'),
            '95ci_logodds': None if combined_stats is None else combined_stats.get('ci'),
            'odds_ratio': combined_or,
            'odds_ratio_95ci': combined_or_ci
        },
        'interaction': {
            'name': interaction_name,
            'coef_logodds': None if interaction_stats is None else interaction_stats.get('coef'),
            'se': None if interaction_stats is None else interaction_stats.get('se'),
            'z': None if interaction_stats is None else interaction_stats.get('z'),
            'p_value': None if interaction_stats is None else interaction_stats.get('p_value'),
            '95ci_logodds': None if interaction_stats is None else interaction_stats.get('ci')
        },
        'average_marginal_effects': {
            'ATE_overall_prob_diff': ate_overall,
            'ATE_black0_prob_diff': ate_black0,
            'ATE_black1_prob_diff': ate_black1
        },
        'notes': {
            'exog_names': exog_names
        }
    }

    # Helpers for safe formatting in description
    def sig_marker(p):
        if p is None:
            return "n/a"
        return ("p<0.01" if p < 0.01 else "p<0.05" if p < 0.05 else "n.s. (p>=0.05)")

    def fmt_num(x, decimals=3):
        try:
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                return "n/a"
            return f"{x:.{decimals}f}"
        except Exception:
            return "n/a"

    def fmt_ci(ci):
        if not ci or ci[0] is None or ci[1] is None:
            return "n/a"
        return f"{fmt_num(ci[0])} to {fmt_num(ci[1])}"

    desc_lines = []
    # Female main effect (for black=0)
    if female_stats and female_stats.get('coef') is not None:
        desc_lines.append(
            f"The estimated log-odds coefficient for being female (baseline, i.e. for non-Black if an interaction is present) "
            f"is {fmt_num(female_stats['coef'])} (SE={fmt_num(female_stats['se'])}, {sig_marker(female_stats.get('p_value'))}). "
            f"This corresponds to an odds ratio = {fmt_num(female_or) if female_or is not None else 'n/a'} (95% CI {fmt_ci(female_or_ci)})."
        )
    else:
        desc_lines.append("No coefficient found for 'female' in the model.")

    # Interaction / combined effect
    if combined_stats and combined_stats.get('coef') is not None:
        desc_lines.append(
            f"For Black applicants, the gender effect (female + female:black) has log-odds = {fmt_num(combined_stats['coef'])} "
            f"(SE={fmt_num(combined_stats['se'])}, {sig_marker(combined_stats.get('p_value'))}), odds ratio = {fmt_num(combined_or) if combined_or is not None else 'n/a'} "
            f"(95% CI {fmt_ci(combined_or_ci)})."
        )
    elif interaction_stats and interaction_stats.get('coef') is not None:
        desc_lines.append(
            f"An interaction term {interaction_name} is present with coef = {fmt_num(interaction_stats['coef'])} "
            f"(SE={fmt_num(interaction_stats['se'])}, {sig_marker(interaction_stats.get('p_value'))}), indicating the gender effect differs by race."
        )
    else:
        desc_lines.append("No female x black interaction found; the female coefficient applies to all applicants.")

    # Average marginal effects
    if ate_overall is not None:
        try:
            desc_lines.append(
                f"Model-based average marginal effect of being female (change in predicted approval probability): overall = {ate_overall:.3%}."
            )
            if ate_black0 is not None or ate_black1 is not None:
                desc_lines.append(
                    f"Among non-Black applicants: {ate_black0:.3% if ate_black0 is not None else 'n/a'}; among Black applicants: {ate_black1:.3% if ate_black1 is not None else 'n/a'}."
                )
        except Exception:
            desc_lines.append("Model-based average marginal effects computed but could not be formatted.")
    else:
        desc_lines.append("Could not compute average marginal effects from model exog/predict.")

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}