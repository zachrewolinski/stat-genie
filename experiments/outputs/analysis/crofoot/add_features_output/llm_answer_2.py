def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, odds ratios, and
    tests the marginal effect of relative group size (size_diff) when focal_home=0
    and when focal_home=1 (i.e., accounting for the interaction).
    
    Returns a dictionary:
      - "object": dict with numeric results for:
          * size_diff (coef, se, p, 95% CI, odds_ratio, odds_ratio_CI)
          * interaction (coef for size_diff:focal_home, se, p, 95% CI)
          * marginal_size_when_focal_home_1 (coef, se, p, 95% CI, odds_ratio, odds_ratio_CI)
      - "description": text explaining what these numbers mean and how to interpret them.
    """
    import math
    import numpy as np
    import pandas as pd

    res = model_output  # statsmodels GLMResultsWrapper expected

    # Get parameter names
    param_index = list(res.params.index)

    # Helper to find parameter name containing substrings
    def find_param(contains_all, exclude_colon=False):
        for name in param_index:
            if all(sub in name for sub in contains_all):
                if exclude_colon and ':' in name:
                    continue
                return name
        return None

    # Identify parameter names (robust to ordering like "size_diff:focal_home" or "focal_home:size_diff")
    name_size = find_param(['size_diff'], exclude_colon=True)
    name_focal = find_param(['focal_home'], exclude_colon=True)
    name_inter = find_param(['size_diff', 'focal_home'])  # interaction (may contain ':') 

    # Prepare containers
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    try:
        ci = res.conf_int()
        # conf_int returns DataFrame-like with same index as params
        ci = pd.DataFrame(ci, index=params.index, columns=['ci_lower', 'ci_upper'])
    except Exception:
        # fallback: use params +/- 1.96*se
        ci = pd.DataFrame(index=params.index, columns=['ci_lower', 'ci_upper'])
        for n in params.index:
            se = bse.get(n, np.nan)
            coef = params.get(n, np.nan)
            ci.loc[n, 'ci_lower'] = coef - 1.96 * se
            ci.loc[n, 'ci_upper'] = coef + 1.96 * se

    cov = res.cov_params()

    def safe_get(name):
        if name is None:
            return None
        return {
            'name': name,
            'coef': float(params.get(name, np.nan)),
            'se': float(bse.get(name, np.nan)),
            'p': float(pvals.get(name, np.nan)),
            'ci_lower': float(ci.loc[name, 'ci_lower']),
            'ci_upper': float(ci.loc[name, 'ci_upper'])
        }

    out = {}
    out['size_diff'] = safe_get(name_size)
    out['interaction_size_diff_x_focal_home'] = safe_get(name_inter)
    out['focal_home_term'] = safe_get(name_focal)

    # Function to compute p-value for a linear combination (here used for marginal effect)
    def linear_combination_test(names, coeffs):
        """
        names: list of parameter names
        coeffs: list of multipliers to apply to each parameter (same length)
        Returns dict with coef, se, z, p, ci_lower, ci_upper
        """
        # Build coefficient value
        coef = 0.0
        for n, c in zip(names, coeffs):
            if n is None:
                # Parameter missing -> treat as zero contribution
                continue
            coef += float(params.get(n, 0.0)) * float(c)
        # Variance
        var = 0.0
        for i, (ni, ci_i) in enumerate(zip(names, coeffs)):
            if ni is None:
                continue
            for j, (nj, ci_j) in enumerate(zip(names, coeffs)):
                if nj is None:
                    continue
                # cov_params may return ndarray or DataFrame
                try:
                    cov_ij = float(cov.loc[ni, nj])
                except Exception:
                    try:
                        cov_ij = float(cov[ni][nj])
                    except Exception:
                        cov_ij = 0.0
                var += ci_i * ci_j * cov_ij
        se = math.sqrt(var) if var >= 0 else float('nan')
        z = coef / se if (se and not math.isnan(se)) else float('nan')
        # p-value from z (two-sided) using math.erf to avoid external deps
        if not (math.isnan(z) or math.isinf(z)):
            cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            p = 2.0 * (1.0 - cdf) if z >= 0 else 2.0 * cdf
        else:
            p = float('nan')
        ci_lower = coef - 1.96 * se if not math.isnan(se) else float('nan')
        ci_upper = coef + 1.96 * se if not math.isnan(se) else float('nan')
        return {
            'coef': float(coef),
            'se': float(se),
            'z': float(z),
            'p': float(p),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        }

    # Marginal effect of size_diff when focal_home == 0 is simply the coefficient on size_diff
    if name_size is not None:
        sz = out['size_diff']
        # compute odds ratio and CI
        try:
            or_val = math.exp(sz['coef'])
            or_ci_lower = math.exp(sz['ci_lower'])
            or_ci_upper = math.exp(sz['ci_upper'])
        except Exception:
            or_val = or_ci_lower = or_ci_upper = None
        out['size_diff']['odds_ratio'] = or_val
        out['size_diff']['odds_ratio_ci_lower'] = or_ci_lower
        out['size_diff']['odds_ratio_ci_upper'] = or_ci_upper
    else:
        out['size_diff'] = None

    # Marginal effect of size_diff when focal_home == 1:
    # beta_size + beta_interaction
    if name_size is not None:
        names_for_marginal = [name_size]
        coeffs_for_marginal = [1.0]
        if name_inter is not None:
            names_for_marginal.append(name_inter)
            coeffs_for_marginal.append(1.0)
        marginal = linear_combination_test(names_for_marginal, coeffs_for_marginal)
        # odds ratio for marginal
        try:
            marginal_or = math.exp(marginal['coef'])
            marginal_or_ci_lower = math.exp(marginal['ci_lower'])
            marginal_or_ci_upper = math.exp(marginal['ci_upper'])
        except Exception:
            marginal_or = marginal_or_ci_lower = marginal_or_ci_upper = None
        out['marginal_size_diff_when_focal_home_1'] = {
            'coef': marginal['coef'],
            'se': marginal['se'],
            'z': marginal['z'],
            'p': marginal['p'],
            'ci_lower': marginal['ci_lower'],
            'ci_upper': marginal['ci_upper'],
            'odds_ratio': marginal_or,
            'odds_ratio_ci_lower': marginal_or_ci_lower,
            'odds_ratio_ci_upper': marginal_or_ci_upper,
            'components_used': names_for_marginal
        }
    else:
        out['marginal_size_diff_when_focal_home_1'] = None

    # Also include the raw interaction term details (already in out['interaction_size_diff_x_focal_home'])
    # Build description explaining how to interpret these extracted stats
    desc_lines = []
    desc_lines.append("Extracted statistics for the effect of relative group size (size_diff) and its interaction with home location (focal_home).")
    desc_lines.append("Fields returned in 'object':")
    desc_lines.append(" - size_diff: coefficient (log-odds change per one-unit increase in focal - other group size) when focal_home == 0 (i.e., contest not nearer focal group's center). Includes se, p, 95% CI, and odds-ratio with CI.")
    desc_lines.append(" - interaction_size_diff_x_focal_home: coefficient for the interaction term (how the slope of size_diff changes when focal_home == 1).")
    desc_lines.append(" - marginal_size_diff_when_focal_home_1: combined effect (size_diff + interaction) giving the effect of size_diff when the contest is closer to the focal group's center. Includes se, z, p, 95% CI, and odds-ratio with CI.")
    desc_lines.append("")
    desc_lines.append("How to interpret the numbers:")
    desc_lines.append(" - A positive coefficient means that increasing relative group size increases the log-odds of the focal group winning; exponentiating gives the multiplicative change in odds per extra individual.")
    desc_lines.append(" - The interaction coefficient indicates whether the effect of size_diff is different when focal_home == 1. If the interaction is positive and statistically significant, the size advantage is larger when the contest is near the focal group's home.")
    desc_lines.append(" - Use the p-values (two-sided) to assess significance (commonly p < 0.05). The 'marginal_size_diff_when_focal_home_1' provides a formal test (Wald test) for the effect of size_diff at focal_home == 1.")
    desc_lines.append("")
    desc_lines.append("Note: If any parameter name could not be found in the model (e.g., because of different naming), that entry will be None. The function attempts to be robust to interaction naming like 'size_diff:focal_home' or 'focal_home:size_diff'.")

    description = "\n".join(desc_lines)

    return {"object": out, "description": description}