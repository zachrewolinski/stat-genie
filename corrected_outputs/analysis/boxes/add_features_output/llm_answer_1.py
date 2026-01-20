def extract_final_answer(model_output):
    """
    Extracts culture-specific linear and quadratic age effects (coefficients, SEs, p-values, 95% CIs)
    for 'reliance on majority' from a fitted statsmodels GLMResults (or robustified results).
    Returns a dictionary with:
      - "object": dict of per-culture age effect summaries (including reference/baseline culture)
      - "description": short explanation of the returned numbers and how to interpret them
    """
    import re
    import numpy as np
    import math
    try:
        from scipy.stats import norm
    except Exception:
        # If scipy not available, use a normal approximation via math (less convenient)
        norm = None

    res = model_output  # GLMResultsWrapper or robust results wrapper

    # Safely get parameter info
    try:
        params = res.params.copy()
    except Exception:
        raise ValueError("Could not read params from model_output")

    try:
        bse = res.bse.copy()
    except Exception:
        bse = None

    try:
        pvals = res.pvalues.copy()
    except Exception:
        pvals = None

    try:
        ci_df = res.conf_int()
        # conf_int returns ndarray or DataFrame; convert to DataFrame with index
        if not hasattr(ci_df, 'index'):
            ci_df = np.asarray(ci_df)
            ci_df = None  # we'll compute CIs later if needed
    except Exception:
        ci_df = None

    # Get covariance matrix if available for linear combination variances
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    # Identify main age terms
    if 'age_c' not in params.index:
        raise ValueError("Model does not contain 'age_c' main effect; cannot extract age effects.")
    if 'age_c2' not in params.index:
        raise ValueError("Model does not contain 'age_c2' main effect; cannot extract quadratic age effects.")

    main_age_name = 'age_c'
    main_age2_name = 'age_c2'
    main_age_coef = float(params[main_age_name])
    main_age2_coef = float(params[main_age2_name])

    # Parse culture interaction terms from parameter names
    param_names = list(params.index)
    culture_interactions = {}  # dict: culture -> {'age_c': paramname or None, 'age_c2': paramname or None}
    pattern = re.compile(r"C\(culture\)\[T\.([^\]]+)\]")
    for pname in param_names:
        if 'C(culture)[T.' in pname:
            m = pattern.search(pname)
            if not m:
                continue
            culture = m.group(1)
            if culture not in culture_interactions:
                culture_interactions[culture] = {'age_c': None, 'age_c2': None, 'main_param': None}
            # decide which term this is
            if pname.endswith(':age_c') or pname.startswith('age_c:') or ':age_c:' in pname or pname.endswith('age_c'):
                # avoid matching age_c2
                if 'age_c2' not in pname:
                    culture_interactions[culture]['age_c'] = pname
            if pname.endswith(':age_c2') or pname.startswith('age_c2:') or ':age_c2:' in pname or 'age_c2' in pname:
                culture_interactions[culture]['age_c2'] = pname
            # store main culture dummy param if present (not needed for age effects but helpful)
            if re.fullmatch(r"C\(culture\)\[T\." + re.escape(culture) + r"\]", pname):
                culture_interactions[culture]['main_param'] = pname

    # Build results for each culture, including reference culture (baseline)
    results_by_culture = {}

    # Reference/baseline culture: any culture not appearing in C(culture)[T.*] terms.
    # We can attempt to infer the baseline from the original data frame if available; otherwise label as "reference".
    baseline_label = "reference (baseline)"
    try:
        df = res.model.data.frame
        if 'culture' in df.columns:
            unique_cultures = list(df['culture'].astype(str).unique())
            # find baseline as the one not appearing in the T.* list if possible
            interacted = set(culture_interactions.keys())
            baseline_candidates = [c for c in unique_cultures if c not in interacted]
            if len(baseline_candidates) == 1:
                baseline_label = baseline_candidates[0]
            else:
                # default to first level in the data
                baseline_label = unique_cultures[0]
    except Exception:
        # keep default "reference (baseline)"
        pass

    # Helper to compute combined coef, se, p, ci for sum of main + interaction
    def combine(main_name, inter_name):
        """
        main_name: param name for main term (e.g., 'age_c')
        inter_name: param name for interaction term (or None)
        returns dict with coef, se, z, p, ci_lower, ci_upper
        """
        coef = float(params[main_name])
        if inter_name is not None and inter_name in params.index:
            coef += float(params[inter_name])

        # compute se using covariance matrix if available
        se = np.nan
        if cov is not None:
            try:
                # prefer DataFrame-style access if available
                if hasattr(cov, 'loc') and hasattr(cov, 'index'):
                    if inter_name is not None and inter_name in cov.index and main_name in cov.index:
                        var = float(cov.loc[main_name, main_name])
                        var += float(cov.loc[inter_name, inter_name])
                        var += 2.0 * float(cov.loc[main_name, inter_name])
                        se = math.sqrt(max(var, 0.0))
                    elif main_name in cov.index:
                        var = float(cov.loc[main_name, main_name])
                        se = math.sqrt(max(var, 0.0))
                else:
                    # try numpy ndarray with mapping by params.index order
                    if isinstance(cov, np.ndarray):
                        idx = list(params.index)
                        if main_name in idx:
                            i = idx.index(main_name)
                            if inter_name is not None and inter_name in idx:
                                j = idx.index(inter_name)
                                var = float(cov[i, i]) + float(cov[j, j]) + 2.0 * float(cov[i, j])
                                se = math.sqrt(max(var, 0.0))
                            else:
                                var = float(cov[i, i])
                                se = math.sqrt(max(var, 0.0))
            except Exception:
                se = np.nan

        if (not np.isfinite(se)) or np.isnan(se):
            # fallback: combine standard errors (ignoring covariance)
            try:
                se_main = float(bse[main_name]) if bse is not None and main_name in bse.index else np.nan
                se_inter = float(bse[inter_name]) if (inter_name is not None and bse is not None and inter_name in bse.index) else 0.0
                if np.isfinite(se_main):
                    se = math.sqrt(se_main**2 + se_inter**2)
                else:
                    se = np.nan
            except Exception:
                se = np.nan

        # z and p
        if se is not None and not np.isnan(se) and se > 0:
            z = coef / se
            if norm is not None:
                p = 2 * norm.sf(abs(z))
            else:
                # approximate using math.erf (less precise)
                p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        else:
            z = np.nan
            p = np.nan
            ci_lower = np.nan
            ci_upper = np.nan

        return {
            'coef': coef,
            'se': se,
            'z': z,
            'p': p,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper
        }

    # First add baseline culture
    results_by_culture[baseline_label] = {
        'linear_age': combine(main_age_name, None),
        'quadratic_age': combine(main_age2_name, None),
        'note': "This is the model reference / baseline culture (no 'C(culture)[T.<name>]' interaction)."
    }

    # Now add each culture that had interactions
    for cult, terms in culture_interactions.items():
        inter_linear_name = terms.get('age_c')
        inter_quad_name = terms.get('age_c2')
        results_by_culture[cult] = {
            'linear_age': combine(main_age_name, inter_linear_name),
            'quadratic_age': combine(main_age2_name, inter_quad_name),
            'note': "Effects shown are (baseline main effect) + (culture-specific interaction) for this culture."
        }

    # Additionally, provide a compact table of the raw parameter estimates for the main age terms and their interactions
    raw_params = {}
    keys_of_interest = [main_age_name, main_age2_name]
    # include any culture interaction param names
    for cult, terms in culture_interactions.items():
        if terms.get('age_c'):
            keys_of_interest.append(terms['age_c'])
        if terms.get('age_c2'):
            keys_of_interest.append(terms['age_c2'])
    for k in keys_of_interest:
        if k in params.index:
            raw_params[k] = {
                'coef': float(params[k]),
                'se': float(bse[k]) if bse is not None and k in getattr(bse, 'index', []) else None,
                'p': float(pvals[k]) if pvals is not None and k in getattr(pvals, 'index', []) else None
            }

    description = (
        "Returned object contains per-culture estimated linear (age_c) and quadratic (age_c2) effects "
        "on the log-odds of choosing the majority option. For each culture, the reported coefficient is "
        "the sum of the model's baseline (main) age coefficient and that culture's interaction coefficient "
        "(if present). SEs and p-values are computed using the model's covariance matrix when available; "
        "otherwise standard-error combination ignores covariance (noted). Interpretation: a positive linear "
        "coefficient means reliance on majority increases with age (on the log-odds scale); a negative quadratic "
        "coefficient indicates a decelerating (concave down) relationship. Use p-values and 95% CIs to judge "
        "statistical support (p < .05 conventionally indicates a reliable effect)."
    )

    return {
        "object": {
            "per_culture_age_effects": results_by_culture,
            "raw_params": raw_params
        },
        "description": description
    }