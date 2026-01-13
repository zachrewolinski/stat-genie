def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, z-stats, p-values and 95% CIs for:
      - Age_c (linear age effect in the reference culture)
      - Age_sq (quadratic age effect, global)
      - Per-culture linear age slopes (Age_c + interaction term when present)

    Returns:
      {
        "object": {
          "Age_c": {coef, se, z, p, ci_low, ci_high},
          "Age_sq": { ... },
          "per_culture_slopes": {
              culture_level: {coef, se, z, p, ci_low, ci_high}, ...
          }
        },
        "description": "<brief interpretation>"
      }
    """
    import numpy as np
    from types import SimpleNamespace
    try:
        # stats for normal-based p-values / CIs
        from scipy.stats import norm
    except Exception:
        # fallback: approximate normal cdf via numpy (very rare)
        def _norm_cdf(x):
            return 0.5 * (1 + np.erf(x / np.sqrt(2)))
        class _N:
            cdf = staticmethod(_norm_cdf)
        norm = _N()

    # Retrieve robust results or fitted model
    robust = model_output.get('robust_results', None)
    fitted = model_output.get('fitted_model', None)

    # Helper to get params (pd.Series), cov (DataFrame or ndarray), and bse if available
    params = None
    cov = None
    bse = None

    # Try robust first (preferred)
    if robust is not None:
        # robust may be a statsmodels Results object or a SimpleNamespace-like wrapper
        if hasattr(robust, 'params'):
            params = robust.params
        if hasattr(robust, 'cov_params'):
            try:
                cov = robust.cov_params()
            except Exception:
                # cov_params might be a callable property or raise; ignore here
                cov = None
        if hasattr(robust, 'bse'):
            bse = robust.bse

    # Fall back to fitted model
    if params is None and fitted is not None:
        try:
            params = fitted.params
        except Exception:
            params = None
    if cov is None and fitted is not None:
        try:
            cov = fitted.cov_params()
        except Exception:
            cov = None
    if bse is None and fitted is not None:
        try:
            bse = fitted.bse
        except Exception:
            bse = None

    if params is None:
        raise ValueError("Could not find parameter estimates in model_output.")

    # Ensure params is a pandas Series-like with index
    try:
        param_index = list(params.index)
    except Exception:
        # if params is ndarray, attempt to get names from fitted.model
        raise ValueError("Parameter names not available; cannot proceed.")

    # helper to get covariance between two parameter names
    def get_cov(name1, name2):
        nonlocal cov, params, param_index
        if cov is None:
            raise ValueError("Covariance matrix not available in model_output; cannot compute SE for combined terms.")
        # If cov is a pandas DataFrame, use .loc
        if hasattr(cov, 'loc'):
            try:
                return float(cov.loc[name1, name2])
            except Exception as e:
                raise KeyError(f"Could not find covariance entries for {name1}, {name2} in cov DataFrame: {e}")
        else:
            # assume numpy ndarray with ordering matching params.index
            try:
                i = param_index.index(name1)
                j = param_index.index(name2)
            except ValueError as e:
                raise KeyError(f"Parameter name not found in parameter index: {e}")
            return float(cov[i, j])

    # helper to get var (cov(name,name))
    def get_var(name):
        return get_cov(name, name)

    # compute stats for a single parameter name
    def single_param_stats(name):
        if name not in param_index:
            return None
        coef = float(params[name])
        # se
        if bse is not None:
            try:
                # bse might be array aligned with params
                if hasattr(bse, 'index'):
                    se = float(bse.loc[name])
                else:
                    se = float(bse[param_index.index(name)])
            except Exception:
                se = float(np.sqrt(get_var(name)))
        else:
            se = float(np.sqrt(get_var(name)))
        z = coef / se if se != 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z)))
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        return {"coef": coef, "se": se, "z": z, "p": p, "ci_low": ci_low, "ci_high": ci_high}

    results = {}
    # Age_c
    age_stats = single_param_stats('Age_c')
    if age_stats is None:
        raise KeyError("Parameter 'Age_c' not found in model parameters.")
    results['Age_c'] = age_stats

    # Age_sq
    age_sq_stats = single_param_stats('Age_sq')
    if age_sq_stats is None:
        raise KeyError("Parameter 'Age_sq' not found in model parameters.")
    results['Age_sq'] = age_sq_stats

    # Determine culture levels from predicted_by_age_and_culture (preferred) or from params naming
    preds = model_output.get('predicted_by_age_and_culture', None)
    culture_levels = None
    if preds:
        try:
            # keys may be ints or strings; keep original representation
            culture_levels = list(preds.keys())
        except Exception:
            culture_levels = None

    # Parse interaction parameter names to find explicit T.* levels
    interaction_prefix = 'Age_c:C(culture)[T.'
    interacted_levels = []
    for name in param_index:
        if name.startswith(interaction_prefix):
            # extract what's inside after prefix and before ]
            tail = name[len(interaction_prefix):]
            # tail like '2]' or " 'a']" etc. Remove trailing ]
            if tail.endswith(']'):
                level = tail[:-1]
            else:
                level = tail
            # Try convert to int if possible
            try:
                lvl = int(level)
            except Exception:
                lvl = level
            interacted_levels.append(lvl)

    # Determine all levels if possible: if preds present use it; else try to infer from C(culture)[T.X] and C(culture)[T.X] main effects
    if culture_levels is None:
        # look for C(culture)[T.*] main-effects names to collect levels
        main_prefix = 'C(culture)[T.'
        main_levels = []
        for name in param_index:
            if name.startswith(main_prefix):
                tail = name[len(main_prefix):]
                if tail.endswith(']'):
                    level = tail[:-1]
                else:
                    level = tail
                try:
                    lvl = int(level)
                except Exception:
                    lvl = level
                main_levels.append(lvl)
        # if we have main_levels plus interacted_levels, union them
        all_levels = list(dict.fromkeys(main_levels + interacted_levels))
        if all_levels:
            culture_levels = all_levels
        else:
            # as a last resort, set culture_levels to interacted_levels
            culture_levels = interacted_levels if interacted_levels else []

    # Determine reference culture: the culture level present in data but omitted from main-effect dummies.
    reference = None
    if culture_levels:
        # find which culture in culture_levels is NOT present as main-effect dummy C(culture)[T.X]
        main_prefix = 'C(culture)[T.'
        main_present = []
        for name in param_index:
            if name.startswith(main_prefix):
                tail = name[len(main_prefix):]
                if tail.endswith(']'):
                    level = tail[:-1]
                else:
                    level = tail
                try:
                    lvl = int(level)
                except Exception:
                    lvl = level
                main_present.append(lvl)
        # If main_present is empty, likely reference is the smallest/first level found in preds
        if main_present:
            # choose a culture from culture_levels that is not in main_present
            for lvl in culture_levels:
                if lvl not in main_present:
                    reference = lvl
                    break
            # if none found, pick the first culture_levels as reference
            if reference is None and len(culture_levels) > 0:
                reference = culture_levels[0]
        else:
            # use first culture_levels as reference
            reference = culture_levels[0]
    else:
        # No culture information found; cannot compute per-culture slopes
        reference = None

    # Compute per-culture slopes for Age_c
    per_culture = {}
    for lvl in culture_levels:
        # Build the expected interaction parameter name used in params
        # interaction name in the model output was: 'Age_c:C(culture)[T.<lvl>]'
        # But lvl might be int or string; ensure matching format
        if isinstance(lvl, int):
            inter_name = f'Age_c:C(culture)[T.{lvl}]'
            main_name = f'C(culture)[T.{lvl}]'
        else:
            inter_name = f'Age_c:C(culture)[T.{lvl}]'
            main_name = f'C(culture)[T.{lvl}]'

        if reference is not None and lvl == reference:
            # slope is simply Age_c
            per_coef = float(params['Age_c'])
            var = get_var('Age_c')
            se = float(np.sqrt(var))
            z = per_coef / se if se != 0 else np.nan
            p = 2 * (1 - norm.cdf(abs(z)))
            ci_low = per_coef - 1.96 * se
            ci_high = per_coef + 1.96 * se
            per_culture[lvl] = {"coef": per_coef, "se": se, "z": z, "p": p, "ci_low": ci_low, "ci_high": ci_high, "note": "reference"}
        else:
            # If interaction term present, slope = Age_c + Age_c:C(culture)[T.lvl]
            if inter_name in param_index:
                a = float(params['Age_c'])
                b = float(params[inter_name])
                coef = a + b
                # var = var(a) + var(b) + 2*cov(a,b)
                try:
                    var_a = get_var('Age_c')
                    var_b = get_var(inter_name)
                    cov_ab = get_cov('Age_c', inter_name)
                    var = var_a + var_b + 2.0 * cov_ab
                    se = float(np.sqrt(max(var, 0.0)))
                except Exception as e:
                    # fallback: try to use bse elements if available (less accurate for sum)
                    if bse is not None:
                        try:
                            if hasattr(bse, 'index'):
                                se_a = float(bse.loc['Age_c'])
                                se_b = float(bse.loc[inter_name])
                            else:
                                se_a = float(bse[param_index.index('Age_c')])
                                se_b = float(bse[param_index.index(inter_name)])
                            # approximate var by summing variances (ignoring covariance) - less accurate
                            se = float(np.sqrt(se_a**2 + se_b**2))
                            var = se**2
                        except Exception:
                            se = np.nan
                            var = np.nan
                    else:
                        se = np.nan
                        var = np.nan
                z = coef / se if (se and not np.isnan(se)) else np.nan
                p = 2 * (1 - norm.cdf(abs(z))) if (not np.isnan(z)) else np.nan
                ci_low = coef - 1.96 * se if (not np.isnan(se)) else np.nan
                ci_high = coef + 1.96 * se if (not np.isnan(se)) else np.nan
                per_culture[lvl] = {"coef": coef, "se": se, "z": z, "p": p, "ci_low": ci_low, "ci_high": ci_high}
            else:
                # No interaction term found for this level: interpret as same slope as reference
                # i.e., slope = Age_c
                per_coef = float(params['Age_c'])
                var = get_var('Age_c')
                se = float(np.sqrt(var))
                z = per_coef / se if se != 0 else np.nan
                p = 2 * (1 - norm.cdf(abs(z)))
                ci_low = per_coef - 1.96 * se
                ci_high = per_coef + 1.96 * se
                per_culture[lvl] = {"coef": per_coef, "se": se, "z": z, "p": p, "ci_low": ci_low, "ci_high": ci_high, "note": "no interaction term found; same as reference"}

    results['per_culture_slopes'] = per_culture
    results['reference_culture'] = reference

    # Prepare a short textual interpretation
    # Use age linear p-value and age_sq p-value and collect which cultures show significant positive or negative slopes
    interp_lines = []
    # Age linear
    p_age = results['Age_c']['p']
    coef_age = results['Age_c']['coef']
    interp_lines.append(f"Overall (reference culture) linear age effect (Age_c): coef={coef_age:.3f}, p={p_age:.3f}.")
    # Age quadratic
    p_age_sq = results['Age_sq']['p']
    coef_age_sq = results['Age_sq']['coef']
    interp_lines.append(f"Quadratic age effect (Age_sq): coef={coef_age_sq:.3f}, p={p_age_sq:.3f} (global, not interacted).")

    # Per-culture summary
    sig_pos = []
    sig_neg = []
    nonsig = []
    for lvl, stats in per_culture.items():
        p = stats.get('p', np.nan)
        coef = stats.get('coef', np.nan)
        if not np.isnan(p) and p < 0.05:
            if coef > 0:
                sig_pos.append(lvl)
            elif coef < 0:
                sig_neg.append(lvl)
            else:
                nonsig.append(lvl)
        else:
            nonsig.append(lvl)
    interp_lines.append(f"Significant positive age-related increase in reliance on majority observed in cultures: {sig_pos if sig_pos else 'none'}.")
    interp_lines.append(f"Significant negative age-related decrease in cultures: {sig_neg if sig_neg else 'none'}.")
    interp_lines.append("Note: Age_sq being significant indicates a nonlinear (accelerating/decelerating) change with age across all cultures; per-culture slopes above reflect linear marginal slopes (Age_c + interaction).")

    description = " ".join(interp_lines)

    return {"object": results, "description": description}