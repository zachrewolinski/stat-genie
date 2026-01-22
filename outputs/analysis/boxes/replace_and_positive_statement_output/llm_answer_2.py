def extract_final_answer(model_output):
    """
    Extracts per-culture age effects (log-odds slopes), standard errors, z-values, p-values,
    and 95% CIs for both SocialReliance and MajorityChoice clustered models.

    Returns:
      {
        "object": {
          "SocialReliance": { culture_level: {slope, se, z, p, ci_lower, ci_upper, signif}, ... },
          "MajorityChoice": { ... }
        },
        "description": <text summary of what these numbers mean>
      }
    """
    import math
    import numpy as np
    import pandas as pd

    def _p_from_z(z):
        # two-sided p from z using erfc (no scipy dependency)
        return math.erfc(abs(z) / math.sqrt(2))

    def _ensure_series(obj, index=None):
        """Ensure params/bse-like objects are pandas Series with given index when possible."""
        if isinstance(obj, pd.Series):
            return obj
        try:
            return pd.Series(obj, index=index) if index is not None else pd.Series(obj)
        except Exception:
            return pd.Series(obj)

    def summarize_model(clustered_res):
        # clustered_res: statsmodels results-like object with .params (Series), .bse (Series), and preferably .cov_params() or similar
        # Robustly obtain params
        if not hasattr(clustered_res, 'params'):
            raise KeyError("Model result does not have 'params'.")
        params = clustered_res.params
        params = _ensure_series(params)

        # Try to get covariance matrix via several possible attributes/methods
        cov = None
        # 1) cov_params() method
        cov_getters = [
            ('cov_params', True),
            ('cov_params_default', True),
            ('normalized_cov_params', False),  # sometimes an attribute or array
            ('cov', False),
            ('covariance', False),
        ]
        for attr, call in cov_getters:
            if hasattr(clustered_res, attr):
                val = getattr(clustered_res, attr)
                try:
                    cov_candidate = val() if call and callable(val) else val() if callable(val) else val
                except Exception:
                    cov_candidate = None
                if cov_candidate is not None:
                    cov = cov_candidate
                    break

        # If cov is still None, try to build from bse if available
        bse = getattr(clustered_res, 'bse', None)
        bse = _ensure_series(bse, index=params.index) if bse is not None else None

        if cov is None:
            if bse is None:
                raise RuntimeError("Could not obtain covariance matrix or standard errors from clustered results.")
            # Build diagonal covariance matrix from bse
            cov = pd.DataFrame(np.diag((bse.values) ** 2), index=bse.index, columns=bse.index)
        else:
            # Convert cov to DataFrame if it's array-like
            if isinstance(cov, np.ndarray):
                cov = pd.DataFrame(cov, index=params.index, columns=params.index)
            elif isinstance(cov, pd.DataFrame):
                # ensure index/columns align with params if possible
                try:
                    # if cov's index/cols are not labeled, set them from params
                    if cov.shape[0] == len(params) and (cov.index is None or len(cov.index) != len(params)):
                        cov.index = params.index
                        cov.columns = params.index
                except Exception:
                    pass
            else:
                # attempt to coerce
                try:
                    cov = pd.DataFrame(cov, index=params.index, columns=params.index)
                except Exception:
                    # fallback to diagonal from bse if possible
                    if bse is None:
                        raise RuntimeError("Could not interpret covariance matrix from clustered results.")
                    cov = pd.DataFrame(np.diag((bse.values) ** 2), index=bse.index, columns=bse.index)

        # At this point, params is a Series and cov is a DataFrame
        # Find base Age_z param name (should be exactly 'Age_z')
        if 'Age_z' not in params.index:
            raise KeyError("Age_z parameter not found in model parameters.")
        age_coef = float(params['Age_z'])

        # Find interaction terms of the form Age_z:C(culture)[T.X] (or similar)
        interaction_terms = [name for name in params.index if name.startswith('Age_z') and 'C(culture)' in name]

        # Extract culture levels from interaction term names (look for 'T.' followed by level)
        levels = set()
        for term in interaction_terms:
            if 'T.' in term:
                try:
                    after_t = term.split('T.', 1)[1]
                    # remove trailing non-alphanumeric characters, keep digits and minus sign
                    level_str = ''.join(ch for ch in after_t if (ch.isdigit() or ch == '-'))
                    if level_str == '':
                        continue
                    levels.add(int(level_str))
                except Exception:
                    continue

        # Determine list of culture levels; if no interactions, assume only reference culture (1)
        if levels:
            max_level = max(levels)
            all_levels = list(range(1, max_level + 1))
        else:
            all_levels = [1]

        summary = {}
        for lev in sorted(all_levels):
            if lev == 1:
                inter_name = None
                inter_coef = 0.0
            else:
                inter_name_candidates = [t for t in interaction_terms if f'T.{lev}' in t]
                inter_name = inter_name_candidates[0] if inter_name_candidates else None
                inter_coef = float(params[inter_name]) if inter_name is not None and inter_name in params.index else 0.0

            slope = age_coef + inter_coef  # log-odds change per 1 SD increase in age for this culture

            # standard error: var(slope) = var(age) + var(inter) + 2*cov(age,inter)
            var_age = float(cov.loc['Age_z', 'Age_z']) if 'Age_z' in cov.index and 'Age_z' in cov.columns else None
            if var_age is None:
                # fallback to bse if available
                if bse is not None and 'Age_z' in bse.index:
                    var_age = float((bse['Age_z']) ** 2)
                else:
                    raise RuntimeError("Cannot determine variance for 'Age_z' from covariance matrix or bse.")

            if inter_name is not None and inter_name in cov.index and inter_name in cov.columns:
                var_inter = float(cov.loc[inter_name, inter_name])
                cov_ai = float(cov.loc['Age_z', inter_name]) if 'Age_z' in cov.index and inter_name in cov.columns else 0.0
            else:
                # fallback to bse if available
                if bse is not None and inter_name is not None and inter_name in bse.index:
                    var_inter = float((bse[inter_name]) ** 2)
                else:
                    var_inter = 0.0
                cov_ai = 0.0

            var_slope = var_age + var_inter + 2.0 * cov_ai
            se_slope = math.sqrt(var_slope) if var_slope > 0 else float('nan')

            z = slope / se_slope if se_slope and not math.isnan(se_slope) else float('nan')
            p = _p_from_z(z) if not math.isnan(z) else float('nan')
            ci_lower = slope - 1.96 * se_slope if not math.isnan(se_slope) else float('nan')
            ci_upper = slope + 1.96 * se_slope if not math.isnan(se_slope) else float('nan')
            signif = (p < 0.05) if not math.isnan(p) else False

            summary[lev] = {
                'slope_logodds_per_Age_z': slope,
                'se': se_slope,
                'z': z,
                'p': p,
                'ci_95_low': ci_lower,
                'ci_95_high': ci_upper,
                'significant_p_lt_0.05': bool(signif),
                'direction': 'increase_with_age' if slope > 0 else ('decrease_with_age' if slope < 0 else 'no_change')
            }
        return summary

    out = {}
    descriptions = []

    # Expecting keys like 'SocialReliance_model' and 'MajorityChoice_model' with ['clustered'] results
    for key in ['SocialReliance_model', 'MajorityChoice_model']:
        if key not in model_output:
            raise KeyError(f"{key} not present in model_output.")
        clustered = model_output[key].get('clustered', None)
        if clustered is None:
            raise KeyError(f"Clustered results not found for {key}.")
        # Summarize model
        try:
            summary = summarize_model(clustered)
        except Exception as e:
            raise RuntimeError(f"Error summarizing {key}: {e}")
        out[key.replace('_model', '')] = summary

        # Build a short human-readable line about the overall Age_z main effect (reference culture = 1)
        ref = summary.get(1)
        if ref:
            desc_line = (f"{key.replace('_model','')}: reference culture (level 1) age slope = "
                         f"{ref['slope_logodds_per_Age_z']:.3f} (SE={ref['se']:.3f}, p={ref['p']:.3g}) -> "
                         f"{'significant' if ref['significant_p_lt_0.05'] else 'ns'}, "
                         f"{ref['direction'].replace('_',' ')}.")
            descriptions.append(desc_line)

    # Compose a concise explanatory description
    full_description = (
        "Extracted per-culture age effects (change in log-odds per 1 SD increase in age) for both outcomes.\n"
        "- 'slope_logodds_per_Age_z' is the model-estimated change in log-odds per standard-deviation increase in age for that culture.\n"
        "- 'se' is the standard error for that slope (accounting for clustering by culture via robust covariance when available).\n"
        "- 'z' and 'p' are the z-statistic and two-sided p-value testing slope != 0.\n"
        "- 'ci_95_low' and 'ci_95_high' give a 95% Wald CI on the slope.\n"
        "- 'significant_p_lt_0.05' flags p < 0.05. 'direction' indicates whether reliance/preferences increase or decrease with age.\n\n"
        "Summary for reference culture and per-culture results:\n" + "\n".join(descriptions)
    )

    return {"object": out, "description": full_description}