def extract_final_answer(model_output):
    """
    Extracts statistics about the 'female' effect from a model output dictionary
    and returns a structured summary and an interpretation.

    Returned dictionary keys:
      - "object": dict with numeric values (coef, pvalue, OR, CI, avg_marginal_effect,
                  nobs, pseudo_r2, significant (bool), conclusion (str))
      - "description": human-readable explanation of what these numbers imply
                       about whether banks treat female applicants differently.

    The function is defensive and will try multiple places in model_output to
    find the needed quantities (pvalues, params, odds_ratios, or_conf_int,
    avg_marginal_effect, nobs, mcfadden_pseudo_r2).
    """

    def safe_get(mapping, key):
        if mapping is None:
            return None
        # mapping may be a dict-like or statsmodels result object
        try:
            # if mapping supports __getitem__ or get
            return mapping[key]
        except Exception:
            try:
                # try .get (for pandas/dict-like)
                if hasattr(mapping, "get"):
                    return mapping.get(key)
            except Exception:
                pass
            try:
                # try attribute access for statsmodels objects
                return getattr(mapping, key)
            except Exception:
                return None

    def to_float(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            try:
                # if it's a one-element array/Series
                import numpy as _np  # noqa: F401
                if isinstance(x, (list, tuple)) and len(x) == 1:
                    return float(x[0])
                if hasattr(x, 'item'):
                    return float(x.item())
            except Exception:
                pass
        return None

    def first_not_none(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    # defaults
    female_coef = None
    female_p = None
    female_or = None
    female_or_ci_lower = None
    female_or_ci_upper = None
    avg_marginal_effect = None
    nobs = None
    pseudo_r2 = None

    # Try extracting directly from provided keys if present
    if isinstance(model_output, dict):
        # p-values
        pvals = first_not_none(
            model_output.get('pvalues'),
            safe_get(model_output.get('model_result'), 'pvalues')
        )
        if pvals is not None:
            try:
                # try dict-like access first, then fallback
                if hasattr(pvals, 'get'):
                    female_p = to_float(pvals.get('female', pvals['female'] if 'female' in getattr(pvals, 'keys', lambda: {})() else None))
                    if female_p is None:
                        # fallback to indexing for Series-like objects
                        try:
                            female_p = to_float(pvals['female'])
                        except Exception:
                            female_p = None
                else:
                    female_p = to_float(pvals['female'])
            except Exception:
                try:
                    female_p = to_float(pvals['female'])
                except Exception:
                    female_p = None

        # params / coefficient
        model_res = model_output.get('model_result')
        params = None
        if model_res is not None:
            try:
                params = safe_get(model_res, 'params')
            except Exception:
                params = None
        if params is None:
            params = model_output.get('params')
        if params is not None:
            try:
                if hasattr(params, 'get'):
                    female_coef = to_float(params.get('female', params['female'] if 'female' in getattr(params, 'keys', lambda: {})() else None))
                    if female_coef is None:
                        try:
                            female_coef = to_float(params['female'])
                        except Exception:
                            female_coef = None
                else:
                    female_coef = to_float(params['female'])
            except Exception:
                try:
                    female_coef = to_float(params['female'])
                except Exception:
                    female_coef = None

        # odds ratios and CI
        ors = model_output.get('odds_ratios')
        or_ci = first_not_none(
            model_output.get('or_conf_int'),
            model_output.get('or_ci'),
            model_output.get('or_ci_int')
        )
        if ors is not None:
            try:
                if hasattr(ors, 'get'):
                    female_or = to_float(ors.get('female', ors['female'] if 'female' in getattr(ors, 'keys', lambda: {})() else None))
                    if female_or is None:
                        try:
                            female_or = to_float(ors['female'])
                        except Exception:
                            female_or = None
                else:
                    female_or = to_float(ors['female'])
            except Exception:
                try:
                    female_or = to_float(ors['female'])
                except Exception:
                    female_or = None

        if or_ci is not None:
            # or_ci may be a DataFrame-like with columns ci_lower/ci_upper or index by variable
            # Try multiple access patterns safely
            try:
                # try pandas DataFrame .loc access
                female_or_ci_lower = to_float(or_ci.loc['female'].ci_lower)
                female_or_ci_upper = to_float(or_ci.loc['female'].ci_upper)
            except Exception:
                try:
                    # try dict-like columns
                    lower_col = None
                    upper_col = None
                    if hasattr(or_ci, 'get'):
                        lower_col = safe_get(or_ci, 'ci_lower') or safe_get(or_ci, 'lower') or safe_get(or_ci, 0)
                        upper_col = safe_get(or_ci, 'ci_upper') or safe_get(or_ci, 'upper') or safe_get(or_ci, 1)
                        if lower_col is not None and upper_col is not None:
                            try:
                                if hasattr(lower_col, 'get'):
                                    female_or_ci_lower = to_float(lower_col.get('female', lower_col['female'] if 'female' in getattr(lower_col, 'keys', lambda: {})() else None))
                                else:
                                    female_or_ci_lower = to_float(lower_col['female'])
                            except Exception:
                                try:
                                    female_or_ci_lower = to_float(lower_col['female'])
                                except Exception:
                                    female_or_ci_lower = None
                            try:
                                if hasattr(upper_col, 'get'):
                                    female_or_ci_upper = to_float(upper_col.get('female', upper_col['female'] if 'female' in getattr(upper_col, 'keys', lambda: {})() else None))
                                else:
                                    female_or_ci_upper = to_float(upper_col['female'])
                            except Exception:
                                try:
                                    female_or_ci_upper = to_float(upper_col['female'])
                                except Exception:
                                    female_or_ci_upper = None
                    # try row-like dict: or_ci.get('female') -> [lower, upper]
                    if female_or_ci_lower is None and hasattr(or_ci, 'get'):
                        row = safe_get(or_ci, 'female')
                        if row is None:
                            # Maybe or_ci is dict-of-lists or DataFrame-like where keys are columns and 'female' is index
                            row = None
                        if row is not None:
                            try:
                                if isinstance(row, (list, tuple)):
                                    female_or_ci_lower = to_float(row[0]) if len(row) >= 1 else None
                                    female_or_ci_upper = to_float(row[1]) if len(row) >= 2 else None
                                else:
                                    # row could be a pandas Series
                                    female_or_ci_lower = to_float(row.iloc[0]) if hasattr(row, 'iloc') else to_float(row[0])
                                    female_or_ci_upper = to_float(row.iloc[1]) if hasattr(row, 'iloc') else to_float(row[1])
                            except Exception:
                                pass
                except Exception:
                    pass

        # average marginal effect
        ame = first_not_none(
            model_output.get('avg_marginal_effect_female'),
            model_output.get('avg_marginal_effect'),
            model_output.get('average_marginal_effect_female')
        )
        avg_marginal_effect = to_float(ame)

        # nobs and pseudo-R2
        nobs_val = first_not_none(
            model_output.get('nobs'),
            safe_get(model_res, 'nobs') if 'model_res' in locals() else None
        )
        nobs = to_float(nobs_val)
        pseudo_r2_val = first_not_none(
            model_output.get('mcfadden_pseudo_r2'),
            model_output.get('pseudo_r2'),
            safe_get(model_res, 'mcfadden_pseudo_r2') if 'model_res' in locals() else None
        )
        pseudo_r2 = to_float(pseudo_r2_val)

    # Final interpretation logic
    significance = None
    if female_p is not None:
        try:
            significance = bool(female_p < 0.05)
        except Exception:
            significance = None

    # Build conclusion string
    if significance is True:
        if female_coef is None or female_coef >= 0:
            concl = ("There is statistically significant evidence (p < 0.05) that gender is associated "
                     "with mortgage approval in this model; the estimated effect favors female applicants.")
        else:
            concl = ("There is statistically significant evidence (p < 0.05) that gender is associated "
                     "with mortgage approval in this model; the estimated effect favors male applicants.")
    elif significance is False:
        concl = ("No statistically significant evidence at the 5% level that banks treat female applicants "
                 "differently in approval decisions (female p = "
                 f"{female_p:.4f}).")
    else:
        concl = ("Could not determine statistical significance because p-value for 'female' is missing.")

    # Compose object to return (JSON-serializable)
    result_object = {
        'female_coef': to_float(female_coef),
        'female_pvalue': to_float(female_p),
        'female_odds_ratio': to_float(female_or),
        'female_or_ci_lower': to_float(female_or_ci_lower),
        'female_or_ci_upper': to_float(female_or_ci_upper),
        'avg_marginal_effect_female': to_float(avg_marginal_effect),
        'nobs': int(nobs) if nobs is not None else None,
        'mcfadden_pseudo_r2': to_float(pseudo_r2),
        'significant_at_0.05': significance,
        'conclusion': concl
    }

    # Human-readable description
    # Provide interpretation in probability and odds terms where possible.
    descr_parts = []
    if result_object['female_coef'] is not None:
        descr_parts.append(f"Estimated log-odds coefficient for female = {result_object['female_coef']:.4f}.")
    if result_object['female_odds_ratio'] is not None:
        descr_parts.append(f"Odds ratio = {result_object['female_odds_ratio']:.3f}.")
    if result_object['female_or_ci_lower'] is not None and result_object['female_or_ci_upper'] is not None:
        descr_parts.append(f"95% CI for OR = [{result_object['female_or_ci_lower']:.3f}, {result_object['female_or_ci_upper']:.3f}].")
    if result_object['female_pvalue'] is not None:
        descr_parts.append(f"p-value = {result_object['female_pvalue']:.4f}.")
    if result_object['avg_marginal_effect_female'] is not None:
        ame_pct = result_object['avg_marginal_effect_female'] * 100.0
        descr_parts.append(f"Average marginal effect = {result_object['avg_marginal_effect_female']:.4f} (≈ {ame_pct:.2f} percentage points).")
    if result_object['nobs'] is not None:
        descr_parts.append(f"Sample size = {result_object['nobs']}.")
    if result_object['mcfadden_pseudo_r2'] is not None:
        descr_parts.append(f"McFadden pseudo-R^2 = {result_object['mcfadden_pseudo_r2']:.4f} (model explains little variation).")

    descr_parts.append(concl)

    description = " ".join(descr_parts)

    return {
        "object": result_object,
        "description": description
    }