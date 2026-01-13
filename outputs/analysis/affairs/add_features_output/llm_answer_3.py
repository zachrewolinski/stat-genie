def extract_final_answer(model_output):
    """
    Extracts statistics for the 'Children' predictor from the provided model_output.
    Expected model_output: {'negative_binomial': GLMResultsWrapper, 'poisson': GLMResultsWrapper}
    Returns: dict with keys:
      - "object": dict containing extracted numeric results for each model (coef, se, p, CI, IRR, IRR_CI)
      - "description": human-readable interpretation based primarily on the Negative Binomial result
    """
    import numpy as np

    def _safe_format(value, fmt):
        """Format numeric values safely. Returns 'NA' if value is None or not finite or cannot be formatted."""
        try:
            # convert to float where possible
            val = float(value)
            if not np.isfinite(val):
                return "NA"
            return format(val, fmt)
        except Exception:
            return "NA"

    results = {}
    models_to_check = ['negative_binomial', 'poisson']

    for mname in models_to_check:
        model = model_output.get(mname)
        if model is None:
            continue

        # Prepare container for this model's extracted stats
        stats = {
            'coef': None,
            'std_err': None,
            'p_value': None,
            'ci_lower': None,
            'ci_upper': None,
            'irr': None,         # incidence rate ratio = exp(coef)
            'irr_ci_lower': None,
            'irr_ci_upper': None,
        }

        # Check that 'Children' is in the model parameters
        try:
            params = model.params
            if 'Children' not in params.index:
                # try string variants
                found = [n for n in params.index if n.lower() == 'children']
                if found:
                    key = found[0]
                else:
                    raise KeyError("Variable 'Children' not found in model parameters")
            else:
                key = 'Children'

            coef = None
            try:
                coef = float(params.loc[key])
            except Exception:
                coef = None

            # standard error
            try:
                se = float(model.bse.loc[key])
            except Exception:
                # fallback: compute from cov_params if available
                try:
                    cov = model.cov_params()
                    idx = list(params.index).index(key)
                    se = float(np.sqrt(np.diag(cov).tolist()[idx]))
                except Exception:
                    se = None

            # p-value
            try:
                pval = float(model.pvalues.loc[key])
            except Exception:
                pval = None

            # confidence interval
            try:
                ci = model.conf_int()
                # ci may be a DataFrame or ndarray; try to access by label then fallback by index
                if hasattr(ci, 'loc'):
                    # DataFrame: typically two columns [lower, upper]
                    row = ci.loc[key]
                    # row may be a Series with 0 and 1 or named columns; handle both
                    try:
                        ci_lower = float(row.iloc[0])
                        ci_upper = float(row.iloc[1])
                    except Exception:
                        # try by column names
                        ci_lower = float(row.iloc[0])
                        ci_upper = float(row.iloc[1])
                else:
                    # assume ndarray in same order as params
                    idx = list(params.index).index(key)
                    ci_lower = float(ci[idx, 0])
                    ci_upper = float(ci[idx, 1])
            except Exception:
                ci_lower = None
                ci_upper = None

            # incidence rate ratio and its CI (exp for count models)
            try:
                irr = float(np.exp(coef)) if coef is not None else None
            except Exception:
                irr = None
            try:
                irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            except Exception:
                irr_ci_lower = None
            try:
                irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
            except Exception:
                irr_ci_upper = None

            stats.update({
                'coef': coef,
                'std_err': se,
                'p_value': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'irr': irr,
                'irr_ci_lower': irr_ci_lower,
                'irr_ci_upper': irr_ci_upper
            })

        except KeyError:
            # Variable missing
            stats['error'] = "Variable 'Children' not found in model parameters."

        results[mname] = stats

    # Build an interpretable description focusing on Negative Binomial (preferred)
    desc_lines = []
    if 'negative_binomial' in results:
        nb = results['negative_binomial']
        if 'error' in nb:
            desc_lines.append("Negative Binomial model does not contain variable 'Children'.")
        else:
            coef = nb.get('coef')
            p = nb.get('p_value')
            irr = nb.get('irr')
            ci_low = nb.get('ci_lower')
            ci_high = nb.get('ci_upper')
            irr_ci_low = nb.get('irr_ci_lower')
            irr_ci_high = nb.get('irr_ci_upper')

            # Determine significance robustly
            try:
                sig = (p is not None) and np.isfinite(p) and (p < 0.05)
            except Exception:
                sig = False

            if coef is None:
                desc_lines.append("Could not extract coefficient for 'Children' from Negative Binomial model.")
            else:
                # Percent change interpretation from IRR
                pct_change_str = "NA"
                try:
                    if irr is not None and np.isfinite(irr):
                        pct_change = (irr - 1.0) * 100.0
                        pct_change_str = _safe_format(pct_change, ".2f") + "%"
                except Exception:
                    pct_change_str = "NA"

                coef_str = _safe_format(coef, ".3f")
                p_str = _safe_format(p, ".3g")
                irr_str = _safe_format(irr, ".3f")
                irr_ci_low_str = _safe_format(irr_ci_low, ".3f")
                irr_ci_high_str = _safe_format(irr_ci_high, ".3f")

                if sig:
                    # If statistically significant, state direction
                    if coef < 0:
                        desc_lines.append(
                            f"In the Negative Binomial model, the 'Children' coefficient = {coef_str} "
                            f"(p = {p_str}), IRR = {irr_str} which implies ~{pct_change_str} fewer expected affairs "
                            f"for respondents with children compared to those without. 95% CI for IRR: "
                            f"[{irr_ci_low_str}, {irr_ci_high_str}]."
                        )
                    else:
                        desc_lines.append(
                            f"In the Negative Binomial model, the 'Children' coefficient = {coef_str} "
                            f"(p = {p_str}), IRR = {irr_str} which implies ~{pct_change_str} higher expected affairs "
                            f"for respondents with children compared to those without. 95% CI for IRR: "
                            f"[{irr_ci_low_str}, {irr_ci_high_str}]."
                        )
                else:
                    # not statistically significant
                    desc_lines.append(
                        f"In the Negative Binomial model, the 'Children' coefficient = {coef_str} "
                        f"(p = {p_str}), IRR = {irr_str}. "
                        "This indicates no statistically significant evidence that having children changes the "
                        "expected number of extramarital affairs (at α = 0.05)."
                    )
                    if (irr_ci_low is not None) and (irr_ci_high is not None):
                        irr_ci_low_str = _safe_format(irr_ci_low, ".3f")
                        irr_ci_high_str = _safe_format(irr_ci_high, ".3f")
                        desc_lines.append(
                            f"95% CI for IRR: [{irr_ci_low_str}, {irr_ci_high_str}] — this range includes 1.0, "
                            "consistent with no clear effect."
                        )
    else:
        desc_lines.append("No Negative Binomial model provided in model_output.")

    # Add a short note about Poisson robustness (if available)
    if 'poisson' in results:
        poi = results['poisson']
        if 'error' in poi:
            desc_lines.append("Poisson model does not contain variable 'Children'.")
        else:
            coef = poi.get('coef')
            p = poi.get('p_value')
            irr = poi.get('irr')
            if coef is not None:
                coef_str = _safe_format(coef, ".3f")
                p_str = _safe_format(p, ".3g")
                irr_str = _safe_format(irr, ".3f")
                desc_lines.append(
                    f"Robustness check (Poisson): coef = {coef_str}, p = {p_str}, IRR = {irr_str}."
                )

    description = " ".join(desc_lines)

    # Final return object: include the numeric extractions and the description
    return {
        "object": results,
        "description": description
    }