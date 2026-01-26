def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, test statistics, p-values, 95% CIs,
    and approximate multiplicative effects for the predictors of interest
    (z_age, sex_male, help_yes) from a fitted statsmodels model object.

    Returns:
      dict with keys:
        - "object": dict mapping each predictor to its extracted metrics
        - "description": brief interpretation of the coefficients in context
    """
    import numpy as np
    import pandas as pd
    try:
        from scipy import stats
        normal_cdf = stats.norm.cdf
    except Exception:
        # fallback: use approximated normal CDF from math.erf if scipy not available
        import math
        normal_cdf = lambda x: 0.5 * (1 + math.erf(x / np.sqrt(2)))

    # Predictors of interest
    vars_of_interest = ['z_age', 'sex_male', 'help_yes']

    # Prepare containers
    results = {}

    try:
        res = model_output

        # Extract parameters
        params = getattr(res, 'params', None)
        if params is None:
            # sometimes params stored as attribute .params; error otherwise
            raise ValueError("Model output has no 'params' attribute")

        # Ensure params is a pandas Series for easy indexing
        if not isinstance(params, pd.Series):
            params = pd.Series(params)

        # Standard errors: prefer .bse, else diag of cov_params
        bse = getattr(res, 'bse', None)
        if bse is None:
            cov = getattr(res, 'cov_params', None)
            if callable(cov):
                covmat = cov()
            else:
                covmat = getattr(res, 'cov_params_', None)
            if covmat is None:
                raise ValueError("Cannot obtain standard errors or covariance matrix from model output")
            bse = np.sqrt(np.diag(covmat))
            bse = pd.Series(bse, index=params.index)
        else:
            if not isinstance(bse, pd.Series):
                bse = pd.Series(bse, index=params.index)

        # p-values: try .pvalues, otherwise compute from normal approx (z = coef / se)
        pvals = getattr(res, 'pvalues', None)
        if pvals is None:
            zvals = params / bse
            # two-sided p-value using normal approx
            pvals = 2 * (1 - normal_cdf(np.abs(zvals)))
            pvals = pd.Series(pvals, index=params.index)
            test_stat_name = 'z'
            test_stats = zvals
        else:
            # If pvalues exist, try to derive corresponding test stats if possible
            if not isinstance(pvals, pd.Series):
                pvals = pd.Series(pvals, index=params.index)
            # compute z/t as params / bse (use this as test stat)
            test_stats = params / bse
            test_stat_name = 't/z'

        # Confidence intervals: try .conf_int(), else +/- 1.96*se
        try:
            ci = res.conf_int()
            # conf_int may return DataFrame indexed by param names
            if isinstance(ci, pd.DataFrame):
                ci_lower = ci[0]
                ci_upper = ci[1]
            else:
                # fallback if conf_int returns array-like
                ci = np.asarray(ci)
                ci_lower = pd.Series(ci[:, 0], index=params.index)
                ci_upper = pd.Series(ci[:, 1], index=params.index)
        except Exception:
            ci_lower = params - 1.96 * bse
            ci_upper = params + 1.96 * bse

        # Build results for each variable of interest
        for v in vars_of_interest:
            if v in params.index:
                coef = float(params.loc[v])
                se = float(bse.loc[v]) if v in bse.index else float(np.nan)
                stat = float(test_stats.loc[v]) if v in test_stats.index else float(np.nan)
                pval = float(pvals.loc[v]) if v in pvals.index else float(np.nan)
                ci_l = float(ci_lower.loc[v]) if v in ci_lower.index else float(np.nan)
                ci_u = float(ci_upper.loc[v]) if v in ci_upper.index else float(np.nan)

                # Interpret coefficient on log(1 + nuts_per_minute):
                # approximate multiplicative change in (1 + nuts_per_minute) = exp(coef)-1
                try:
                    multiplicative_change = float(np.exp(coef) - 1.0)
                except Exception:
                    multiplicative_change = float(np.nan)

                results[v] = {
                    'coef': coef,
                    'se': se,
                    test_stat_name: stat,
                    'p_value': pval,
                    'ci_95_lower': ci_l,
                    'ci_95_upper': ci_u,
                    # approximate relative change in (1 + nuts_per_minute)
                    'approx_relative_change': multiplicative_change,
                    'interpretation_note': (
                        "Coefficient is on log(1 + nuts_per_minute). "
                        "exp(coef)-1 gives approximate proportional change in (1 + nuts_per_minute). "
                        "For binary predictors (sex_male, help_yes) this is the change for 1 vs 0; "
                        "for z_age this is per one unit of centered age."
                    )
                }
            else:
                results[v] = {'error': f"Variable '{v}' not found in model parameters."}

        # Compose concise description interpreting direction and significance
        desc_lines = []
        alpha = 0.05
        for v in vars_of_interest:
            entry = results.get(v, {})
            if 'error' in entry:
                desc_lines.append(f"{v}: {entry['error']}")
                continue
            coef = entry['coef']
            pval = entry['p_value']
            ci_l = entry['ci_95_lower']
            ci_u = entry['ci_95_upper']
            rel = entry['approx_relative_change']
            sig = ("statistically significant" if (not np.isnan(pval) and pval < alpha) else "not statistically significant")
            # short human-readable summary
            if v == 'z_age':
                desc = (f"Age (centered): coef={coef:.4f}, p={pval:.3g} ({sig}). "
                        f"95% CI [{ci_l:.4f}, {ci_u:.4f}]. "
                        f"Approx. multiplicative change in (1 + nuts/min) per centered-year: {rel:.3f} (i.e. {rel*100:.1f}% change).")
            elif v == 'sex_male':
                desc = (f"Sex (male vs female): coef={coef:.4f}, p={pval:.3g} ({sig}). "
                        f"95% CI [{ci_l:.4f}, {ci_u:.4f}]. "
                        f"Approx. multiplicative change for males relative to females: {rel:.3f} (i.e. {rel*100:.1f}% change).")
            elif v == 'help_yes':
                desc = (f"Received help (yes vs no): coef={coef:.4f}, p={pval:.3g} ({sig}). "
                        f"95% CI [{ci_l:.4f}, {ci_u:.4f}]. "
                        f"Approx. multiplicative change when help received: {rel:.3f} (i.e. {rel*100:.1f}% change).")
            else:
                desc = f"{v}: coef={coef:.4f}, p={pval:.3g} ({sig})."
            desc_lines.append(desc)

        description = " | ".join(desc_lines)

        return {'object': results, 'description': description}

    except Exception as e:
        # If anything goes wrong, return the error message to help debugging
        return {
            'object': None,
            'description': f"Failed to extract results from model_output: {repr(e)}"
        }