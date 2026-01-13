def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals and interprets
    the effect of name femininity on hurricane fatalities from provided statsmodels results.

    Expects model_output to be a dict containing one or more of the following keys:
      - 'nb_masfem'         : NegativeBinomial GLM with masfem_z (primary)
      - 'nb_gender_mf'      : NegativeBinomial GLM with gender_mf (binary)
      - 'ols_masfem'        : OLS on log_alldeaths with masfem_z (robustness)
      - 'nb_masfem_mturk'   : NegativeBinomial GLM with masfem_mturk_z (sensitivity)

    Returns a dict with keys:
      - "object": dict of extracted numeric results per model
      - "description": brief interpretive summary about whether feminine names are
                       associated with fewer fatalities (support for hypothesis)
    """
    import math
    import numpy as np
    import pandas as pd

    out = {}
    summary_lines = []

    def safe_extract(res, var):
        """Extract stats for parameter var from a statsmodels results object res."""
        info = {}
        try:
            params = res.params
            if var not in params.index:
                info['error'] = f"Variable '{var}' not in model parameters."
                return info
            coef = float(params.loc[var])
            se = float(res.bse.loc[var]) if hasattr(res, 'bse') else float(np.nan)
            pval = float(res.pvalues.loc[var]) if hasattr(res, 'pvalues') else float(np.nan)
            # confidence interval: handle different return types
            try:
                ci = res.conf_int()
                if isinstance(ci, pd.DataFrame):
                    lower, upper = float(ci.loc[var].iloc[0]), float(ci.loc[var].iloc[1])
                else:
                    # numpy array, find index of var in params.index
                    idx = list(params.index).index(var)
                    lower, upper = float(ci[idx, 0]), float(ci[idx, 1])
            except Exception:
                lower, upper = float(np.nan), float(np.nan)

            info.update({
                'coef': coef,
                'se': se,
                'pvalue': pval,
                'ci_lower': lower,
                'ci_upper': upper,
                'nobs': int(res.nobs) if hasattr(res, 'nobs') else None,
            })
        except Exception as e:
            info['error'] = f"Extraction error: {e}"
        return info

    # Helper to create interpretation for NB (log-link count model) and OLS on log outcome
    def interpret_nb(stat):
        if 'error' in stat:
            return stat
        coef = stat['coef']
        p = stat['pvalue']
        ci_lower, ci_upper = stat['ci_lower'], stat['ci_upper']
        irr = math.exp(coef) if not (math.isnan(coef) or abs(coef) > 1e9) else float(np.nan)
        irr_ci_lower = math.exp(ci_lower) if not math.isnan(ci_lower) else float(np.nan)
        irr_ci_upper = math.exp(ci_upper) if not math.isnan(ci_upper) else float(np.nan)
        stat.update({
            'irr': irr,
            'irr_ci_lower': irr_ci_lower,
            'irr_ci_upper': irr_ci_upper
        })
        signif = (p < 0.05) if (not math.isnan(p)) else False
        direction = 'negative' if coef < 0 else ('positive' if coef > 0 else 'null')
        return {
            'coef': coef, 'se': stat['se'], 'pvalue': p,
            'ci_lower': ci_lower, 'ci_upper': ci_upper,
            'irr': irr, 'irr_ci_lower': irr_ci_lower, 'irr_ci_upper': irr_ci_upper,
            'significant_at_0.05': signif,
            'direction': direction,
            'interpretation': (
                f"{'Significant' if signif else 'Not significant'} {direction} association: "
                f"a one-unit increase in predictor multiplies expected death counts by ≈{irr:.3f} "
                f"(95% CI [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}])."
            )
        }

    def interpret_ols_log(stat):
        if 'error' in stat:
            return stat
        coef = stat['coef']
        p = stat['pvalue']
        ci_lower, ci_upper = stat['ci_lower'], stat['ci_upper']
        # For log outcome, approximate percent change: (exp(beta)-1)*100
        pct = (math.exp(coef) - 1) * 100 if not math.isnan(coef) else float('nan')
        pct_ci_lower = (math.exp(ci_lower) - 1) * 100 if not math.isnan(ci_lower) else float('nan')
        pct_ci_upper = (math.exp(ci_upper) - 1) * 100 if not math.isnan(ci_upper) else float('nan')
        signif = (p < 0.05) if (not math.isnan(p)) else False
        direction = 'negative' if coef < 0 else ('positive' if coef > 0 else 'null')
        return {
            'coef': coef, 'se': stat['se'], 'pvalue': p,
            'ci_lower': ci_lower, 'ci_upper': ci_upper,
            'pct_change': pct, 'pct_ci_lower': pct_ci_lower, 'pct_ci_upper': pct_ci_upper,
            'significant_at_0.05': signif,
            'direction': direction,
            'interpretation': (
                f"{'Significant' if signif else 'Not significant'} {direction} association: "
                f"a one-unit increase in predictor is associated with ≈{pct:.2f}% change in (alldeaths+1) "
                f"(95% CI [{pct_ci_lower:.2f}%, {pct_ci_upper:.2f}%])."
            )
        }

    # Primary NB model with continuous femininity
    if 'nb_masfem' in model_output:
        res = model_output['nb_masfem']
        if hasattr(res, 'params'):
            stat = safe_extract(res, 'masfem_z')
            interpreted = interpret_nb(stat) if 'error' not in stat else stat
            out['nb_masfem'] = interpreted
            summary_lines.append(
                "Primary NB (masfem_z): " + (interpreted.get('interpretation') if isinstance(interpreted, dict) else str(interpreted))
            )
        else:
            out['nb_masfem'] = {'error': str(res)}

    # Binary gender NB model
    if 'nb_gender_mf' in model_output:
        res = model_output['nb_gender_mf']
        if hasattr(res, 'params'):
            stat = safe_extract(res, 'gender_mf')
            interpreted = interpret_nb(stat) if 'error' not in stat else stat
            out['nb_gender_mf'] = interpreted
            summary_lines.append(
                "Binary NB (gender_mf: female vs male): " + (interpreted.get('interpretation') if isinstance(interpreted, dict) else str(interpreted))
            )
        else:
            out['nb_gender_mf'] = {'error': str(res)}

    # OLS robustness on log deaths
    if 'ols_masfem' in model_output:
        res = model_output['ols_masfem']
        if hasattr(res, 'params'):
            stat = safe_extract(res, 'masfem_z')
            interpreted = interpret_ols_log(stat) if 'error' not in stat else stat
            out['ols_masfem'] = interpreted
            summary_lines.append(
                "OLS on log deaths (masfem_z): " + (interpreted.get('interpretation') if isinstance(interpreted, dict) else str(interpreted))
            )
        else:
            out['ols_masfem'] = {'error': str(res)}

    # MTurk femininity NB sensitivity
    if 'nb_masfem_mturk' in model_output:
        res = model_output['nb_masfem_mturk']
        if hasattr(res, 'params'):
            stat = safe_extract(res, 'masfem_mturk_z')
            interpreted = interpret_nb(stat) if 'error' not in stat else stat
            out['nb_masfem_mturk'] = interpreted
            summary_lines.append(
                "NB (masfem_mturk_z): " + (interpreted.get('interpretation') if isinstance(interpreted, dict) else str(interpreted))
            )
        else:
            out['nb_masfem_mturk'] = {'error': str(res)}

    # Final high-level conclusion based primarily on the primary NB model if present
    final_conclusion = "No primary NB model found to judge the hypothesis."
    if 'nb_masfem' in out:
        primary = out['nb_masfem']
        if 'error' in primary:
            final_conclusion = f"Could not extract primary NB model results: {primary['error']}"
        else:
            sign = primary['direction']
            sig = primary['significant_at_0.05']
            if sign == 'negative' and sig:
                final_conclusion = (
                    "Evidence supports the hypothesis: more feminine hurricane names are associated "
                    "with significantly fewer fatalities (primary NB model)."
                )
            elif sign == 'negative' and not sig:
                final_conclusion = (
                    "Point estimate in the direction predicted by the hypothesis (feminine -> fewer fatalities), "
                    "but the effect is not statistically significant in the primary NB model."
                )
            elif sign == 'positive' and sig:
                final_conclusion = (
                    "Result contradicts the hypothesis: more feminine names are associated with significantly more fatalities "
                    "(primary NB model)."
                )
            else:
                final_conclusion = (
                    "No evidence of a significant association between name femininity and fatalities in the primary NB model."
                )
    # Put together description
    description = " | ".join(summary_lines + [final_conclusion]) if summary_lines else final_conclusion

    return {"object": out, "description": description}