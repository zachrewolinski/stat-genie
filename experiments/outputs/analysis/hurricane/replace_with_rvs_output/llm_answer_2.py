def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, p-values, and 95% CIs for the primary
    independent variables (masfem_s and FemaleName) from the model_output dict.
    Returns a dictionary with:
      - "object": structured numeric results for Negative Binomial and OLS models
      - "description": short, plain-language interpretation of the key coefficients
    """
    import numpy as np
    import math

    def safe_extract(res, var):
        """Safely extract stats from a statsmodels results object for variable var."""
        out = {'present': False}
        if res is None:
            return out
        try:
            params = res.params
            if var not in params.index:
                return out
            out['present'] = True
            coef = float(params[var])
            # bse, pvalues, conf_int might be attributes or methods depending on wrapper;
            # use try/except to handle.
            try:
                se = float(res.bse[var])
            except Exception:
                se = None
            try:
                pval = float(res.pvalues[var])
            except Exception:
                pval = None
            try:
                ci = res.conf_int()
                # conf_int returns DataFrame-like; get rows by var
                ci_low = float(ci.loc[var, 0])
                ci_upp = float(ci.loc[var, 1])
            except Exception:
                ci_low = None
                ci_upp = None

            out.update({
                'coef': coef,
                'std_err': se,
                'p_value': pval,
                'ci_95_lower': ci_low,
                'ci_95_upper': ci_upp
            })
        except Exception:
            # fallback: return minimal info if extraction fails
            out['error'] = 'could not extract'
        return out

    results = {}

    # Get models from model_output safely
    nb = model_output.get('nb_model', None)
    ols = model_output.get('ols_log_deaths', None)

    # Variables of interest
    vars_of_interest = ['masfem_s', 'FemaleName']

    # Extract from Negative Binomial (alldeaths)
    nb_results = {}
    for v in vars_of_interest:
        nb_results[v] = safe_extract(nb, v)
        # compute IRR and IRR CIs if coef and CIs available
        if nb_results[v].get('present'):
            try:
                coef = nb_results[v]['coef']
                ci_low = nb_results[v]['ci_95_lower']
                ci_upp = nb_results[v]['ci_95_upper']
                irr = math.exp(coef)
                irr_low = math.exp(ci_low) if ci_low is not None else None
                irr_upp = math.exp(ci_upp) if ci_upp is not None else None
                nb_results[v].update({
                    'irr': irr,
                    'irr_95_lower': irr_low,
                    'irr_95_upper': irr_upp
                })
            except Exception:
                pass

    # Extract from OLS on log1p_alldeaths
    ols_results = {}
    for v in vars_of_interest:
        ols_results[v] = safe_extract(ols, v)
        # for log model, compute approximate percent-change: (exp(coef)-1)*100
        if ols_results[v].get('present'):
            try:
                coef = ols_results[v]['coef']
                pct_change = (math.exp(coef) - 1) * 100.0
                ci_low = ols_results[v]['ci_95_lower']
                ci_upp = ols_results[v]['ci_95_upper']
                pct_low = (math.exp(ci_low) - 1) * 100.0 if ci_low is not None else None
                pct_upp = (math.exp(ci_upp) - 1) * 100.0 if ci_upp is not None else None
                ols_results[v].update({
                    'approx_pct_change_in_1plus_deaths': pct_change,
                    'pct_change_95_lower': pct_low,
                    'pct_change_95_upper': pct_upp
                })
            except Exception:
                pass

    results['negative_binomial_alldeaths'] = nb_results
    results['ols_log1p_alldeaths'] = ols_results
    results['used_covariates'] = model_output.get('used_covariates', None)
    results['formula_nb'] = model_output.get('formula_nb', None)
    results['formula_ols'] = model_output.get('formula_ols', None)

    # Build a concise interpretation focused on the hypothesis.
    # We'll examine sign and statistical significance (p < 0.05) for masfem_s primarily,
    # and also report FemaleName as a complementary check.
    def interpret_entry(entry, model_name, varname):
        if not entry.get('present'):
            return f"{model_name}: {varname} not present in the model."
        coef = entry.get('coef')
        p = entry.get('p_value')
        irr = entry.get('irr') if 'irr' in entry else None
        pct = entry.get('approx_pct_change_in_1plus_deaths') if 'approx_pct_change_in_1plus_deaths' in entry else None
        sig = (p is not None) and (p < 0.05)
        sign = 'positive' if coef > 0 else ('zero' if coef == 0 else 'negative')
        lines = []
        if model_name.startswith('Negative Binomial'):
            lines.append(f"{model_name}: {varname} coef = {coef:.4f}, SE ≈ {entry.get('std_err'):.4f} "
                         f"(p = {p:.3g}). IRR = {irr:.3f} (95% CI: {entry.get('irr_95_lower'):.3f} - {entry.get('irr_95_upper'):.3f}).")
            if sig:
                lines.append(f"Interpretation: Statistically significant ({p:.3g}); a 1 SD increase in {varname} is associated with a { (irr-1)*100:.1f}% change in expected death counts.")
            else:
                lines.append(f"Interpretation: Not statistically significant (p = {p:.3g}); no strong evidence of an effect on deaths.")
        else:
            # OLS on log(1+deaths)
            lines.append(f"{model_name}: {varname} coef = {coef:.4f}, SE ≈ {entry.get('std_err'):.4f} (p = {p:.3g}).")
            if pct is not None:
                lines.append(f"Approx. percent change in (1+deaths): {pct:.1f}% (95% CI: {entry.get('pct_change_95_lower'):.1f}% - {entry.get('pct_change_95_upper'):.1f}%).")
            if sig:
                lines.append(f"Interpretation: Statistically significant ({p:.3g}); direction = {sign}.")
            else:
                lines.append(f"Interpretation: Not statistically significant (p = {p:.3g}); no strong evidence of an effect on deaths.")
        return " ".join(lines)

    # Compose human-readable description focusing on masfem_s then FemaleName
    descriptions = []
    descriptions.append(interpret_entry(nb_results['masfem_s'], 'Negative Binomial (alldeaths)', 'masfem_s'))
    descriptions.append(interpret_entry(ols_results['masfem_s'], 'OLS on log1p_alldeaths', 'masfem_s'))
    descriptions.append(interpret_entry(nb_results['FemaleName'], 'Negative Binomial (alldeaths)', 'FemaleName'))
    descriptions.append(interpret_entry(ols_results['FemaleName'], 'OLS on log1p_alldeaths', 'FemaleName'))

    description_text = ("Summary of coefficients for name femininity measures. "
                        + "Below are model-specific results for masfem_s (continuous perceived femininity) "
                        + "and FemaleName (binary female-coded name). "
                        + "Statistical significance uses p < 0.05 threshold. "
                        + "Interpretations indicate whether higher femininity is associated with higher deaths (consistent with fewer precautions). "
                        + "Detailed numeric results follow:\n\n"
                        + "\n\n".join(descriptions))

    return {
        "object": results,
        "description": description_text
    }