def extract_final_answer(model_output):
    """
    Extract relevant statistics from the modeling output produced by the provided `model` function.

    Returns a dictionary:
      - "object": nested dict with results for each available model ('ols', 'nb', 'ols_mturk')
                  For each model and for the two focal variables ('masfem_z', 'gender_female')
                  includes: coef, se, pvalue, ci_lower, ci_upper, and model-specific transforms:
                    - For OLS: percent_change = (exp(coef)-1)*100 and CI on percent change
                    - For NB: irr = exp(coef) and IRR CI
                  Also includes a simple "conclusion" for each focal variable:
                    "supports", "contradicts" or "inconclusive" relative to the hypothesis that
                    higher femininity -> higher deaths (i.e., positive effect).
      - "description": short explanation of what the numbers mean in context.
    """
    import numpy as np
    import math

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing 'ols', 'nb', and optionally 'ols_mturk'.")

    focal_vars = ['masfem_z', 'gender_female']
    available_models = [k for k in ['ols', 'nb', 'ols_mturk'] if k in model_output and model_output[k] is not None]

    results = {}

    for m in available_models:
        res = model_output[m]
        model_res = {}
        # get param names
        try:
            param_index = list(res.params.index)
        except Exception:
            # fallback: attempt to read as pandas Series/dict-like
            param_index = list(res.params.keys())

        # get confidence intervals table
        try:
            ci_table = res.conf_int()
        except Exception:
            ci_table = None

        for var in focal_vars:
            if var not in param_index:
                # variable not in model (shouldn't happen), skip
                continue
            try:
                coef = float(res.params[var])
            except Exception:
                coef = float(res.params.loc[var])

            try:
                se = float(res.bse[var])
            except Exception:
                se = float(res.bse.loc[var])

            try:
                pval = float(res.pvalues[var])
            except Exception:
                pval = float(res.pvalues.loc[var])

            # confidence interval robust extraction
            try:
                if hasattr(ci_table, 'loc'):
                    ci_low = float(ci_table.loc[var].iloc[0])
                    ci_high = float(ci_table.loc[var].iloc[1])
                else:
                    # ci_table likely ndarray; find index position
                    idx = param_index.index(var)
                    ci_low = float(ci_table[idx, 0])
                    ci_high = float(ci_table[idx, 1])
            except Exception:
                # fallback to coef +/- 1.96*se if conf_int not available
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se

            entry = {
                'coef': coef,
                'se': se,
                'pvalue': pval,
                'ci_lower': ci_low,
                'ci_upper': ci_high
            }

            # model-specific transforms and interpretation
            if m == 'nb':
                # Negative binomial: coef on log scale -> exponentiate to get incidence rate ratio (IRR)
                irr = math.exp(coef)
                irr_ci_low = math.exp(ci_low)
                irr_ci_high = math.exp(ci_high)
                entry.update({
                    'irr': irr,
                    'irr_ci_lower': irr_ci_low,
                    'irr_ci_upper': irr_ci_high,
                    'interpretation': (
                        "Negative binomial model: coefficient is on log scale. IRR >1 indicates higher expected "
                        "death counts per one-unit increase in the predictor (here: per 1 SD increase for masfem_z; "
                        "gender_female is binary)."
                    )
                })
                # conclusion relative to hypothesis (higher femininity -> higher deaths)
                if pval < 0.05:
                    if coef > 0:
                        conclusion = "supports"
                    else:
                        conclusion = "contradicts"
                else:
                    conclusion = "inconclusive"
                entry['conclusion'] = conclusion

            else:
                # OLS on log(deaths): coefficient is change in log deaths.
                # Convert to approximate percent change: (exp(coef)-1)*100
                pct_change = (math.exp(coef) - 1.0) * 100.0
                pct_ci_low = (math.exp(ci_low) - 1.0) * 100.0
                pct_ci_high = (math.exp(ci_high) - 1.0) * 100.0
                entry.update({
                    'pct_change': pct_change,
                    'pct_ci_lower': pct_ci_low,
                    'pct_ci_upper': pct_ci_high,
                    'interpretation': (
                        "OLS on log(alldeaths): coefficient approximates change in log deaths. "
                        "Exp(coef)-1 gives the proportional change in expected deaths (in percent) "
                        "for a one-unit change in the predictor (here: one SD increase in masfem_z)."
                    )
                })
                if pval < 0.05:
                    if coef > 0:
                        conclusion = "supports"
                    else:
                        conclusion = "contradicts"
                else:
                    conclusion = "inconclusive"
                entry['conclusion'] = conclusion

            model_res[var] = entry

        results[m] = model_res

    # Build a concise description
    description_lines = [
        "Extracted statistics for focal predictors from available models: " + ", ".join(available_models) + ".",
        "For 'masfem_z' (continuous femininity rating, standardized):",
        "- In OLS models: coef is change in log(alldeaths); pct_change = (exp(coef)-1)*100 gives percent change in deaths per 1 SD increase in femininity.",
        "- In Negative Binomial: coef is on log scale; irr = exp(coef) is multiplicative change in expected death counts per 1 SD increase.",
        "For 'gender_female' (binary): reported effects are relative to male-named storms (0 -> 1).",
        "Each focal variable includes coef, standard error, p-value, 95% CI, and a simple conclusion: 'supports' if coef>0 and p<0.05, 'contradicts' if coef<0 and p<0.05, otherwise 'inconclusive'.",
        "Controls included in the models (wind_z, min_z, category_z, year_z) are assumed held constant when interpreting these effects."
    ]
    description = " ".join(description_lines)

    return {
        "object": results,
        "description": description
    }