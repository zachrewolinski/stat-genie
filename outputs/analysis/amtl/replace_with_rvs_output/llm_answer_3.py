def extract_final_answer(model_output):
    """
    Extracts genus-related coefficients (comparisons to reference 'Homo sapiens')
    from the provided GLM model output and interprets whether modern humans
    (Homo sapiens) have higher AMTL than each non-human genus (Pan, Pongo, Papio).

    Returns:
      {
        "object": pandas.DataFrame with rows for each non-human genus and columns:
                    coef, se, z_or_t, pvalue, ci_lower, ci_upper,
                    odds_ratio, or_ci_lower, or_ci_upper, interpretation
        "description": short textual summary answering the research question
      }
    """
    import numpy as np
    import pandas as pd

    # Obtain the clustered-SE result if available, otherwise fallback to raw fit
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('glm_result_clustered_se') or model_output.get('glm_result')
    else:
        # if a bare result object was passed
        res = model_output

    if res is None:
        raise ValueError("model_output did not contain 'glm_result_clustered_se' or 'glm_result'")

    params = getattr(res, 'params', None)
    if params is None:
        raise ValueError("The result object does not expose .params")

    pvalues = getattr(res, 'pvalues', None)
    bse = getattr(res, 'bse', None)

    # Confidence intervals: try conf_int() method
    try:
        ci = res.conf_int()
        # conf_int returns a DataFrame-like with two columns [0,1] or named
    except Exception:
        ci = None

    # t/z values: try to get tvalues or use coef/bse
    tvals = getattr(res, 'tvalues', None)
    # If not present we will compute z = coef / se (if se available)
    # Parameter names in the model: look for terms that indicate genus dummies
    param_index = list(params.index)
    # Candidate pattern: 'C(genus, Treatment(reference="Homo sapiens"))[T.Pan]' etc.
    # We'll search for substring 'T.Pan', 'T.Pongo', 'T.Papio'
    target_genera = ['Pan', 'Pongo', 'Papio']
    rows = []
    for g in target_genera:
        # find parameter name that ends with 'T.<g>' or contains 'T.<g>'
        match_name = None
        for name in param_index:
            if f"T.{g}" in name:
                match_name = name
                break
        if match_name is None:
            # try a more permissive match (e.g., if encoding produced 'genus[T.Pan]' etc.)
            for name in param_index:
                if f"{g}" in name and 'genus' in name:
                    match_name = name
                    break

        if match_name is None:
            # Parameter not found; record NA row
            rows.append({
                'genus': g,
                'param_name': None,
                'coef': None,
                'se': None,
                'z_or_t': None,
                'pvalue': None,
                'ci_lower': None,
                'ci_upper': None,
                'odds_ratio': None,
                'or_ci_lower': None,
                'or_ci_upper': None,
                'interpretation': f"No coefficient found for genus '{g}' in model results."
            })
            continue

        coef = float(params[match_name])
        se = float(bse[match_name]) if (bse is not None and match_name in bse.index) else None
        pval = float(pvalues[match_name]) if (pvalues is not None and match_name in pvalues.index) else None

        if tvals is not None and match_name in tvals.index:
            z_or_t = float(tvals[match_name])
        else:
            z_or_t = float(coef / se) if (se is not None and se != 0) else None

        # confidence interval
        if ci is not None and match_name in ci.index:
            # ci may be a DataFrame with two columns
            try:
                ci_lower = float(ci.loc[match_name].iloc[0])
                ci_upper = float(ci.loc[match_name].iloc[1])
            except Exception:
                # different shape: try as array
                ci_vals = ci.loc[match_name]
                ci_lower = float(ci_vals[0])
                ci_upper = float(ci_vals[1])
        else:
            ci_lower = None
            ci_upper = None

        # odds ratio and its CI
        try:
            odds_ratio = float(np.exp(coef))
            or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
        except Exception:
            odds_ratio = None
            or_ci_lower = None
            or_ci_upper = None

        # Interpretation relative to Homo sapiens:
        # coef = (genus - Homo) on log-odds scale. If coef < 0 => genus has lower AMTL than humans,
        # which means humans have higher AMTL than that genus. Statistical significance judged by p < 0.05.
        if coef is None or pval is None:
            interp = "Insufficient information to interpret."
        else:
            sig = (pval < 0.05)
            if coef < 0:
                if sig:
                    interp = (f"Homo sapiens has significantly higher AMTL than {g} "
                              f"(coefficient for {g} vs Homo < 0, p = {pval:.3g}).")
                else:
                    interp = (f"Homo sapiens tends to have higher AMTL than {g} (coef < 0), "
                              f"but the difference is not statistically significant (p = {pval:.3g}).")
            elif coef > 0:
                if sig:
                    interp = (f"{g} has significantly higher AMTL than Homo sapiens "
                              f"(coefficient for {g} vs Homo > 0, p = {pval:.3g}).")
                else:
                    interp = (f"{g} tends to have higher AMTL than Homo sapiens (coef > 0), "
                              f"but the difference is not statistically significant (p = {pval:.3g}).")
            else:
                interp = "No difference in estimated AMTL between Homo sapiens and this genus."

        rows.append({
            'genus': g,
            'param_name': match_name,
            'coef': coef,
            'se': se,
            'z_or_t': z_or_t,
            'pvalue': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'odds_ratio': odds_ratio,
            'or_ci_lower': or_ci_lower,
            'or_ci_upper': or_ci_upper,
            'interpretation': interp
        })

    df_out = pd.DataFrame(rows).set_index('genus')

    # Build an overall textual description answering the yes/no research question.
    # We say "Yes" for humans having higher AMTL than a genus only if coef < 0 and p < 0.05.
    conclusions = []
    for g, row in df_out.iterrows():
        if row['param_name'] is None:
            conclusions.append(f"{g}: no parameter found.")
            continue
        if (row['coef'] is not None) and (row['pvalue'] is not None):
            if (row['coef'] < 0) and (row['pvalue'] < 0.05):
                conclusions.append(f"Homo sapiens has significantly higher AMTL than {g}.")
            elif (row['coef'] < 0) and (row['pvalue'] >= 0.05):
                conclusions.append(f"Homo sapiens tends to have higher AMTL than {g}, but not significantly.")
            elif (row['coef'] > 0) and (row['pvalue'] < 0.05):
                conclusions.append(f"{g} has significantly higher AMTL than Homo sapiens.")
            elif (row['coef'] > 0) and (row['pvalue'] >= 0.05):
                conclusions.append(f"{g} tends to have higher AMTL than Homo sapiens, but not significantly.")
            else:
                conclusions.append(f"No clear difference between Homo sapiens and {g}.")
        else:
            conclusions.append(f"Insufficient info for {g}.")

    overall_description = ("Summary of genus comparisons (each vs reference 'Homo sapiens'):\n"
                           + "\n".join(conclusions))

    return {
        "object": df_out,
        "description": overall_description
    }